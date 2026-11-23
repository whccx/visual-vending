"""在柜体附近建立手部和商品轨迹，再跟踪实际检测到的候选。

所有关联均为二维启发式判断；漏检帧不会凭空生成位置。
"""

import math

from app import config


def box_distance(a, b):
    """计算两个矩形框之间的最短欧氏距离，重叠时返回零。"""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _center(box):
    """返回左上、右下坐标定义的矩形框中心点。"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(a, b):
    """计算两个矩形框的交并比，空框使用极小分母避免除零。"""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + bb - inter)


def _box_area(box):
    """返回有效矩形框面积，退化框面积为零。"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _linked_hands(candidate, hands, ids):
    """返回候选关联的手部轨迹编号及手框，忽略无效索引。"""
    linked = []
    for index in candidate.get("hand_indices", []):
        if isinstance(index, int) and not isinstance(index, bool) and index in ids:
            linked.append((ids[index], hands[index]["box"]))
    return linked


def _cosine(a, b):
    """计算两个特征向量的余弦相似度；范数或结果无效时返回 None。"""
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0 or not np.isfinite(norm):
        return None
    value = float(np.dot(a, b) / norm)
    return value if math.isfinite(value) else None


def filter_hand_linked_candidates(
    hands_by_frame,
    candidates_by_frame,
    areas_by_frame,
    near_ratio=0.10,
    max_gap=5,
    embeddings_by_frame=None,
    max_linked_hand_area_ratio=None,
    require_departure=False,
    hand_track_base_distance=70.0,
    hand_track_iou_threshold=None,
):
    """筛选从柜体附近建立并确认的手部关联商品轨迹。

    参数：
        hands_by_frame: 每帧检测到的手框列表。
        candidates_by_frame: 每帧商品候选，候选可通过 hand_indices 关联手框。
        areas_by_frame: 每帧柜体矩形框，未检测到时为 None。
        near_ratio: 判断手或商品接近柜体时使用的柜体对角线比例。
        max_gap: 延续轨迹允许的最大帧间隔。
        embeddings_by_frame: 可选的、以从 1 开始的帧号索引的外观向量。
        max_linked_hand_area_ratio: 商品框相对关联手框的最大面积倍数；None 时不筛选。
        require_departure: 是否仅输出已由同一只手证实离开货架的商品轨迹。
        hand_track_base_distance: 相邻帧手中心位移的基础距离上限。
        hand_track_iou_threshold: 中心位移超限时，上一手框与当前手框的 IoU 兜底阈值；None 时禁用。
    返回：
        与输入帧一一对应的已确认商品候选，以及门控过程的统计字典。
    异常：
        手部、商品和柜体帧数不一致时抛出 ValueError。

    手和商品都接近柜体时才建立种子；三个帧间隔内出现两次种子后确认轨迹。
    可选的离架模式还要求同一只关联手连续持物远离货架，避免柜外物体被误选。
    """
    if not (len(hands_by_frame) == len(candidates_by_frame) == len(areas_by_frame)):
        raise ValueError("手部、商品与柜内区域帧数不一致")
    hands_tracks = []
    frame_hand_ids = []
    for frame_index, (hands, area) in enumerate(zip(hands_by_frame, areas_by_frame)):
        centers = [_center(hand["box"]) for hand in hands]
        distance_proposals = []
        iou_proposals = []
        for track_index, track in enumerate(hands_tracks):
            gap = frame_index - track["last_frame"]
            if not 1 <= gap <= max_gap:
                continue
            limit = min(
                220.0,
                max(
                    hand_track_base_distance + 30.0 * (gap - 1),
                    1.5 * track["speed"] * gap + 10.0,
                ),
            )
            for hand_index, hand in enumerate(hands):
                distance = math.dist(track["center"], centers[hand_index])
                if distance <= limit:
                    distance_proposals.append((distance, track_index, hand_index))
                    continue
                if hand_track_iou_threshold is None:
                    continue
                overlap = _iou(track["box"], hand["box"])
                if overlap >= hand_track_iou_threshold:
                    iou_proposals.append((-overlap, track_index, hand_index))
        proposals = sorted(distance_proposals) + sorted(iou_proposals)
        assigned_tracks, assigned_hands, ids = set(), set(), {}
        for _, track_index, hand_index in sorted(proposals):
            if track_index in assigned_tracks or hand_index in assigned_hands:
                continue
            assigned_tracks.add(track_index)
            assigned_hands.add(hand_index)
            ids[hand_index] = track_index
        for hand_index, hand in enumerate(hands):
            if hand_index not in ids:
                ids[hand_index] = len(hands_tracks)
                hands_tracks.append(
                    {
                        "last_frame": frame_index,
                        "box": hand["box"],
                        "center": centers[hand_index],
                        "speed": 0.0,
                        "first_near": None,
                    }
                )
            track = hands_tracks[ids[hand_index]]
            gap = frame_index - track["last_frame"]
            if gap > 0:
                track["speed"] = math.dist(track["center"], centers[hand_index]) / gap
            track["last_frame"] = frame_index
            track["box"] = hand["box"]
            track["center"] = centers[hand_index]
            if area is not None:
                diagonal = math.hypot(area[2] - area[0], area[3] - area[1])
                if (
                    diagonal > 0
                    and box_distance(hand["box"], area) <= near_ratio * diagonal
                    and track["first_near"] is None
                ):
                    track["first_near"] = frame_index
        frame_hand_ids.append(ids)

    filtered_candidates = []
    area_ratio_rejected = 0
    unlinked_rejected = 0
    for hands, candidates, ids in zip(hands_by_frame, candidates_by_frame, frame_hand_ids):
        kept = []
        for candidate_index, candidate in enumerate(candidates):
            linked = _linked_hands(candidate, hands, ids)
            if max_linked_hand_area_ratio is None:
                kept.append((candidate_index, candidate, linked))
                continue
            if not linked:
                unlinked_rejected += 1
                continue
            hand_area = max(_box_area(hand) for _, hand in linked)
            if hand_area <= 0 or _box_area(candidate["box"]) > (
                max_linked_hand_area_ratio * hand_area
            ):
                area_ratio_rejected += 1
                continue
            kept.append((candidate_index, candidate, linked))
        filtered_candidates.append(kept)

    products = []
    seed_candidates = 0
    for frame_index, (candidates, area) in enumerate(
        zip(filtered_candidates, areas_by_frame)
    ):
        used = set()
        embedding = (embeddings_by_frame or {}).get(frame_index + 1)
        for candidate_index, candidate, linked_hands in candidates:
            box = candidate["box"]
            position = _center(box)
            linked_ids = {track_id for track_id, _ in linked_hands}
            diagonal = (
                math.hypot(area[2] - area[0], area[3] - area[1]) if area is not None else 0.0
            )
            near_ids = [
                track_id
                for track_id, _ in linked_hands
                if area is not None
                and box_distance(box, area) <= near_ratio * diagonal
                and hands_tracks[track_id]["first_near"] is not None
                and hands_tracks[track_id]["first_near"] <= frame_index
            ]
            seed_hand_id = min(
                near_ids,
                key=lambda track_id: next(
                    box_distance(box, hand)
                    for linked_id, hand in linked_hands
                    if linked_id == track_id
                ),
                default=None,
            )
            direct = seed_hand_id is not None
            seed_candidates += direct
            matches = []
            for track_index, track in enumerate(products):
                gap = frame_index - track["last_frame"]
                if not 1 <= gap <= max_gap or track_index in used:
                    continue
                if track["hand_track_id"] not in linked_ids:
                    continue
                prev = track["box"]
                size = _box_area(box)
                prev_size = _box_area(prev)
                if min(size, prev_size) <= 0 or max(size, prev_size) / min(size, prev_size) > 3.5:
                    continue
                velocity = track["velocity"]
                predicted = (
                    track["center"][0] + velocity[0] * gap,
                    track["center"][1] + velocity[1] * gap,
                )
                distance = math.dist(position, predicted)
                diagonal = math.hypot(prev[2] - prev[0], prev[3] - prev[1])
                limit = min(
                    250.0, max(40.0, 0.65 * diagonal, 1.5 * math.hypot(*velocity) * gap + 10.0)
                )
                overlap = _iou(box, prev)
                if distance > limit or (overlap < 0.05 and distance > 0.6 * limit):
                    continue
                if embedding is not None and track["embedding"] is not None:
                    similarity = _cosine(embedding, track["embedding"])
                    if similarity is None or similarity < 0.60:
                        continue
                elif overlap < 0.05:
                    # 缺少外观特征时，不关联互不重叠的商品框。
                    continue
                matches.append((distance / max(1.0, limit), track_index))
            matches.sort()
            # 多条候选轨迹代价接近时，仅凭位置无法安全关联。
            ambiguous = len(matches) > 1 and matches[1][0] - matches[0][0] < 0.05
            if matches and not ambiguous:
                track_index = matches[0][1]
                track = products[track_index]
                gap = frame_index - track["last_frame"]
                track["velocity"] = (
                    (position[0] - track["center"][0]) / gap,
                    (position[1] - track["center"][1]) / gap,
                )
                track.update(last_frame=frame_index, box=box, center=position, embedding=embedding)
            elif direct and not ambiguous:
                track_index = len(products)
                track = {
                    "last_frame": frame_index,
                    "box": box,
                    "center": position,
                    "velocity": (0.0, 0.0),
                    "embedding": embedding,
                    "hand_track_id": seed_hand_id,
                    "seeds": [],
                    "observations": [],
                    "near_distance": None,
                    "last_distance": None,
                    "outward_steps": 0,
                    "product_steps": 0,
                    "taken": False,
                }
                products.append(track)
            else:
                continue
            used.add(track_index)
            track["observations"].append((frame_index, candidate_index, direct))
            if direct:
                track["seeds"].append(frame_index)
            if require_departure and area is not None:
                hand = next(
                    hand for track_id, hand in linked_hands if track_id == track["hand_track_id"]
                )
                hand_distance = box_distance(hand, area)
                if hand_distance <= near_ratio * diagonal:
                    track["near_distance"] = (
                        hand_distance
                        if track["near_distance"] is None
                        else min(track["near_distance"], hand_distance)
                    )
                    track["last_distance"] = hand_distance
                    track["outward_steps"] = 0
                    track["product_steps"] = 0
                elif track["near_distance"] is not None and track["last_distance"] is not None:
                    if hand_distance > track["last_distance"]:
                        track["outward_steps"] += 1
                        track["product_steps"] += 1
                    else:
                        track["outward_steps"] = 0
                        track["product_steps"] = 0
                    track["last_distance"] = hand_distance
                    if (
                        track["outward_steps"] >= config.taken_min_outward_steps()
                        and track["product_steps"] >= config.taken_min_product_frames()
                        and hand_distance - track["near_distance"]
                        >= config.taken_min_distance_gain_ratio() * diagonal
                    ):
                        track["taken"] = True

    filtered = [[] for _ in candidates_by_frame]
    confirmed = 0
    taken_tracks = 0
    not_taken_tracks = 0
    continuation = 0
    for track_index, track in enumerate(products):
        if not any(b - a <= 3 for a, b in zip(track["seeds"], track["seeds"][1:])):
            continue
        confirmed += 1
        if require_departure and not track["taken"]:
            not_taken_tracks += 1
            continue
        taken_tracks += 1
        for frame_index, candidate_index, direct in track["observations"]:
            candidate = dict(candidates_by_frame[frame_index][candidate_index])
            candidate["area_track_id"] = track_index
            filtered[frame_index].append((candidate_index, candidate))
            continuation += not direct
    filtered = [
        [candidate for _, candidate in sorted(frame, key=lambda entry: entry[0])]
        for frame in filtered
    ]
    total = sum(map(len, candidates_by_frame))
    kept = sum(map(len, filtered))
    return filtered, {
        "applied": True,
        "frames": len(areas_by_frame),
        "area_frames": sum(area is not None for area in areas_by_frame),
        "hand_frames": sum(bool(hands) for hands in hands_by_frame),
        "hand_tracks": len(hands_tracks),
        "near_hand_tracks": sum(t["first_near"] is not None for t in hands_tracks),
        "candidate_count": total,
        "area_ratio_rejected": area_ratio_rejected,
        "unlinked_rejected": unlinked_rejected,
        "seed_candidates": seed_candidates,
        "confirmed_product_tracks": confirmed,
        "taken_tracks": taken_tracks,
        "not_taken_tracks": not_taken_tracks,
        "continuation_candidates": continuation,
        "eligible_candidates": kept,
        "near_ratio": near_ratio,
    }
