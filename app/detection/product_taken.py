"""检测手是否携带商品连续离开货架。"""

import math
import statistics

import cv2
import numpy as np

from app import config


def _area(box):
    """返回矩形框面积，退化框面积为零。"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _mask_bbox(mask):
    """返回掩码的外接矩形框。"""
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _area_distance_map(frame):
    """返回帧内背景像素到区域掩码的距离图；无掩码时返回 None。

    掩码外接矩形会把柜内地面、人体倒影一并划入区域，导致手携商品
    离开货架后仍"贴着区域"；以掩码本身为参考才能度量真实的离架位移。
    距离图惰性计算并缓存在帧上。
    """
    dist = frame.get("_area_dist")
    if dist is not None:
        return dist
    mask = frame.get("area_mask")
    if mask is None or not mask.any():
        return None
    dist = cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 3)
    frame["_area_dist"] = dist
    return dist


def _hand_area_distance(frame, hand_box, area_box):
    """返回手框到区域参考的距离：优先掩码距离图，无掩码时退回矩形框。"""
    dist = _area_distance_map(frame)
    if dist is None:
        return _box_distance(hand_box, area_box)
    height, width = dist.shape
    x1 = max(0, min(width, int(hand_box[0])))
    y1 = max(0, min(height, int(hand_box[1])))
    x2 = max(0, min(width, int(hand_box[2])))
    y2 = max(0, min(height, int(hand_box[3])))
    if x2 <= x1 or y2 <= y1:
        return float(dist.min())
    return float(dist[y1:y2, x1:x2].min())


def _center(box):
    """返回矩形框中心点。"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_distance(first, second):
    """返回两个矩形框的最短距离，重叠时为零。"""
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def _near_hand(candidate, hand):
    """判断商品候选是否落在扩展后的手部邻域。"""
    x1, y1, x2, y2 = hand
    width, height = x2 - x1, y2 - y1
    return (
        candidate[2] >= x1 - 0.25 * width
        and candidate[0] <= x2 + 0.25 * width
        and candidate[3] >= y1 - 0.25 * height
        and candidate[1] <= y2 + 0.25 * height
    )


def _iou(first, second):
    """返回两个矩形框的交并比。"""
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    return intersection / max(1e-9, _area(first) + _area(second) - intersection)


def _dedupe_hands(hands):
    """合并四个旋转方向检测到的同一只手的重叠框。"""
    unique = []
    for hand in hands:
        if not any(_iou(hand["box"], existing["box"]) >= 0.35 for existing in unique):
            unique.append(hand)
    return unique


def _fill_missing_areas(frames):
    """补全单帧货架漏检：矩形框用整段中位数，掩码用时间最近邻。"""
    areas = [frame["area"] for frame in frames if frame.get("area") is not None]
    stable_area = [statistics.median(box[index] for box in areas) for index in range(4)] if areas else None
    forward = []
    last_mask = None
    for frame in frames:
        if frame.get("area_mask") is None or not frame["area_mask"].any():
            frame["area_mask"] = last_mask
        else:
            last_mask = frame["area_mask"]
        forward.append(frame["area_mask"])
    fill = None
    for index in range(len(frames) - 1, -1, -1):
        if forward[index] is None:
            forward[index] = fill
        else:
            fill = forward[index]
    return [
        dict(
            frame,
            area_mask=mask,
            area=frame.get("area")
            or (_mask_bbox(mask) if mask is not None else None)
            or stable_area,
        )
        for frame, mask in zip(frames, forward)
    ]


def _assign_hand_tracks(tracks, hands, frame_index, max_gap):
    """将当前手框按短时中心点距离关联到已有轨迹。"""
    proposals = []
    centers = [_center(hand["box"]) for hand in hands]
    for track_index, track in enumerate(tracks):
        gap = frame_index - track["last_frame"]
        if not 1 <= gap <= max_gap:
            continue
        limit = min(450.0, max(100.0, 45.0 * gap, track["speed"] * gap * 1.5 + 10.0))
        for hand_index, center in enumerate(centers):
            distance = math.dist(track["center"], center)
            if distance <= limit:
                proposals.append((distance, track_index, hand_index))

    matched_tracks, matched_hands, assignment = set(), set(), {}
    for _, track_index, hand_index in sorted(proposals):
        if track_index in matched_tracks or hand_index in matched_hands:
            continue
        matched_tracks.add(track_index)
        matched_hands.add(hand_index)
        assignment[hand_index] = track_index

    for hand_index, center in enumerate(centers):
        if hand_index in assignment:
            continue
        assignment[hand_index] = len(tracks)
        tracks.append(
            {
                "last_frame": frame_index,
                "center": center,
                "speed": 0.0,
                "near_area": False,
                "near_distance": None,
                "last_distance": None,
                "outward_steps": 0,
                "product_steps": 0,
            }
        )
    return assignment, centers


def _has_sparse_departure_event(
    frames,
    near_area_ratio,
    min_distance_gain_ratio,
    min_product_frames,
    max_gap,
):
    """以稀疏手部证据确认先近货架、后持物离架的事件。"""
    near_frames = []
    far_frames = []
    for frame_index, frame in enumerate(frames):
        area = frame.get("area")
        if area is None or _area(area) <= 0:
            continue
        diagonal = math.hypot(area[2] - area[0], area[3] - area[1])
        if diagonal <= 0:
            continue
        mask_metric = frame.get("area_mask") is not None
        near_limit = (
            config.taken_mask_near_distance() if mask_metric else near_area_ratio * diagonal
        )
        gain_limit = (
            config.taken_mask_gain_distance() if mask_metric else min_distance_gain_ratio * diagonal
        )
        product_hands = [
            hand["box"]
            for hand in frame.get("hands") or []
            if any(_near_hand(product, hand["box"]) for product in frame.get("products") or [])
        ]
        if not product_hands:
            continue
        distances = [_hand_area_distance(frame, hand, area) for hand in product_hands]
        if any(distance <= near_limit for distance in distances):
            near_frames.append((frame_index, min(distances)))
            continue
        if not near_frames:
            continue
        if len({near_frame for near_frame, _ in near_frames}) < min_product_frames:
            continue
        seed_index, seed_distance = near_frames[0]
        if frame_index - seed_index > max_gap:
            continue
        if any(distance - seed_distance >= gain_limit for distance in distances):
            far_frames.append(frame_index)
            if len(set(far_frames)) >= min_product_frames:
                return True
    return False


def _has_degenerate_area_departure_event(
    frames,
    near_area_ratio,
    min_distance_gain_ratio,
    min_product_frames,
    max_gap,
):
    """在全高异常货架框中，以同手持物位移判断离架。"""
    seed = None
    seed_frames = set()
    far_frames = set()
    last_frame = None
    last_center = None
    for frame_index, frame in enumerate(frames):
        area = frame.get("area")
        if area is None or _area(area) <= 0:
            continue
        width, height = area[2] - area[0], area[3] - area[1]
        if width <= 0 or height < 1.5 * width:
            continue
        diagonal = math.hypot(width, height)
        candidates = [
            (hand["box"], _center(hand["box"]))
            for hand in frame.get("hands") or []
            if any(_near_hand(product, hand["box"]) for product in frame.get("products") or [])
        ]
        if not candidates:
            continue
        if seed is None:
            near_candidates = [
                center
                for hand, center in candidates
                if _hand_area_distance(frame, hand, area) <= near_area_ratio * diagonal
            ]
            if not near_candidates:
                continue
            seed = near_candidates[0]
            seed_frames.add(frame_index)
            last_frame = frame_index
            last_center = seed
            continue
        if frame_index - last_frame > max_gap:
            seed = candidates[0]
            seed_frames = {frame_index}
            far_frames = set()
            last_frame = frame_index
            last_center = seed
            continue
        center = min(candidates, key=lambda candidate: math.dist(candidate[1], last_center))[1]
        last_frame = frame_index
        last_center = center
        if math.dist(center, seed) < min_distance_gain_ratio * diagonal:
            seed_frames.add(frame_index)
            continue
        if len(seed_frames) < min_product_frames:
            continue
        far_frames.add(frame_index)
        if len(far_frames) >= min_product_frames:
            return True
    return False


def is_product_taken(
    frames,
    *,
    near_area_ratio=None,
    min_distance_gain_ratio=None,
    min_outward_steps=None,
    min_product_frames=None,
    max_hand_gap=None,
    sparse_event_max_gap=None,
):
    """判断是否同一只手携带商品连续离开货架。

    `frames` 的每项含有 `area`、`hands` 和 `products`：货架和商品是矩形框，
    手项必须含 `box`。手轨迹必须先接近货架，随后距离连续增加；同时只有
    外移阶段连续出现手旁商品时才返回 True。静止持物、靠近货架或单帧误检都
    不会产生拿走事件。
    """
    near_area_ratio = config.taken_near_area_ratio() if near_area_ratio is None else near_area_ratio
    min_distance_gain_ratio = (
        config.taken_min_distance_gain_ratio()
        if min_distance_gain_ratio is None
        else min_distance_gain_ratio
    )
    min_outward_steps = (
        config.taken_min_outward_steps() if min_outward_steps is None else min_outward_steps
    )
    min_product_frames = (
        config.taken_min_product_frames() if min_product_frames is None else min_product_frames
    )
    max_hand_gap = config.taken_hand_max_gap() if max_hand_gap is None else max_hand_gap
    sparse_event_max_gap = (
        config.taken_sparse_event_max_gap()
        if sparse_event_max_gap is None
        else sparse_event_max_gap
    )
    tracks = []
    for frame_index, frame in enumerate(frames):
        area = frame.get("area")
        hands = frame.get("hands") or []
        products = frame.get("products") or []
        assignment, centers = _assign_hand_tracks(tracks, hands, frame_index, max_hand_gap)
        if area is None or _area(area) <= 0:
            continue
        diagonal = math.hypot(area[2] - area[0], area[3] - area[1])
        if diagonal <= 0:
            continue
        # 掩码外接矩形几乎全帧，比例阈值（0.1×对角线）会把有限的外移距离吞掉；
        # 掩码度量改用固定像素距离，矩形框度量保持原比例。
        mask_metric = frame.get("area_mask") is not None
        near_limit = (
            config.taken_mask_near_distance() if mask_metric else near_area_ratio * diagonal
        )
        gain_limit = (
            config.taken_mask_gain_distance() if mask_metric else min_distance_gain_ratio * diagonal
        )
        for hand_index, hand in enumerate(hands):
            track = tracks[assignment[hand_index]]
            center = centers[hand_index]
            gap = frame_index - track["last_frame"]
            if gap > 0:
                track["speed"] = math.dist(track["center"], center) / gap
            track["last_frame"] = frame_index
            track["center"] = center
            distance = _hand_area_distance(frame, hand["box"], area)
            nearby_product = any(_near_hand(candidate, hand["box"]) for candidate in products)
            if distance <= near_limit:
                track["near_area"] = True
                track["near_distance"] = (
                    distance
                    if track["near_distance"] is None
                    else min(track["near_distance"], distance)
                )
                track["last_distance"] = distance
                track["outward_steps"] = 0
                track["product_steps"] = 0
                continue
            if not track["near_area"] or track["last_distance"] is None:
                track["last_distance"] = distance
                continue
            if distance > track["last_distance"]:
                track["outward_steps"] += 1
                track["product_steps"] = track["product_steps"] + 1 if nearby_product else 0
            else:
                track["outward_steps"] = 0
                track["product_steps"] = 0
            track["last_distance"] = distance
            gain = distance - track["near_distance"]
            if (
                track["outward_steps"] >= min_outward_steps
                and track["product_steps"] >= min_product_frames
                and gain >= gain_limit
            ):
                return True
    if _has_sparse_departure_event(
        frames,
        near_area_ratio,
        min_distance_gain_ratio,
        min_product_frames,
        sparse_event_max_gap,
    ):
        return True
    return _has_degenerate_area_departure_event(
        frames,
        near_area_ratio,
        min_distance_gain_ratio,
        min_product_frames,
        sparse_event_max_gap,
    )


def _detect_hands(frame, hands, gpu_hands):
    """优先以 GPU 检测四个方向的人手，失败时完整回退到 CPU。"""
    from app.detection.held_product_detect import gpu_hand_angle, hand_angle

    if gpu_hands is not None:
        try:
            detected = [
                hand for rotation in range(4) for hand in gpu_hand_angle(frame, rotation, gpu_hands)
            ]
            return _dedupe_hands(detected)
        except Exception:
            pass
    detected = [hand for rotation in range(4) for hand in hand_angle(frame, rotation, hands)]
    return _dedupe_hands(detected)


def _detect_products(frame, detector):
    """运行已有 held product 模型并返回有效的候选框。"""
    from app.detection.held_product_detect import NoWarmupPredictor

    height, width = frame.shape[:2]
    prediction = detector.predict(
        frame,
        device=0,
        imgsz=640,
        rect=False,
        conf=0.005,
        iou=0.5,
        max_det=300,
        verbose=False,
        predictor=NoWarmupPredictor,
    )[0]
    if not len(prediction.boxes):
        return []
    return [
        [float(value) for value in box]
        for box in prediction.boxes.xyxy.cpu().numpy()
        if _area(box) <= 0.5 * width * height
    ]


def detect_product_taken(video_path, detector, hands, area_model, gpu_hands=None):
    """从视频中提取货架、手和持物证据，返回商品是否被拿走。"""
    import cv2

    from app.detection.area_detection import detect_frame_mask

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("无法打开视频")
    frames = []
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("无法读取视频尺寸或帧率")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError("暂不支持视频分辨率中途变化")
            mask, area = detect_frame_mask(frame, area_model)
            hand_rows = _detect_hands(frame, hands, gpu_hands)
            frames.append(
                {
                    "area": area,
                    "area_mask": mask,
                    "hands": hand_rows,
                    "products": _detect_products(frame, detector),
                }
            )
    finally:
        capture.release()
    if not frames:
        raise ValueError("视频没有可解码的帧")
    return is_product_taken(_fill_missing_areas(frames))
