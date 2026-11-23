"""基于运动和外观信息选择商品轨迹，供多条识别路径复用。"""

import numpy as np
from app.tracking.product_motion import inspect_video_motion


def select_stable_track(
    video_path,
    boxes_by_frame,
    width,
    height,
    embeddings,
    close=0.15,
    hold=0.45,
    confirm=3,
    min_points=5,
    margin_ratio=0.02,
    min_displacement=5.0,
    product_track_ids=None,
):
    """选择相邻帧 DINOv3 特征保持接近的商品轨迹。

    特征差异较大的候选先暂存，连续 `confirm` 帧接近后才并入轨迹，
    避免将短暂接触的其他商品接入正在携带的商品轨迹。"""
    if product_track_ids is not None and len(product_track_ids) != len(boxes_by_frame):
        raise ValueError("商品轨迹与视频帧数不一致")
    frames = [
        [
            {
                "box": box,
                "_frame": index + 1,
                "_embedding": (embeddings or {}).get(index + 1),
                "_area_track_id": product_track_ids[index]
                if product_track_ids is not None
                else None,
            }
        ]
        if box is not None
        else []
        for index, box in enumerate(boxes_by_frame)
    ]
    if not any(frames):
        return None, {"tracks": 0, "confirmed_points": 0, "rankable_points": 0}
    moving, audit = inspect_video_motion(video_path, frames)

    def cosine_distance(a, b):
        """计算两个特征向量的余弦距离，缺失或零向量返回空值。"""
        if a is None or b is None:
            return None
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0 or nb <= 0:
            return None
        return 1.0 - float(np.dot(a, b)) / (na * nb)

    def center(box):
        """计算检测框中心坐标。"""
        return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]

    def make_point(frame_no, candidate):
        """将候选框转换为轨迹点，并保留运动幅度。"""
        return {
            "frame": frame_no,
            "box": candidate["box"],
            "point": center(candidate["box"]),
            "motion_px": float(candidate.get("_motion_px") or 0.0),
        }

    tracks = []
    for frame_no, candidates in enumerate(moving, 1):
        assigned = set()
        used = set()
        proposals = []
        for candidate_index, candidate in enumerate(candidates):
            embedding = candidate.get("_embedding")
            if embedding is None:
                continue
            for track_index, track in enumerate(tracks):
                if track["area_track_id"] != candidate.get("_area_track_id"):
                    continue
                distance = cosine_distance(track["last"], embedding)
                if distance is not None and distance <= hold:
                    proposals.append((distance, track_index, candidate_index))
        proposals.sort()
        for distance, track_index, candidate_index in proposals:
            if track_index in used or candidate_index in assigned:
                continue
            candidate = candidates[candidate_index]
            track = tracks[track_index]
            used.add(track_index)
            assigned.add(candidate_index)
            if distance <= close:
                track["streak"] += 1
                track["buffer"].append(make_point(frame_no, candidate))
                if track["streak"] >= confirm:
                    track["points"].extend(track["buffer"])
                    track["buffer"] = []
                    track["streak"] = 0
            else:
                track["buffer"].append(make_point(frame_no, candidate))
                track["streak"] = 0
                if len(track["buffer"]) >= confirm:
                    track["buffer"] = []
            track["last"] = embedding
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in assigned:
                continue
            tracks.append(
                {
                    "track_id": len(tracks) + 1,
                    "area_track_id": candidate.get("_area_track_id"),
                    "points": [make_point(frame_no, candidate)],
                    "buffer": [],
                    "streak": 0,
                    "last": candidate.get("_embedding"),
                }
            )
    valid = []
    displacement_rejected = 0
    for track in tracks:
        if len(track["points"]) < min_points:
            continue
        first, last = track["points"][0]["point"], track["points"][-1]["point"]
        track["displacement_px"] = float(np.hypot(last[0] - first[0], last[1] - first[1]))
        if track["displacement_px"] < min_displacement:
            displacement_rejected += 1
            continue
        valid.append(track)
    for track in valid:
        track["motion_sum_px"] = sum(p["motion_px"] for p in track["points"])
    chosen = max(valid, key=lambda t: (t["motion_sum_px"], len(t["points"]))) if valid else None
    summary = {
        "raw_candidates": len(audit),
        "moving_candidates": sum(x["status"] == "moving" for x in audit),
        "stationary_candidates": sum(x["status"] == "stationary" for x in audit),
        "unknown_candidates": sum(x["status"] == "unknown" for x in audit),
        "tracks": len(tracks),
        "valid_tracks": len(valid),
        "displacement_rejected": displacement_rejected,
        "selected_track_points": len(chosen["points"]) if chosen else 0,
    }
    if not chosen:
        return None, summary
    margin_x, margin_y = margin_ratio * width, margin_ratio * height
    points = [
        p
        for p in chosen["points"]
        if p["box"][0] >= margin_x
        and p["box"][1] >= margin_y
        and p["box"][2] <= width - margin_x
        and p["box"][3] <= height - margin_y
    ]
    summary["rankable_points"] = len(points)
    return dict(chosen, points=points), summary
