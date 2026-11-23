"""为 HOI-DETR 在视频帧之间关联与手部相连的商品框。"""

import math


def center(box):
    """返回左上、右下坐标表示的矩形框中心点。"""
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def iou(a, b):
    """计算两个矩形框的交并比；空框使用极小分母避免除零。"""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-6, area_a + area_b - intersection)


def build_visual_motion_tracks(frames, max_appearance_distance=0.5, ambiguity_margin=0.025):
    """根据运动和外观将逐帧候选关联成轨迹。

    参数：
        frames: 每帧候选列表，候选须包含商品框、运动量和外观特征。
        max_appearance_distance: 两个候选之间允许的最大外观距离。
        ambiguity_margin: 同一候选的不同轨迹匹配代价最小差距。
    返回：
        包含轨迹编号、轨迹点和最近候选的字典列表。

    不设置像素距离或帧间隔上限；框不重叠时还要求局部特征匹配。
    """
    from app.tracking.hoi_detr_motion import appearance_distance, feature_correspondence

    tracks = []
    for frame_no, candidates in enumerate(frames, 1):
        valid = [
            c
            for c in candidates
            if c.get("_motion_px", 0) >= 1.0
            and c.get("_appearance") is not None
            and len(c.get("box", [])) == 4
            and all(math.isfinite(x) for x in c["box"])
            and c["box"][2] > c["box"][0]
            and c["box"][3] > c["box"][1]
        ]
        proposals = []
        for ci, candidate in enumerate(valid):
            for ti, track in enumerate(tracks):
                last = track["last_candidate"]
                if last.get("area_track_id") != candidate.get("area_track_id"):
                    continue
                distance = appearance_distance(
                    last.get("_appearance"), candidate.get("_appearance")
                )
                if distance is None or distance > max_appearance_distance:
                    continue
                overlap = iou(last["box"], candidate["box"])
                features = None
                if overlap < 0.05:
                    features = feature_correspondence(
                        last.get("_features"), candidate.get("_features")
                    )
                    if features is None:
                        continue
                # 使用外观和经过验证的局部运动关联，不限制像素距离。
                cost = distance - 0.03 * min(overlap, 1.0)
                proposals.append((cost, ti, ci))
        proposals.sort()
        used_tracks, used_candidates = set(), set()
        for cost, ti, ci in proposals:
            if ti in used_tracks or ci in used_candidates:
                continue
            alternatives = [
                other
                for other, other_ti, other_ci in proposals
                if other_ci == ci and other_ti != ti and other_ti not in used_tracks
            ]
            if alternatives and min(alternatives) - cost < ambiguity_margin:
                # 同一候选存在难以区分的轨迹时，不与已有轨迹关联。
                continue
            candidate = valid[ci]
            point = {
                "frame": frame_no,
                "box": candidate["box"],
                "point": center(candidate["box"]),
                "motion_px": candidate["_motion_px"],
            }
            tracks[ti]["points"].append(point)
            tracks[ti]["last_candidate"] = candidate
            used_tracks.add(ti)
            used_candidates.add(ci)
        for ci, candidate in enumerate(valid):
            if ci in used_candidates:
                continue
            point = {
                "frame": frame_no,
                "box": candidate["box"],
                "point": center(candidate["box"]),
                "motion_px": candidate["_motion_px"],
            }
            tracks.append(
                {"track_id": len(tracks) + 1, "points": [point], "last_candidate": candidate}
            )
    return tracks


def select_visually_moving_track(video_path, frames, min_points=5):
    """检测候选运动并选择累计运动量最大的有效轨迹。

    参数：
        video_path: 视频路径，传给 HOI-DETR 运动检测函数。
        frames: 按帧排列的检测候选。
        min_points: 轨迹参与选择所需的最少点数。
    返回：
        所选轨迹及统计字典；无有效轨迹时返回 None 和统计字典。
    """
    from app.tracking.hoi_detr_motion import inspect_video_motion

    moving, audit = inspect_video_motion(video_path, frames)
    tracks = build_visual_motion_tracks(moving)
    candidates = [t for t in tracks if len(t["points"]) >= min_points]
    for track in candidates:
        track["motion_sum_px"] = sum(p["motion_px"] for p in track["points"])
    chosen = (
        max(candidates, key=lambda t: (t["motion_sum_px"], len(t["points"])))
        if candidates
        else None
    )
    summary = {
        "raw_candidates": len(audit),
        "moving_candidates": sum(x["status"] == "moving" for x in audit),
        "stationary_candidates": sum(x["status"] == "stationary" for x in audit),
        "unknown_candidates": sum(x["status"] == "unknown" for x in audit),
        "tracks": len(tracks),
        "selected_track_points": len(chosen["points"]) if chosen else 0,
    }
    if chosen:
        chosen = {k: v for k, v in chosen.items() if k != "last_candidate"}
    return chosen, summary
