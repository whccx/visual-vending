"""YOLOE 和 100DOH 共用的、基于独立运动验证的商品轨迹关联。"""

import math


def center(box):
    """返回矩形框 [左、上、右、下] 的中心坐标。"""
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def iou(a, b):
    """计算两个矩形框的交并比，使用极小下限避免分母为零。"""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-6, aa + bb - inter)


def build_visual_motion_tracks(
    frames, max_appearance_distance=0.5, ambiguity_margin=0.025, match_fallback=None
):
    """将经过运动验证的候选框关联为轨迹，不设置距离或帧间隔上限。

    参数：
        frames: 按帧排列的候选列表，候选需包含运动量、外观和商品框。
        max_appearance_distance: 允许直接关联的最大外观距离。
        ambiguity_margin: 最优匹配与其他轨迹匹配之间的最小差距。
        match_fallback: 直接匹配失败时使用的可选匹配函数。
    返回：
        含轨迹编号、轨迹点和最近候选的字典列表。

    两个框不重叠时，直接匹配还要求局部特征对应成功。
    """
    from app.tracking.product_motion import appearance_distance, feature_correspondence

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
                overlap = iou(last["box"], candidate["box"])
                features = None
                if distance is not None and distance <= max_appearance_distance and overlap < 0.05:
                    features = feature_correspondence(
                        last.get("_features"), candidate.get("_features")
                    )
                direct_match = (
                    distance is not None
                    and distance <= max_appearance_distance
                    and (overlap >= 0.05 or features is not None)
                )
                fallback_match = (
                    not direct_match
                    and match_fallback is not None
                    and match_fallback(last, candidate)
                )
                if not direct_match and not fallback_match:
                    continue
                # 匹配依据是外观和已验证的局部运动，而非像素距离上限。
                cost = (
                    distance - 0.03 * min(overlap, 1.0) if direct_match else max_appearance_distance
                )
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
                # 身份存在歧义时，不把候选关联到任一已有轨迹。
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


def select_visually_moving_track(
    video_path, frames, min_points=5, match_fallback=None, motion_inspector=None
):
    """检测候选运动并选择累计运动量最大的有效轨迹。

    参数：
        video_path: 运动检测使用的视频路径。
        frames: 按帧排列的商品候选。
        min_points: 轨迹参与选择所需的最少点数。
        match_fallback: 外观直接匹配失败时的可选回退函数。
        motion_inspector: 可替代默认视频运动检测的函数。
    返回：
        所选轨迹和候选、轨迹统计信息；无有效轨迹时轨迹为 None。
    """
    if motion_inspector is None:
        from app.tracking.product_motion import inspect_video_motion

        motion_inspector = inspect_video_motion
    moving, audit = motion_inspector(video_path, frames)
    tracks = build_visual_motion_tracks(moving, match_fallback=match_fallback)
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
