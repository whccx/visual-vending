"""选择经运动验证的商品轨迹，并排除靠近视频边缘的裁剪区域。"""

from app.tracking.product_motion_tracking import select_visually_moving_track
from app.tracking.product_motion import inspect_video_motion


def select_motion_rank_points(
    video_path,
    boxes_by_frame,
    width,
    height,
    margin_ratio=0.02,
    embeddings=None,
    match_fallback=None,
    product_track_ids=None,
):
    """选择可用于商品匹配的运动轨迹点。

    参数：
        video_path: 用于检查运动的视频路径。
        boxes_by_frame: 每帧的商品框，无检测时对应元素为 None。
        width, height: 视频宽高，必须为正数。
        margin_ratio: 排除画面边缘裁剪框时使用的宽高比例。
        embeddings: 按从 1 开始的帧号索引的可选特征。
        match_fallback: 轨迹匹配的可选回退函数。
        product_track_ids: 与逐帧商品框对应的可选区域轨迹编号。
    返回：
        筛选后的轨迹及统计信息；没有合适轨迹时轨迹为 None。
    异常：
        视频尺寸无效或轨迹编号数量与帧数不一致时抛出 ValueError。
    """
    if width <= 0 or height <= 0:
        raise ValueError("无法确认视频尺寸")
    if not any(box is not None for box in boxes_by_frame):
        return None, {"raw_candidates": 0, "moving_candidates": 0, "rankable_points": 0}
    if product_track_ids is not None and len(product_track_ids) != len(boxes_by_frame):
        raise ValueError("商品轨迹与视频帧数不一致")
    frames = [
        [
            {
                "box": box,
                "_frame": i + 1,
                "_embedding": (embeddings or {}).get(i + 1),
                "area_track_id": product_track_ids[i] if product_track_ids is not None else None,
            }
        ]
        if box is not None
        else []
        for i, box in enumerate(boxes_by_frame)
    ]
    track, summary = select_visually_moving_track(
        video_path, frames, match_fallback=match_fallback, motion_inspector=inspect_video_motion
    )
    if not track:
        return None, summary
    mx, my = margin_ratio * width, margin_ratio * height
    points = [
        p
        for p in track["points"]
        if p["box"][0] >= mx
        and p["box"][1] >= my
        and p["box"][2] <= width - mx
        and p["box"][3] <= height - my
    ]
    summary["rankable_points"] = len(points)
    return dict(track, points=points), summary
