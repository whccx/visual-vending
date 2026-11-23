"""将 YOLOE 识别过程中实际产生的框和最终轨迹转为可序列化数据。"""


def build_visualization(boxes, hands_by_frame, primary):
    """返回逐帧候选框、手框和最终选中轨迹；帧号从 1 开始。"""
    if len(boxes) != len(hands_by_frame):
        raise ValueError("商品框和手框的帧数不一致")
    return {
        "boxes": [[float(v) for v in box] if box is not None else None for box in boxes],
        "hands": [
            [[float(v) for v in hand["box"]] for hand in frame]
            for frame in hands_by_frame
        ],
        "track": [
            {
                "frame": int(point["frame"]),
                "box": [float(v) for v in point["box"]],
                "point": [float(v) for v in point["point"]],
            }
            for point in (primary or {}).get("points", [])
        ],
    }
