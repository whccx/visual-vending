#!/usr/bin/env python3
"""testlab 可视化：完整跑单个视频的检测流水线，逐帧绘制柜区掩码、商品框、
手部框与拿取轨迹，输出标注帧目录供合成完整视频。

在 visual-vending-worker 容器内运行：
    python -m scripts.testlab.visualize_video tests/video/no/2.mp4

输出：data/testlab-visualize/<视频名>/annotated/%06d.jpg（标注帧）
"""

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/workspace")
CONTAINER_ID = "862337050531238"
FONT_PATH = ROOT / "data" / "testlab-visualize" / "font.ttc"

COLOR_PRODUCT = (0, 255, 255)      # 商品框：黄色
COLOR_HAND = (255, 255, 0)         # 手部框：青色
COLOR_TAKEN = (0, 0, 255)          # 拿取轨迹：红色
COLOR_TRACK = (0, 150, 255)        # 其他运动轨迹：橙色
COLOR_AREA = (0, 220, 0)           # 柜区掩码：绿色


def load_font(size=22):
    try:
        from PIL import ImageFont

        return ImageFont.truetype(str(FONT_PATH), size)
    except Exception:
        return None


def draw_label(image, xy, text, color, font):
    """优先用 PIL 绘制中文标签，字体不可用时退回 cv2 ASCII。"""
    if font is not None:
        from PIL import Image, ImageDraw

        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(rgb)
        draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
        image[:] = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_box(image, item, color, label, labels):
    box = item["box"] if isinstance(item, dict) else item
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    labels.append(((x1, max(12, y1 - 4)), label, color))


def annotate_frame(image, frame, detection_by_frame, mask_by_frame, tracks, font):
    labels = []
    row = detection_by_frame.get(frame)
    if row is not None:
        if row["area"] is not None:
            mask = mask_by_frame.get(frame)
            if mask is not None:
                overlay = image.copy()
                overlay[mask > 0] = COLOR_AREA
                cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
        for item in row["products"]:
            draw_box(image, item, COLOR_PRODUCT, "商品", labels)
        for item in row["hands"]:
            draw_box(image, item, COLOR_HAND, "手", labels)
    for track in tracks:
        color = COLOR_TAKEN if track["taken"] else COLOR_TRACK
        thickness = 2 if track["taken"] else 1
        history = [point for point in track["points"] if point["frame"] <= frame]
        if not history:
            continue
        for point in history[-12:]:
            x1, y1, x2, y2 = (int(v) for v in point["box"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if len(history) >= 2:
            points = [
                (int((p["box"][0] + p["box"][2]) / 2), int((p["box"][1] + p["box"][3]) / 2))
                for p in history
            ]
            cv2.polylines(image, [np.array(points)], False, color, thickness)
            label = f"轨迹{track['id']}" + ("（拿取）" if track["taken"] else "")
            labels.append((points[-1], label, color))
    for xy, text, color in labels:
        draw_label(image, xy, text, color, font)
    return image


def main() -> None:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/video/no/2.mp4"
    if not video.is_absolute():
        video = ROOT / video
    if not video.is_file():
        print(f"缺少视频：{video}")
        sys.exit(1)
    import scripts.testlab.run_videos as driver
    from vending.engine import unpack_features

    from app import config
    from app.detection.camera_block import judge_camera_blocked
    from app.detection.hand_area_gate import (
        cabinet_masks_for_candidates,
        filter_hand_linked_candidates,
    )
    from app.tracking.hoi_detr_motion import inspect_video_motion
    from app.tracking.trajectory_hoi_detr import build_visual_motion_tracks

    assets = driver.load_assets()
    catalog = {i: unpack_features(data) for i, data in enumerate(assets.values(), start=1)}
    sku_ids = dict(enumerate(assets, start=1))
    out_dir = ROOT / "data" / "testlab-visualize" / video.stem
    annotated = out_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    if annotated.exists():
        shutil.rmtree(annotated)
    annotated.mkdir()

    driver.SPOOL.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="testlab-viz-", dir=driver.SPOOL) as tmp:
        Path(tmp).chmod(0o750)
        target = Path(tmp) / video.name
        shutil.copy(video, target)
        target.chmod(0o640)
        from scripts.testlab.frame_pipeline import FrameRuntime

        runtime = FrameRuntime(1)
        try:
            result = runtime.recognize_view(target, CONTAINER_ID, catalog, sku_ids)
        finally:
            runtime.close()
        directory = target.parent / (target.stem + "-frames")
        print("识别结果：", json.dumps(result, ensure_ascii=False)[:200])

        files = sorted((directory / "hoi").glob("*.json"))
        detection = [json.loads(path.read_text()) for path in files]
        detection_by_frame = {row["frame"]: row for row in detection}
        areas = [row["area"] for row in detection]
        hands = [row["hands"] for row in detection]
        products = [row["products"] for row in detection]
        sequence = directory / "frames" / "%06d.jpg"
        mask_by_frame = {}
        packed = []
        for row in detection:
            mask = None
            if row["area"] is not None and row["hands"] and row["products"]:
                mask = cv2.imread(
                    str(directory / "area" / f"{row['frame']:06d}-mask.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                mask_by_frame[row["frame"]] = mask
            packed.append(np.packbits(mask.reshape(-1)) if mask is not None else None)
        masks = cabinet_masks_for_candidates(packed, (1080, 1920), hands, products)
        candidates, gate = filter_hand_linked_candidates(
            hands,
            products,
            areas,
            max_linked_hand_area_ratio=config.hoi_max_product_hand_area_ratio(),
            require_departure=True,
            hand_track_base_distance=config.hoi_hand_track_base_distance(),
            hand_track_iou_threshold=config.hoi_hand_track_iou_threshold(),
            area_masks_by_frame=masks,
        )
        print("轨迹筛选：", json.dumps(gate, ensure_ascii=False))
        moving, _ = inspect_video_motion(sequence, candidates)
        tracks_map = {}
        track_list = []
        for track in build_visual_motion_tracks(moving):
            points = track["points"]
            track_id = track["last_candidate"]["area_track_id"]
            motion = sum(point["motion_px"] for point in points)
            taken = len(points) >= config.SCORE_WINDOW_FRAMES
            if taken and (track_id not in tracks_map or motion > tracks_map[track_id][0]):
                tracks_map[track_id] = (motion, points)
            track_list.append({"id": track_id, "points": points, "taken": taken})
        # 与 match_video 一致：达到评分帧数阈值的轨迹即参与商品评分（拿取轨迹）
        track_list = [
            {"id": track_id, "points": points, "taken": True}
            for track_id, (_, points) in tracks_map.items()
        ] + [t for t in track_list if not t["taken"]]

        frame_count = len(detection)
        font = load_font()
        for index, row in enumerate(detection, start=1):
            image = cv2.imread(str(directory / "frames" / f"{row['frame']:06d}.jpg"))
            if image is None:
                continue
            annotate_frame(image, row["frame"], detection_by_frame, mask_by_frame, track_list, font)
            draw_label(
                image, (30, 30), f"帧 {row['frame']}/{frame_count}", (255, 255, 255),
                load_font(30),
            )
            cv2.imwrite(str(annotated / f"{row['frame']:06d}.jpg"), image)
    print(f"标注帧输出：{annotated}（{len(list(annotated.glob('*.jpg')))} 帧）")


if __name__ == "__main__":
    main()
