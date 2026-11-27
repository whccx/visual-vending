#!/usr/bin/env python3
"""testlab 抠图提取：提取视频 2 参与向量比对的评分窗口抠图。

在 visual-vending-worker 容器内运行：
    python -m scripts.testlab.extract_crops tests/video/no/2.mp4

输出到 data/testlab-visualize/<视频名>/crops/：
    参考-商品N号.jpg            库内 SKU 参考图（来自 data/covers）
    轨迹N-窗口帧-000123.png     评分窗口内的原始抠图（与向量比对输入同一裁剪）
    轨迹N-窗口帧-000123-masked.png  抠图叠加 BG0 掩码（手部未去除的版本）
    ranking.txt                 每条轨迹的评分窗口与前两名排名
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
REFERENCE_SKUS = [30, 25]  # 商品6号、商品32号（视频 2 前两名）


def build_tracks(directory):
    """与 match_video 一致地重建拿取轨迹。"""
    from app import config
    from app.detection.camera_block import judge_camera_blocked
    from app.detection.hand_area_gate import (
        cabinet_masks_for_candidates,
        filter_hand_linked_candidates,
    )
    from app.tracking.hoi_detr_motion import inspect_video_motion
    from app.tracking.trajectory_hoi_detr import build_visual_motion_tracks

    files = sorted((directory / "hoi").glob("*.json"))
    detection = [json.loads(path.read_text()) for path in files]
    areas = [row["area"] for row in detection]
    hands = [row["hands"] for row in detection]
    products = [row["products"] for row in detection]
    sequence = directory / "frames" / "%06d.jpg"
    if judge_camera_blocked(sequence, areas):
        raise SystemExit("检测到持续遮挡，无轨迹可提取")
    packed = []
    for row in detection:
        mask = None
        if row["area"] is not None and row["hands"] and row["products"]:
            mask = cv2.imread(
                str(directory / "area" / f"{row['frame']:06d}-mask.png"), cv2.IMREAD_GRAYSCALE
            )
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
    moving, _ = inspect_video_motion(sequence, candidates)
    tracks_map = {}
    for track in build_visual_motion_tracks(moving):
        points = track["points"]
        if len(points) < config.SCORE_WINDOW_FRAMES:
            continue
        track_id = track["last_candidate"]["area_track_id"]
        motion = sum(point["motion_px"] for point in points)
        if track_id not in tracks_map or motion > tracks_map[track_id][0]:
            tracks_map[track_id] = (motion, points)
    return tracks_map, gate


def main() -> None:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/video/no/2.mp4"
    if not video.is_absolute():
        video = ROOT / video
    if not video.is_file():
        print(f"缺少视频：{video}")
        sys.exit(1)
    import scripts.testlab.run_videos as driver
    from app.recognition.product_matching import (
        prepare_catalog,
        rank_scores,
        select_scoring_window,
    )
    from vending.engine import unpack_features

    assets = driver.load_assets()
    catalog = {i: unpack_features(data) for i, data in enumerate(assets.values(), start=1)}
    sku_ids = dict(enumerate(assets, start=1))
    out_dir = ROOT / "data" / "testlab-visualize" / video.stem
    crops = out_dir / "crops"
    if crops.exists():
        shutil.rmtree(crops)
    crops.mkdir(parents=True)
    # detector 只接受 data/spool 挂载内的图片路径
    work = Path(tempfile.mkdtemp(prefix="testlab-crops-", dir=driver.SPOOL))
    work.chmod(0o750)

    target = work / video.name
    shutil.copy(video, target)
    target.chmod(0o640)
    from scripts.testlab.frame_pipeline import FrameRuntime

    runtime = FrameRuntime(1)
    try:
        result = runtime.recognize_view(target, CONTAINER_ID, catalog, sku_ids)
    finally:
        runtime.close()
    directory = work / (target.stem + "-frames")
    print("识别结果：", json.dumps(result, ensure_ascii=False)[:160])

    matching = prepare_catalog(catalog)
    tracks_map, gate = build_tracks(directory)
    print("轨迹筛选：", json.dumps(gate, ensure_ascii=False))

    ranking_lines = []
    for track_id, (_, points) in sorted(tracks_map.items()):
        scores = []
        for point in points:
            x1, y1, x2, y2 = point["box"]
            if x1 < 19.2 or y1 < 10.8 or x2 > 1900.8 or y2 > 1069.2:
                continue
            box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
            key = "-".join(map(str, box))
            score_path = directory / "scores" / f"{point['frame']:06d}-{key}.json"
            if not score_path.is_file():
                continue
            record = json.loads(score_path.read_text())
            if record["scores"] is not None:
                scores.append((point["frame"], {int(k): v for k, v in record["scores"].items()}))
        frames, ranking = select_scoring_window(scores, matching[0], sku_scope=None)
        detail = [
            {k: item[k] for k in ("sku", "vector", "color", "combined")} for item in ranking[:2]
        ]
        ranking_lines.append(
            f"轨迹{track_id}：评分窗口帧={frames} 前两名={json.dumps(detail, ensure_ascii=False)}\n"
        )
        for frame in frames:
            candidates_box = [
                p for p in points
                if p["frame"] == frame
                and (directory / "scores" / f"{frame:06d}-{'-'.join(map(str, [math.floor(p['box'][0]), math.floor(p['box'][1]), math.ceil(p['box'][2]), math.ceil(p['box'][3])]))}.png").is_file()
            ]
            if not candidates_box:
                continue
            point = candidates_box[0]
            box = [math.floor(point["box"][0]), math.floor(point["box"][1]),
                   math.ceil(point["box"][2]), math.ceil(point["box"][3])]
            key = "-".join(map(str, box))
            shutil.copy(
                directory / "scores" / f"{frame:06d}-{key}.png",
                crops / f"轨迹{track_id}-窗口帧-{frame:06d}.png",
            )
            image = cv2.imread(str(directory / "frames" / f"{frame:06d}.jpg"))
            crop = image[box[1]:box[3], box[0]:box[2]]
            mask = cv2.imread(str(directory / "bg0" / f"{frame:06d}-{key}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == crop.shape[:2]:
                masked = crop.copy()
                masked[mask == 0] = 255
                cv2.imwrite(str(crops / f"轨迹{track_id}-窗口帧-{frame:06d}-masked.png"), masked)

    import sqlite3

    db = sqlite3.connect(driver.DB_PATH)
    for index in REFERENCE_SKUS:
        sku = sku_ids[index]
        av = db.execute("SELECT active_version FROM sku WHERE id=?", (sku,)).fetchone()[0]
        hashes, body = db.execute(
            "SELECT image_hashes, body FROM sku_version WHERE id=?", (av,)
        ).fetchone()
        name = json.loads(body).get("sku_name")
        digest = json.loads(hashes)[0]
        cover = driver.ROOT / "data" / "covers" / f"{digest}.jpg"
        if cover.is_file():
            shutil.copy(cover, crops / f"参考-{name}.jpg")
            ranking_lines.append(f"{name}（SKU {index}）参考图：{cover}\n")
        else:
            print(f"缺少参考图：{cover}")
    db.close()
    (crops / "ranking.txt").write_text("".join(ranking_lines), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print("".join(ranking_lines))
    print(f"抠图输出：{crops}")
    for path in sorted(crops.glob("*.png")) + sorted(crops.glob("*.jpg")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
