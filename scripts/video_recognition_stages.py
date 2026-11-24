"""读取并发检测的帧图和 JSON，逐段执行轨迹、裁剪、BG0 和 SKU 匹配。

track、crop 和 match 在 worker 镜像运行，bg0 在 detector 镜像运行。
track 完成后可分别运行 crop 与 bg0，两项均完成后运行 match。
实验目录与检测脚本共用。match 接收 SKU ID 清单，只读打开商品数据库。
"""

import argparse
import math
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from app import config
from scripts.parallel_video_detection import (
    JOB_COUNTS,
    Overlap,
    read_frame,
    read_json,
    write_json,
    write_png,
)


def track_video(video: Path, entry: dict, jobs: int, detection_jobs: int, area_jobs: int) -> dict:
    """按帧号还原检测结果，筛选商品离架轨迹。"""
    from app.api.recognize_hoi_detr import _classify_without_match
    from app.detection.camera_block import judge_camera_blocked
    from app.detection.hand_area_gate import (
        cabinet_masks_for_candidates,
        filter_hand_linked_candidates,
    )
    from app.tracking.hoi_detr_motion import inspect_video_motion
    from app.tracking.trajectory_hoi_detr import build_visual_motion_tracks
    from vending.engine import Pipeline, unmatched_video_result
    from vending.store import outcome

    size = entry["frame_count"]
    hoi = video / f"hoi-{detection_jobs}"
    area = video / f"area-{area_jobs}"
    sequence = video / "frames" / f"%06d{entry.get('suffix', '.png')}"
    rows = [read_json(hoi / f"{frame:06d}.json") for frame in range(1, size + 1)]
    if [row["frame"] for row in rows] != list(range(1, size + 1)):
        raise ValueError(f"检测 JSON 帧号与拆帧清单不一致 {video}")
    areas = [row["area"] for row in rows]
    hands = [row["hands"] for row in rows]
    products = [row["products"] for row in rows]
    report = {"tracks": []}
    if judge_camera_blocked(sequence, areas):
        report["result"] = outcome(6, "检测到与柜门状态不一致的持续遮挡，需复核")
    else:
        packed = []
        for row in rows:
            mask = None
            if row["area"] is not None and row["hands"] and row["products"]:
                mask_path = area / f"{row['frame']:06d}-mask.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (1080, 1920):
                    raise ValueError(f"柜区掩码缺失或分辨率错误 {mask_path}")
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
        moving, audit = inspect_video_motion(sequence, candidates)
        selected = {}
        for track in build_visual_motion_tracks(moving):
            points = track["points"]
            if len(points) < config.SCORE_WINDOW_FRAMES:
                continue
            track_id = track["last_candidate"]["area_track_id"]
            score = sum(point["motion_px"] for point in points)
            if track_id not in selected or score > selected[track_id][0]:
                selected[track_id] = (score, points)
        report.update(gate=gate, motion=audit)
        if not selected:
            state = Pipeline()
            detection = {"areas": areas, "hands": hands, "frames": products}
            status, area_frames, departure = _classify_without_match(
                detection, sequence, state, masks
            )
            report["result"] = unmatched_video_result(
                status,
                departure,
                gate["taken_tracks"],
                door_opened=area_frames >= config.hoi_door_min_area_frames(),
            )
        elif len(selected) != gate["taken_tracks"]:
            report["result"] = outcome(4, "存在未形成有效识别轨迹的拿取，需复核")
        else:
            for track_id, (_, points) in selected.items():
                frames = []
                for point in points:
                    x1, y1, x2, y2 = point["box"]
                    if x1 < 19.2 or y1 < 10.8 or x2 > 1900.8 or y2 > 1069.2:
                        continue
                    frames.append(
                        {
                            "frame": point["frame"],
                            "box": [
                                math.floor(x1),
                                math.floor(y1),
                                math.ceil(x2),
                                math.ceil(y2),
                            ],
                        }
                    )
                report["tracks"].append({"track_id": track_id, "frames": frames})
    output = video / f"track-{jobs}.json"
    write_json(output, report)
    return {"tracks": len(report["tracks"]), "early_result": report.get("result")}


def track_frames(video: Path, track_jobs: int):
    for track in read_json(video / f"track-{track_jobs}.json")["tracks"]:
        for point in track["frames"]:
            yield str(track["track_id"]), point["frame"], point["box"]


def crop_video(video: Path, entry: dict, jobs: int, track_jobs: int) -> dict:
    """逐帧裁剪商品框，JPEG 用 GPU 解码并在显存内裁剪。"""
    import torch
    from torchvision.io import ImageReadMode, decode_jpeg

    output = video / f"crop-{jobs}"
    output.mkdir()
    suffix = entry.get("suffix", ".png")
    count = 0
    for track_id, frame, (x1, y1, x2, y2) in track_frames(video, track_jobs):
        directory = output / track_id
        directory.mkdir(exist_ok=True)
        if suffix == ".jpg":
            jpeg = (video / "frames" / f"{frame:06d}.jpg").read_bytes()
            encoded = torch.from_numpy(np.frombuffer(jpeg, dtype=np.uint8).copy())
            rgb = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")
            crop = rgb[:, y1:y2, x1:x2].permute(1, 2, 0).contiguous().cpu().numpy()
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        else:
            image = read_frame(video / "frames", frame, suffix)
            crop = image[y1:y2, x1:x2]
        write_png(directory / f"{frame:06d}.png", crop)
        count += 1
    return {"crops": count, "image_backend": "nvJPEG CUDA" if suffix == ".jpg" else "OpenCV PNG"}


def segment_video(
    video: Path, entry: dict, jobs: int, track_jobs: int, bg0, overlap: Overlap
) -> dict:
    """在整帧图上运行 TensorRT BG0，保存框内掩码和透明底预览。"""
    output = video / f"bg0-{jobs}"
    output.mkdir()
    suffix = entry.get("suffix", ".png")
    count = 0
    for track_id, frame, box in track_frames(video, track_jobs):
        directory = output / track_id
        directory.mkdir(exist_ok=True)
        image = read_frame(video / "frames", frame, suffix)
        with overlap.call():
            mask = bg0.segment(image, box, config.BG0_MASK_THRESHOLD)
        x1, y1, x2, y2 = box
        rgba = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask.astype(np.uint8) * 255
        write_png(directory / f"{frame:06d}-mask.png", rgba[:, :, 3])
        write_png(directory / f"{frame:06d}.png", rgba)
        count += 1
    return {"masked_crops": count}


def load_sku_features(database: Path, scope: Path) -> dict:
    """只读加载显式列出的已通过审核的 SKU 特征。"""
    from vending.engine import unpack_features

    sku_ids = read_json(scope)
    if (
        not isinstance(sku_ids, list)
        or not sku_ids
        or any(not isinstance(value, str) for value in sku_ids)
    ):
        raise ValueError("SKU 清单需要是非空的识别 SKU ID 数组")
    catalog = {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for sku_id in dict.fromkeys(sku_ids):
            row = db.execute(
                """SELECT active.features FROM sku JOIN sku_version AS active ON active.id=sku.active_version
                   WHERE sku.id=? AND sku.deleted=0 AND active.state=2 AND active.features IS NOT NULL""",
                (sku_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"SKU 缺少有效参考特征 {sku_id}")
            catalog[sku_id] = unpack_features(row[0])
    return catalog


def score_track(
    video: Path,
    entry: dict,
    track: dict,
    output: Path,
    crop_jobs: int,
    bg0_jobs: int,
    scoring: tuple,
    state,
    overlap: Overlap,
) -> dict:
    """按单帧计算特征分数，再复用现有的连续窗口和拒识规则。"""
    import torch

    from app.recognition.hoi_mask_matching import (
        _enhance_crop,
        _reject_small_mask,
        _remove_hand_from_mask,
    )
    from app.recognition.product_matching import (
        ranking_rejection_reason,
        score_crop,
        select_scoring_window,
    )

    model, catalog, mean, std = scoring
    track_id = str(track["track_id"])
    folder = output / track_id
    folder.mkdir()
    frame_scores = []
    for point in track["frames"]:
        frame, box = point["frame"], point["box"]
        name = f"{frame:06d}"
        raw_mask = cv2.imread(
            str(video / f"bg0-{bg0_jobs}" / track_id / f"{name}-mask.png"), cv2.IMREAD_GRAYSCALE
        )
        crop = cv2.imread(str(video / f"crop-{crop_jobs}" / track_id / f"{name}.png"))
        if raw_mask is None or crop is None or crop.shape[:2] != raw_mask.shape:
            raise ValueError(f"裁剪图和 BG0 掩码缺失或尺寸不一致 {video}/{track_id}/{name}")
        mask = raw_mask > 0
        item = {"frame": frame, "mask": mask, "bg0_fraction": float(mask.mean())}
        if config.HOI_REMOVE_HAND and item["bg0_fraction"] >= config.HOI_MASK_MATCH_MIN_FRACTION:
            image = read_frame(video / "frames", frame, entry.get("suffix", ".png"))
            item["mask"], hand, _ = _remove_hand_from_mask(image, box, mask, state)
            write_png(folder / f"{name}-hand.png", hand.astype(np.uint8) * 255)
        _reject_small_mask(item)
        detail = {key: value for key, value in item.items() if key != "mask"}
        if item["mask"] is not None:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            enhanced = _enhance_crop(rgb)
            with torch.inference_mode(), overlap.call():
                _, scores = score_crop(
                    torch.from_numpy(enhanced).cuda().permute(2, 0, 1),
                    torch.from_numpy(np.ascontiguousarray(item["mask"])).cuda(),
                    model,
                    catalog,
                    mean,
                    std,
                    color_crop=torch.from_numpy(rgb).cuda().permute(2, 0, 1),
                )
            frame_scores.append((frame, scores))
            detail["scores"] = scores
        write_json(folder / f"{name}.json", detail)
    frames, ranking = select_scoring_window(frame_scores, catalog[0])
    return {
        "track_id": track["track_id"],
        "frames": frames,
        "ranking": ranking,
        "rejection": ranking_rejection_reason(ranking, len(frames)),
    }


def match_video(video: Path, entry: dict, args, scoring: tuple, state, overlap: Overlap) -> dict:
    """汇总轨迹匹配结果，按轨迹计数并写结果文件。"""
    from vending.store import outcome

    track_report = read_json(video / f"track-{args.track_jobs}.json")
    output = video / f"match-{args.jobs}"
    output.mkdir()
    tracks = [
        score_track(
            video, entry, track, output, args.crop_jobs, args.bg0_jobs, scoring, state, overlap
        )
        for track in track_report["tracks"]
    ]
    result = track_report.get("result")
    counts = Counter()
    unmatched = False
    for track in tracks:
        reason = track["rejection"]
        if reason == "ambiguous_margin":
            result = outcome(4, f"前两名商品综合分差小于 {config.SCORE_MIN_MARGIN:g}，无法确认商品")
            break
        if reason == "insufficient_evidence":
            result = outcome(4, "商品轨迹的有效评分帧或匹配证据不足")
            break
        if reason == "unmatched_scores":
            unmatched = True
        else:
            counts[track["ranking"][0]["sku"]] += 1
    if result is None:
        result = (
            outcome(7, "已确认拿取，存在未能匹配本单范围内 SKU 的商品")
            if unmatched
            else outcome(
                2,
                items=[
                    {"ks_sku_id": sku, "sku_num": count} for sku, count in sorted(counts.items())
                ],
            )
        )
    write_json(output / "result.json", {"result": result, "tracks": tracks})
    return {"result": result, "tracks": len(tracks)}


def run_stage(args, entries: list) -> dict:
    """在隔离运行时按视频并发，各路独立持有有状态模型。"""
    from types import SimpleNamespace

    overlap = Overlap()
    scoring = ()
    if args.stage == "match":
        import torch

        from app.recognition.product_matching import prepare_catalog
        from vending.engine import Pipeline

        catalog = prepare_catalog(load_sku_features(args.database, args.sku_scope))
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
        scoring = (Pipeline().dino, catalog, mean, std)

    def run_lane(videos: list) -> list:
        bg0 = None
        if args.stage == "bg0":
            import tensorrt as trt
            from bg0 import Bg0Segmenter

            bg0 = Bg0Segmenter(Path("/models"), trt.Logger(trt.Logger.WARNING))
        state = SimpleNamespace()
        results = []
        try:
            for entry in videos:
                path = args.output / entry["directory"]
                started = time.monotonic()
                if args.stage == "track":
                    result = track_video(path, entry, args.jobs, args.hoi_jobs, args.area_jobs)
                elif args.stage == "crop":
                    result = crop_video(path, entry, args.jobs, args.track_jobs)
                elif args.stage == "bg0":
                    result = segment_video(path, entry, args.jobs, args.track_jobs, bg0, overlap)
                    result["cuda_stream"] = int(bg0.engine.stream)
                else:
                    result = match_video(path, entry, args, scoring, state, overlap)
                    result["cuda_stream"] = torch.cuda.current_stream().cuda_stream
                finished = time.monotonic()
                result.update(
                    source=entry["source"],
                    wall_seconds=finished - started,
                    started_monotonic=started,
                    finished_monotonic=finished,
                )
                results.append(result)
                print(
                    f"{args.stage} 完成 {entry['source']}，耗时 {result['wall_seconds']:.2f} 秒",
                    flush=True,
                )
        finally:
            if bg0 is not None:
                bg0.close()
        return results

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        reports = pool.map(run_lane, (entries[index :: args.jobs] for index in range(args.jobs)))
        return {
            "videos": [video for report in reports for video in report],
            **overlap.report(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("track", "crop", "bg0", "match"))
    parser.add_argument("output", type=Path, help="已有 split.json 的实验目录")
    for option in ("jobs", "area-jobs", "hoi-jobs", "track-jobs", "crop-jobs", "bg0-jobs"):
        parser.add_argument(f"--{option}", type=int, choices=JOB_COUNTS, default=3)
    parser.add_argument("--sku-scope", type=Path, help="match 阶段使用的识别 SKU ID JSON 数组")
    parser.add_argument("--database", type=Path, default=Path("/workspace/data/platform.sqlite3"))
    args = parser.parse_args()
    if args.stage == "match" and not args.sku_scope:
        parser.error("match 阶段需要 --sku-scope")
    if config.HOI_ORIENTATION_ENABLED:
        parser.error("此实验要求使用未旋转的原帧坐标")
    # 旧的轨迹函数通过 VideoCapture 读取。用图片序列后端可逐像素复用检测读过的帧图。
    os.environ["OPENCV_VIDEOIO_PRIORITY_CV_IMAGES"] = "1100"
    cv2.setNumThreads(1)
    started = time.monotonic()
    entries = read_json(args.output / "split.json")["videos"]
    if args.jobs > len(entries):
        parser.error("并发数不能大于视频数量")
    result = run_stage(args, entries)
    result.update(jobs=args.jobs, wall_seconds=time.monotonic() - started)
    report = args.output / f"{args.stage}-{args.jobs}.json"
    write_json(report, result)
    print(f"{args.stage} 完成，整批耗时 {result['wall_seconds']:.2f} 秒，报告 {report}")


if __name__ == "__main__":
    main()
