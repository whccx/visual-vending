#!/usr/bin/env python3
"""YOLOE 手持商品检测，可供常驻服务和命令行复用。"""

import argparse
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops
from app.recognition.full_color import encode_mask

from app import config

R = config.ROOT


class NoWarmupPredictor(SegmentationPredictor):
    """跳过预测器预热，复用已有检测模型。"""

    def __init__(self, *args, **kwargs):
        """初始化分割预测器，并跳过默认的预热推理。"""
        super().__init__(*args, **kwargs)
        self.done_warmup = True


def load_detector():
    """加载 YOLOE 商品模型与 CPU 手部检测器，不执行预热推理。

    返回：
        商品检测模型和 MediaPipe Hands 对象的二元组。
    """
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    model = YOLOE(str(config.HELD_PRODUCT_MODEL))
    cwd = os.getcwd()
    os.chdir(config.HELD_PRODUCT_DIR)
    try:
        model.set_classes(["held product"])
    finally:
        os.chdir(cwd)
    # 初始化真实推理后端，但不执行空输入预测。
    args = {
        **model.overrides,
        "conf": 0.005,
        "device": 0,
        "imgsz": 640,
        "rect": False,
        "iou": 0.5,
        "max_det": 300,
        "verbose": False,
        "mode": "predict",
        "save": False,
    }
    model.predictor = NoWarmupPredictor(overrides=args, _callbacks=model.callbacks)
    model.predictor.setup_model(model=model.model, verbose=False)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=6,
        model_complexity=1,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    return model, hands


def area(b):
    """计算矩形框面积，无效宽高按零处理。"""
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    """计算两个矩形框的交集面积。"""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def near(b, a):
    """判断框 b 是否与按四分之一宽高扩展后的框 a 相交。"""
    x, y, X, Y = a
    w = X - x
    h = Y - y
    return (
        b[2] >= x - 0.25 * w
        and b[0] <= X + 0.25 * w
        and b[3] >= y - 0.25 * h
        and b[1] <= Y + 0.25 * h
    )


def possible_merge(a, ds):
    """返回与候选 a 有适度交叠且分数足够高的其他候选。"""
    ans = []
    for d in ds:
        if d is a or d["score"] < a["score"] * 0.3:
            continue
        x = inter(a["box"], d["box"])
        sm = min(area(a["box"]), area(d["box"]))
        u = area(a["box"]) + area(d["box"]) - x
        if sm and u and x / u >= 0.03 and x / sm < 0.4:
            ans.append(d)
    return ans


def hand_angle(im, k, hands):
    """使用 CPU 手部模型检测旋转后的图像，并将手框映射回原图。

    参数：
        im: BGR 图像；k: 顺时针旋转的 90 度次数；hands: MediaPipe 检测器。
    返回：
        包含旋转角度和原图手部矩形框的列表。
    """
    h, w = im.shape[:2]
    rot = (
        im
        if k == 0
        else cv2.rotate(
            im,
            {
                1: cv2.ROTATE_90_CLOCKWISE,
                2: cv2.ROTATE_180,
                3: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[k],
        )
    )
    rh, rw = rot.shape[:2]
    r = hands.process(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
    found = []
    for ls in r.multi_hand_landmarks or []:
        pts = []
        for v in ls.landmark:
            x, y = v.x * rw, v.y * rh
            if k == 1:
                x, y = y, h - x
            elif k == 2:
                x, y = w - x, h - y
            elif k == 3:
                x, y = w - y, x
            pts.append((x, y))
        found.append(
            {
                "rotation": k * 90,
                "box": [
                    min(x for x, y in pts),
                    min(y for x, y in pts),
                    max(x for x, y in pts),
                    max(y for x, y in pts),
                ],
            }
        )
    return found


def load_gpu_hands():
    """创建使用 GPU 委托和图像模式的 MediaPipe 手部关键点检测器。"""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(config.HAND_MODEL),
            delegate=python.BaseOptions.Delegate.GPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=6,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    return vision.HandLandmarker.create_from_options(options)


def gpu_hand_angle(im, k, detector):
    """用 GPU 手部模型检测旋转图像，返回映射回原图的手框列表。"""
    h, w = im.shape[:2]
    rot = (
        im
        if k == 0
        else cv2.rotate(
            im,
            {
                1: cv2.ROTATE_90_CLOCKWISE,
                2: cv2.ROTATE_180,
                3: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[k],
        )
    )
    rh, rw = rot.shape[:2]
    r = detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))
    )
    found = []
    for hand in r.hand_landmarks:
        pts = []
        for v in hand:
            x, y = v.x * rw, v.y * rh
            if k == 1:
                x, y = y, h - x
            elif k == 2:
                x, y = w - x, h - y
            elif k == 3:
                x, y = w - y, x
            pts.append((x, y))
        found.append(
            {
                "rotation": k * 90,
                "box": [
                    min(x for x, y in pts),
                    min(y for x, y in pts),
                    max(x for x, y in pts),
                    max(y for x, y in pts),
                ],
            }
        )
    return found


def decide(ds, hs):
    """根据检测置信度及手部位置选择商品框，并合并符合条件的碎片。

    参数：
        ds: 商品候选列表；hs: 检出的手框列表。
    返回：
        最终框、选择方式和合并候选列表；无法选择时返回两个 None 和空列表。
    """
    if not ds:
        return None, None, []
    first = max(ds, key=lambda d: d["score"])
    eligible = [d for d in ds if any(near(d["box"], h["box"]) for h in hs)]
    if first["score"] >= 0.05:
        chosen = first
        method = "highest confidence >= 0.05"
    else:
        chosen = max(eligible, key=lambda d: d["score"], default=None)
        method = "highest confidence near detected hand" if chosen else None
    if not chosen:
        return None, None, []
    selected = chosen["box"][:]
    merged = []
    for d in possible_merge(chosen, ds):
        if not any(near(d["box"], h["box"]) for h in hs):
            continue
        b = d["box"]
        selected = [
            min(selected[0], b[0]),
            min(selected[1], b[1]),
            max(selected[2], b[2]),
            max(selected[3], b[3]),
        ]
        merged.append({"box": b, "score": d["score"]})
    return selected, method, merged


def process(src, model, hands, out, write=True, gpu_hands=None):
    """检测单张图片中的手持商品，生成分割掩码和可选标注图。

    参数：
        src: 输入图片路径；model: YOLOE 模型；hands: CPU 手部检测器。
        out: 标注图片目录；write: 是否写出图片；gpu_hands: 可选 GPU 检测器。
    返回：
        候选框、手框、选中商品、编码掩码及各阶段耗时组成的字典。
    """
    total = time.perf_counter()
    times = {}
    t = time.perf_counter()
    im = cv2.imread(str(src))
    assert im is not None
    h, w = im.shape[:2]
    times["read_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    result = model.predict(
        im,
        device=0,
        imgsz=640,
        rect=False,
        conf=0.005,
        iou=0.5,
        max_det=300,
        verbose=False,
        predictor=NoWarmupPredictor,
    )[0]
    ds = [
        {"box": [float(x) for x in b], "score": float(s), "mask_index": i}
        for i, (b, s) in enumerate(
            zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
        )
    ]
    torch.cuda.synchronize()
    times["yoloe_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    ds = [d for d in ds if area(d["box"]) <= 0.5 * w * h]
    top = max(ds, key=lambda d: d["score"], default=None)
    hs = []
    passes = 0
    hand_ms = 0
    cpu_fallback = False
    gpu_error = None
    if top and not (top["score"] >= 0.05 and not possible_merge(top, ds)):
        for k in range(4):
            v = time.perf_counter()
            try:
                hs.extend(
                    gpu_hand_angle(im, k, gpu_hands)
                    if gpu_hands is not None
                    else hand_angle(im, k, hands)
                )
            except Exception as e:
                if gpu_hands is None:
                    raise
                gpu_error = str(e)
                hs = []
                hand_ms += (time.perf_counter() - v) * 1000
                break
            hand_ms += (time.perf_counter() - v) * 1000
            passes += 1
            eligible = [d for d in ds if any(near(d["box"], q["box"]) for q in hs)]
            chosen = (
                top
                if top["score"] >= 0.05
                else max(eligible, key=lambda d: d["score"], default=None)
            )
            if chosen is top and all(
                any(near(d["box"], q["box"]) for q in hs) for d in possible_merge(top, ds)
            ):
                break
    chosen = (
        top
        if top and top["score"] >= 0.05
        else max(
            (d for d in ds if any(near(d["box"], q["box"]) for q in hs)),
            key=lambda d: d["score"],
            default=None,
        )
    )
    trial, _, _ = decide(ds, hs)
    if (
        gpu_hands is not None
        and top
        and (gpu_error or trial is None or (top["score"] < 0.05 and trial != top["box"]))
    ):
        # GPU 检测失败或无法确定目标时，重新使用 CPU 检测手部。
        cpu_fallback = True
        hs = []
        passes = 0
        for k in range(4):
            v = time.perf_counter()
            hs.extend(hand_angle(im, k, hands))
            hand_ms += (time.perf_counter() - v) * 1000
            passes += 1
            eligible = [d for d in ds if any(near(d["box"], q["box"]) for q in hs)]
            chosen = (
                top
                if top["score"] >= 0.05
                else max(eligible, key=lambda d: d["score"], default=None)
            )
            if chosen is top and all(
                any(near(d["box"], q["box"]) for q in hs) for d in possible_merge(top, ds)
            ):
                break
        chosen = (
            top
            if top["score"] >= 0.05
            else max(
                (d for d in ds if any(near(d["box"], q["box"]) for q in hs)),
                key=lambda d: d["score"],
                default=None,
            )
        )
    box, method, merged = decide(ds, hs)
    mask_data = None
    if box:
        if result.masks is None:
            raise RuntimeError("商品检测没有分割 mask")
        indices = [
            d["mask_index"] for d in ds if d is chosen or any(d["box"] == p["box"] for p in merged)
        ]
        if not indices:
            raise RuntimeError("商品分割 mask 不存在")
        masks = result.masks.data[indices][None].float()
        scaled = ops.scale_masks(masks, (h, w), mode="nearest")[0].amax(0) > 0.5
        x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
        x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
        mask = scaled[y1:y2, x1:x2].cpu().numpy()
        if not mask.any():
            raise RuntimeError("商品分割 mask 为空")
        mask_data = encode_mask(mask)
    times["hands_ms"] = hand_ms
    times["selection_ms"] = (time.perf_counter() - t) * 1000 - hand_ms
    t = time.perf_counter()
    if box:
        x1, y1, x2, y2 = map(round, box)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 3)
        label = f"held product {chosen['score']:.6f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = max(0, min(x1, w - tw - 4))
        ty = max(th + 5, y1 - 5)
        cv2.putText(im, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            im, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA
        )
    footer = f"{src.name} | held product | conf > 0.005"
    (tw, th), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(im, (max(0, w - tw - 20), h - th - 18), (w, h), (0, 0, 0), -1)
    cv2.putText(
        im,
        footer,
        (max(0, w - tw - 10), h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    times["draw_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    if write:
        assert cv2.imwrite(str(out / src.name), im)
    times["write_ms"] = (time.perf_counter() - t) * 1000
    times["total_ms"] = (time.perf_counter() - total) * 1000
    return {
        "source": src.name,
        "candidates": ds,
        "hand_detections": hs,
        "hands_passes": passes,
        "hand_backend": "gpu_with_cpu_fallback" if gpu_hands is not None else "cpu",
        "cpu_fallback": cpu_fallback,
        "gpu_error": gpu_error,
        "selected": chosen,
        "selection_method": method,
        "merged_parts": merged,
        "rendered_box": box,
        "product_mask": mask_data,
        "timing_ms": times,
    }


def make_worker_pool(model, hands, count=1, backend="cpu"):
    """创建逐帧检测工作池，复用首个模型并为其余工作线程加载独立实例。

    参数：
        model: 已加载的 YOLOE 商品模型。
        hands: 与首个工作线程配套的 CPU 手部检测器。
        count: 工作线程数量，范围为 1 到 32。
        backend: 手部检测后端，取值为 cpu 或 gpu。
    返回：
        每项为商品模型、CPU 手部检测器、可选 GPU 手部检测器和 CUDA 流的元组。
    异常：
        参数无效或初始化失败时抛出异常；失败时关闭已加入工作池的手部检测器。
    """
    if count < 1 or count > 32:
        raise ValueError("worker count must be 1..32")
    if backend not in ("cpu", "gpu"):
        raise ValueError("unknown hand backend")
    embedding = model.model.pe.detach().cpu().clone()
    workers = []
    try:
        for i in range(count):
            if i == 0:
                m, h = model, hands
            else:
                m = YOLOE(str(config.HELD_PRODUCT_MODEL))
                m.set_classes(["held product"], embeddings=embedding)
                args = {
                    **m.overrides,
                    "conf": 0.005,
                    "device": 0,
                    "imgsz": 640,
                    "rect": False,
                    "iou": 0.5,
                    "max_det": 300,
                    "verbose": False,
                    "mode": "predict",
                    "save": False,
                }
                m.predictor = NoWarmupPredictor(overrides=args, _callbacks=m.callbacks)
                m.predictor.setup_model(model=m.model, verbose=False)
                h = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=6,
                    model_complexity=1,
                    min_detection_confidence=0.25,
                    min_tracking_confidence=0.25,
                )
            g = load_gpu_hands() if backend == "gpu" else None
            workers.append((m, h, g, torch.cuda.Stream()))
        return workers
    except BaseException:
        for m, h, g, stream in workers:
            if g:
                g.close()
            h.close()
        raise


def run_detection(
    source,
    out,
    result_path,
    model,
    hands,
    cabinet_id,
    log_path=None,
    initialization_ms=0,
    gpu_hands=None,
    render=True,
    worker_pool=None,
):
    """处理目录中的图像帧，输出逐帧结果和耗时统计。

    参数：
        source: 输入图像目录。
        out: 标注图片输出目录；运行前清空其中已有内容。
        result_path: 完整 JSON 报告路径。
        model, hands: 商品检测模型及 CPU 手部检测器。
        log_path: 可选的汇总日志路径。
        initialization_ms: 模型初始化耗时，记入报告。
        gpu_hands: 可选 GPU 手部检测器。
        render: 是否保存标注后的图片。
        worker_pool: 可选的并行工作池；不提供时串行处理。
    返回：
        包含逐帧结果、耗时及统计信息的报告字典。
    异常：
        输出目录包含输入或报告，或者输入目录无图片时抛出 ValueError。
    """
    from app.catalog.product_catalog import validate_cabinet_id

    validate_cabinet_id(cabinet_id)
    source = Path(source)
    out = Path(out)
    result_path = Path(result_path)
    if (
        out.resolve() == source.resolve()
        or out.resolve() in source.resolve().parents
        or out.resolve() in result_path.resolve().parents
    ):
        raise ValueError("Output directory cannot contain input frames or report")
    files = sorted(
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
    if not files:
        raise ValueError("No input frames")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
    rows = []
    processing_start = time.perf_counter()
    if worker_pool is None:
        for i, p in enumerate(files):
            row = process(p, model, hands, out, write=render, gpu_hands=gpu_hands)
            rows.append(row)
            if (i + 1) % 50 == 0:
                print("processed", i + 1, "/", len(files), flush=True)
    else:
        import concurrent.futures
        import queue

        available = queue.Queue()
        for item in worker_pool:
            available.put(item)

        def run_frame(path):
            """从队列借用工作实例处理一帧，并在结束后归还。"""
            worker_model, worker_hands, worker_gpu, stream = available.get()
            try:
                with torch.cuda.stream(stream):
                    return process(
                        path, worker_model, worker_hands, out, write=render, gpu_hands=worker_gpu
                    )
            finally:
                available.put((worker_model, worker_hands, worker_gpu, stream))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_pool)) as executor:
            for i, row in enumerate(executor.map(run_frame, files)):
                rows.append(row)
                if (i + 1) % 50 == 0:
                    print("processed", i + 1, "/", len(files), flush=True)
    wall = time.perf_counter() - processing_start
    keys = rows[0]["timing_ms"]
    avg = {k: round(statistics.mean(r["timing_ms"][k] for r in rows), 3) for k in keys}
    passes = {str(i): sum(r["hands_passes"] == i for r in rows) for i in range(5)}
    report = {
        "cabinet_id": cabinet_id,
        "source": str(source.relative_to(R)) if source.is_relative_to(R) else str(source),
        "frames": len(rows),
        "baseline": None,
        "initialization_ms": initialization_ms,
        "warmup_performed": False,
        "processing_wall_seconds": wall,
        "mean_ms": avg,
        "median_ms": statistics.median(r["timing_ms"]["total_ms"] for r in rows),
        "p95_ms": float(np.percentile([r["timing_ms"]["total_ms"] for r in rows], 95)),
        "rotation_passes": passes,
        "baseline_mismatch_count": None,
        "baseline_mismatches": [],
        "drawn": sum(r["rendered_box"] is not None for r in rows),
        "images": rows,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    result_path.write_text(encoded)
    summary = json.dumps(
        {k: v for k, v in report.items() if k not in ("images", "baseline_mismatches")},
        ensure_ascii=False,
        indent=2,
    )
    if log_path:
        Path(log_path).write_text(summary + "\n")
    print(summary, flush=True)
    return report


def main():
    """解析命令行路径、加载检测模型并执行检测，结束时关闭手部检测器。"""
    parser = argparse.ArgumentParser(description="Detect products on video frames")
    parser.add_argument("--cabinet-id", required=True, help="柜子编码")
    parser.add_argument("--images", required=True, help="待检测视频帧目录")
    parser.add_argument("--output", help="标注帧输出目录")
    parser.add_argument("--result", help="检测结果 JSON")
    parser.add_argument("--log", help="检测汇总日志")
    args = parser.parse_args()

    from app.catalog.product_catalog import validate_cabinet_id

    validate_cabinet_id(args.cabinet_id)
    base = Path("tests/result") / args.cabinet_id
    args.output = args.output or base / "detection_frames"
    args.result = args.result or base / "detections.json"
    args.log = args.log or base / "held_product_detect.log"

    def absolute(p):
        """将相对路径解析到工作区根目录，绝对路径保持不变。"""
        p = Path(p)
        return p if p.is_absolute() else R / p

    source, out, result, log = map(absolute, (args.images, args.output, args.result, args.log))
    if out.resolve() == log.resolve() or out.resolve() in log.resolve().parents:
        raise ValueError("Output directory cannot contain log")
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    model, hands = load_detector()
    initialization_ms = (time.perf_counter() - t) * 1000
    try:
        run_detection(source, out, result, model, hands, args.cabinet_id, log, initialization_ms)
    finally:
        hands.close()


if __name__ == "__main__":
    from app.logging import setup_logging

    setup_logging("held-product")
    main()
