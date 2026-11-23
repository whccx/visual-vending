"""实验性视频识别流水线：NVDEC 解码、商品和手检测、DINOv3 特征匹配。"""

import ctypes
import gc
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import nms, ops

from app.detection.held_product_detect import area, decide, gpu_hand_angle, near, possible_merge
from app.catalog.product_catalog import load_features
from app.recognition.product_matching import (
    prepare_catalog,
    score_crop,
    select_scoring_window,
)
from app.detection.person_gate_policy import PersonGatePolicy
from app.tracking.stable_trajectory import select_stable_track
from app.detection.hand_area_gate import filter_hand_linked_candidates
from app.detection.area_detection import detect_frame_box

# GPU 视频依赖保存在运行时目录，重建容器后仍可使用。
from app import config

_runtime = config.GPU_RUNTIME
sys.path.insert(0, str(_runtime / "nvcodec"))
ctypes.CDLL(str(_runtime / "nvidia-video/libnvcuvid.so.1"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(_runtime / "nvidia-video/libnvidia-encode.so.1"), mode=ctypes.RTLD_GLOBAL)
# 依赖库和模块路径就绪后才能导入解码器。
import PyNvVideoCodec as nvc  # noqa: E402


class GPUVideoPipeline:
    """使用 GPU 缓存和多阶段并行处理视频，返回商品识别与轨迹结果。"""

    def __init__(self, detector, dino, hands, batch_size=16):
        """保存检测模型，并创建可在多次请求间复用的 CUDA 流。"""
        self.detector, self.dino, self.hands = detector, dino, hands
        self.batch_size = batch_size
        # 复用 CUDA 流，避免每次请求新建流时持续保留库工作区。
        self.yolo_stream = torch.cuda.Stream()
        self.dino_stream = torch.cuda.Stream()

    def decode(self, path):
        """将视频解码到 GPU 缓存，返回帧张量和缓存命中标记。"""
        decoder = nvc.SimpleDecoder(
            str(path), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
        )
        count = len(decoder)
        if not count:
            raise ValueError("视频没有可解码的帧")
        cache = None
        i = 0
        while i < count:
            frames = decoder.get_batch_frames(min(self.batch_size, count - i))
            if not frames:
                raise ValueError("视频解码提前结束")
            for frame in frames:
                image = torch.from_dlpack(frame)
                if cache is None:
                    size = count * image.numel() * image.element_size()
                    free, _ = torch.cuda.mem_get_info()
                    # 预留 1 GiB 显存，供后续推理和其他计算使用。
                    if size > free - 1024**3:
                        raise ValueError("视频解码帧超过可用显存，请缩短视频")
                    cache = torch.empty(
                        (count, *image.shape), dtype=image.dtype, device=image.device
                    )
                if image.shape != cache.shape[1:]:
                    raise ValueError("暂不支持分辨率中途变化的视频")
                cache[i].copy_(image)
                i += 1
        torch.cuda.synchronize()
        return cache, False

    def run(
        self, path, cabinet_id, person_flags=None, area_boxes=None, area_model=None,
        capture_visualization=False,
    ):
        """执行识别，并在成功或失败后清理本次请求的 GPU 缓存。"""
        try:
            return self._run(
                path, cabinet_id, person_flags, area_boxes, area_model,
                capture_visualization=capture_visualization,
            )
        finally:
            # 清理请求中形成的循环引用，并将未使用的显存块归还给 CUDA。
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def _run(
        self, path, cabinet_id, person_flags=None, area_boxes=None, area_model=None,
        capture_visualization=False,
    ):
        """并行执行检测、手部关联和特征匹配，再筛选稳定轨迹。"""
        import queue
        import threading
        from concurrent.futures import ThreadPoolExecutor

        start = time.perf_counter()
        cache, hit = self.decode(path)
        decode_s = time.perf_counter() - start
        # 按比例缩放并填充至 640 × 640，记录填充量以便还原分割掩码。
        count, _, h, w = cache.shape
        scale = min(640 / w, 640 / h)
        nw, nh = round(w * scale), round(h * scale)
        left = (640 - nw) // 2
        right = 640 - nw - left
        top = (640 - nh) // 2
        bottom = 640 - nh - top
        if person_flags is not None and len(person_flags) != count:
            logging.getLogger("sku." + __name__).warning(
                "人物扫描帧数与 NVDEC 不一致，将逐帧处理"
            )
            person_flags = None
        if area_boxes is not None and len(area_boxes) != count:
            raise ValueError("柜内区域检测与视频帧数不一致")
        if area_model is not None:
            if area_boxes is not None:
                raise ValueError("不能同时提供区域模型和区域检测结果")
            area_boxes = [None] * count
        gate = PersonGatePolicy(person_flags)
        catalog = load_features(cabinet_id)
        matching_catalog = prepare_catalog(catalog)
        reference_hues = {sku: item["colors"] for sku, item in catalog.items()}
        features = matching_catalog[0]
        torch.cuda.synchronize()
        # 三个有界队列连接处理阶段；停止事件用于向其他线程传播异常。
        yolo_queue = queue.Queue(maxsize=4)
        hands_queue = queue.Queue(maxsize=32)
        dino_queue = queue.Queue(maxsize=32)
        stopped = threading.Event()
        end = object()
        rows = [None] * count
        hands_by_frame = [[] for _ in range(count)]
        scores = {}
        embeddings = {}
        metrics = {"readback": 0, "scoring_readback": 0, "rotations": 0,
                   "yolo": 0.0, "hands": 0.0, "dino": 0.0}

        def put(q, item):
            while not stopped.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def get(q):
            while not stopped.is_set():
                try:
                    return q.get(timeout=0.1)
                except queue.Empty:
                    pass
            return end

        def guarded(fn):
            try:
                fn()
            except BaseException:
                stopped.set()
                raise

        def yolo_worker():
            """批量检测商品，并将每帧候选框和分割掩码送往手部检测。"""
            stream = self.yolo_stream
            with torch.inference_mode(), torch.cuda.stream(stream):
                while True:
                    start = get(yolo_queue)
                    if start is end:
                        break
                    tick = time.perf_counter()
                    indices = list(range(start, min(count, start + self.batch_size)))
                    selected = [i for i in indices if gate.allow(i)]
                    results = {}
                    if selected:
                        raw = cache[selected]
                        x = (
                            F.interpolate(
                                raw.float(), size=(nh, nw), mode="bilinear", align_corners=False
                            )
                            .round()
                            .div_(255)
                        )
                        x = F.pad(x, (left, right, top, bottom), value=114 / 255)
                        out = self.detector.predictor.model(x)
                        preds = nms.non_max_suppression(
                            out[0],
                            conf_thres=0.005,
                            iou_thres=0.5,
                            max_det=300,
                            nc=len(self.detector.predictor.model.names),
                            end2end=getattr(self.detector.predictor.model, "end2end", False),
                        )
                        protos = out[0][1]
                    else:
                        preds = []
                    for off, pred in enumerate(preds):
                        if len(pred):
                            masks = ops.process_mask(
                                protos[off], pred[:, 6:], pred[:, :4], x.shape[2:], upsample=True
                            )
                            valid = masks.amax((-2, -1)) > 0
                            pred = pred[valid]
                            masks = masks[valid]
                            pred[:, :4] = ops.scale_boxes(x.shape[2:], pred[:, :4], (h, w))
                        ds = [
                            {
                                "box": [float(q) for q in r[:4]],
                                "score": float(r[4]),
                                "_mask": masks[j],
                            }
                            for j, r in enumerate(pred[:, :6].cpu().numpy())
                        ]
                        ds = [v for v in ds if area(v["box"]) <= 0.5 * w * h]
                        results[selected[off]] = ds
                    for idx in indices:
                        ds = results.get(idx, [])
                        if idx in results:
                            gate.observe(bool(ds))
                        put(hands_queue, (idx, ds))
                    stream.synchronize()
                    metrics["yolo"] += time.perf_counter() - tick
            put(hands_queue, end)

        def hands_worker():
            """从四个旋转方向检测手部，去重后选择与手关联的商品。"""
            while True:
                item = get(hands_queue)
                if item is end:
                    break
                tick = time.perf_counter()
                idx, ds = item
                if not ds:
                    put(dino_queue, (idx, None, []))
                    continue
                best = max(ds, key=lambda a: a["score"], default=None)
                hs = []
                all_hands = []
                selection_active = bool(
                    best and not (best["score"] >= 0.05 and not possible_merge(best, ds))
                )
                rgb = cache[idx].permute(1, 2, 0).contiguous().cpu().numpy()
                metrics["readback"] += rgb.nbytes
                im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if area_model is not None:
                    area_boxes[idx], _ = detect_frame_box(im, area_model)
                for k in range(4):
                    detected_hands = gpu_hand_angle(im, k, self.hands)
                    for hand in detected_hands:
                        # 不同旋转方向可能检测到同一只手，按重叠比例去重。
                        if not any(
                            (
                                max(
                                    0,
                                    min(hand["box"][2], old["box"][2])
                                    - max(hand["box"][0], old["box"][0]),
                                )
                                * max(
                                    0,
                                    min(hand["box"][3], old["box"][3])
                                    - max(hand["box"][1], old["box"][1]),
                                )
                            )
                            / max(1.0, min(area(hand["box"]), area(old["box"])))
                            > 0.65
                            for old in all_hands
                        ):
                            all_hands.append(hand)
                    if selection_active:
                        hs.extend(detected_hands)
                        eligible = [q for q in ds if any(near(q["box"], v["box"]) for v in hs)]
                        chosen = (
                            best
                            if best["score"] >= 0.05
                            else max(eligible, key=lambda q: q["score"], default=None)
                        )
                        if chosen is best and all(
                            any(near(q["box"], v["box"]) for v in hs)
                            for q in possible_merge(best, ds)
                        ):
                            selection_active = False
                    metrics["rotations"] += 1
                hands_by_frame[idx] = all_hands
                chosen = (
                    best
                    if best and best["score"] >= 0.05
                    else max(
                        (d for d in ds if any(near(d["box"], hand["box"]) for hand in hs)),
                        key=lambda d: d["score"],
                        default=None,
                    )
                )
                box, _, merged = decide(ds, hs)
                selected_masks = (
                    [
                        d["_mask"]
                        for d in ds
                        if box and (d is chosen or any(d["box"] == p["box"] for p in merged))
                    ]
                    if best
                    else []
                )
                metrics["hands"] += time.perf_counter() - tick
                put(dino_queue, (idx, box, selected_masks))
            put(dino_queue, end)

        def dino_worker():
            """还原商品掩码，提取图像特征并计算候选商品得分。"""
            stream = self.dino_stream
            with torch.inference_mode(), torch.cuda.stream(stream):
                mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
                while True:
                    item = get(dino_queue)
                    if item is end:
                        break
                    tick = time.perf_counter()
                    idx, box, selected_masks = item
                    rows[idx] = box
                    if box and catalog:
                        x1, y1, x2, y2 = box
                        x1 = max(0, int(np.floor(x1)))
                        y1 = max(0, int(np.floor(y1)))
                        x2 = min(w, int(np.ceil(x2)))
                        y2 = min(h, int(np.ceil(y2)))
                        crop = cache[idx, :, y1:y2, x1:x2]
                        if not crop.numel():
                            raise ValueError("商品检测框为空")
                        if not selected_masks:
                            raise ValueError("商品分割结果为空")
                        # 掩码由另一个 CUDA 流创建，记录当前流以保证显存生命周期。
                        for selected_mask in selected_masks:
                            selected_mask.record_stream(stream)
                        merged_mask = torch.stack(selected_masks).any(0)
                        mask = merged_mask[top : 640 - bottom, left : 640 - right][
                            None, None
                        ].float()
                        mask = (
                            F.interpolate(mask, size=(h, w), mode="nearest")[0, 0, y1:y2, x1:x2]
                            > 0.5
                        )
                        if not mask.any():
                            raise ValueError("商品分割区域为空")
                        q, rank = score_crop(
                            crop, mask, self.dino, matching_catalog, mean, std,
                            reference_hues=reference_hues,
                        )
                        embeddings[idx + 1] = q.detach().cpu().numpy()
                        metrics["scoring_readback"] += (
                            crop.numel() * crop.element_size() + mask.numel() * mask.element_size()
                        )
                        stream.synchronize()
                        scores[idx + 1] = rank
                    metrics["dino"] += time.perf_counter() - tick

        pipeline_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="gpu-pipeline") as pool:
            futures = [pool.submit(guarded, fn) for fn in (yolo_worker, hands_worker, dino_worker)]
            for offset in range(0, count, self.batch_size):
                put(yolo_queue, offset)
            put(yolo_queue, end)
            for future in futures:
                future.result()
        pipeline_s = time.perf_counter() - pipeline_start
        logging.getLogger("sku." + __name__).info("YOLOE 人物检测统计：%s", gate.summary())
        motion_start = time.perf_counter()
        hand_cache = {}
        max_hand_area_ratio = config.max_product_hand_area_ratio()

        def hand_info(frame, box):
            """缓存轨迹帧的邻近手部及符合面积比例的手部检测结果。"""
            if frame not in hand_cache:
                rgb = cache[frame - 1].permute(1, 2, 0).contiguous().cpu().numpy()
                im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                nearby = []
                plausible = []
                for k in range(4):
                    hs = gpu_hand_angle(im, k, self.hands)
                    nearby.extend(hand for hand in hs if near(box, hand["box"]))
                    plausible = [
                        hand for hand in nearby
                        if area(box) <= max_hand_area_ratio * max(1.0, area(hand["box"]))
                    ]
                    if plausible:
                        break
                hand_cache[frame] = (nearby, plausible)
            return hand_cache[frame]

        if area_boxes is None:
            # 独立诊断调用可能不提供区域；识别 API 通常会提供。
            hand_area = {"applied": False}
            gated_rows = rows
            gated_track_ids = None
        else:
            hand_candidates = [
                [
                    {
                        "box": box,
                        "hand_indices": [
                            j for j, hand in enumerate(hands_by_frame[i]) if near(box, hand["box"])
                        ],
                    }
                ]
                if box
                else []
                for i, box in enumerate(rows)
            ]
            eligible, hand_area = filter_hand_linked_candidates(
                hands_by_frame, hand_candidates, area_boxes, embeddings_by_frame=embeddings
            )
            gated_rows = [items[0]["box"] if items else None for items in eligible]
            gated_track_ids = [items[0]["area_track_id"] if items else None for items in eligible]
        logging.getLogger("sku." + __name__).info("YOLOE 手部区域筛选统计：%s", hand_area)
        primary, motion_summary = select_stable_track(
            path, gated_rows, w, h, embeddings, product_track_ids=gated_track_ids
        )
        if not primary or len(primary["points"]) < 5:
            primary, motion_summary = select_stable_track(
                path,
                gated_rows,
                w,
                h,
                embeddings,
                close=0.25,
                confirm=2,
                product_track_ids=gated_track_ids,
            )
            motion_summary["relaxed_fallback"] = True
        if primary:
            # 移动的人头可能被误认为商品：有邻近手部时按面积比例排除异常框；
            # 仅凭未检测到手，不能判定商品框无效。
            checked = []
            size_rejected = 0
            hand_missing = 0
            for point in primary["points"]:
                box = point["box"]
                nearby, plausible = hand_info(point["frame"], box)
                if nearby and not plausible:
                    size_rejected += 1
                    continue
                if not nearby:
                    hand_missing += 1
                checked.append(point)
            primary = dict(primary, points=checked)
            motion_summary["hand_size_rejected"] = size_rejected
            motion_summary["hand_not_detected"] = hand_missing
            motion_summary["rankable_points"] = len(checked)
        motion_seconds = time.perf_counter() - motion_start
        logging.getLogger("sku." + __name__).info("YOLOE 运动验证统计：%s", motion_summary)
        selected = (
            [(pt["frame"], scores[pt["frame"]]) for pt in primary["points"] if pt["frame"] in scores]
            if primary
            else []
        )
        selected_frames, result = select_scoring_window(selected, features)
        logging.getLogger("sku." + __name__).info(
            "YOLOE 色相比对：候选帧=%s 连续窗口=%s", len(selected), selected_frames
        )
        output = {
            "frames": count,
            "track_points": len(selected_frames),
            "ranking": result,
            "hand_area": hand_area,
            "cache_hit": hit,
            "cache_bytes": cache.numel() * cache.element_size(),
            "gpu_to_cpu_image_bytes": metrics["readback"] + metrics["scoring_readback"],
            "hand_rotations": metrics["rotations"],
            "decode_seconds": decode_s,
            "detection_seconds": pipeline_s,
            "comparison_seconds": 0.0,
            "pipeline_seconds": pipeline_s,
            "motion_seconds": motion_seconds,
            "motion_summary": motion_summary,
            "person_gate": gate.summary(),
            "stage_seconds": {
                "yoloe": metrics["yolo"],
                "mediapipe": metrics["hands"],
                "dinov3": metrics["dino"],
            },
        }
        if capture_visualization:
            from app.recognition.yoloe_visualization import build_visualization

            output["visualization"] = build_visualization(rows, hands_by_frame, primary)
        return output
