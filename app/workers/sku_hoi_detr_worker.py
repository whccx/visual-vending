"""在独立 Python 环境中运行 HOI-DETR，检测视频中需要处理的帧。"""

import argparse
import copy
import json
import logging
import os
import sys
import threading
import contextlib
from pathlib import Path

from app import config as app_config

ROOT = app_config.ROOT
REPO = app_config.HOI_REPO
os.chdir(str(REPO))
sys.path[:0] = [str(REPO), str(REPO / "demo")]
import cv2
import torch
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose
from projects import *  # noqa: F401,F403
from configs import CLASS_NAMES
from helpers import find_interaction_branch, call_interaction
from app.workers.hoi_detr_inference_memory import run_inference_batch
from app.workers.hoi_batch_scheduler import ordered_parallel_map
from app.detection.person_gate_policy import PersonGatePolicy

LOG = logging.getLogger("sku.hoi_detr_worker")


def load_runtime():
    """加载并预热单个常驻 HOI 模型。"""
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    config = "projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py"
    model = init_detector(config, str(app_config.HOI_MODEL), device="cuda:0").eval()
    # 自定义可变形注意力算子不支持半精度，仅骨干和颈部使用混合精度。
    original_extract = model.extract_feat

    def extract_mixed(image):
        """使用混合精度提取特征，并转回 float32 供后续检测头使用。"""
        with torch.cuda.amp.autocast():
            features = original_extract(image)
        return tuple(feature.float() for feature in features)

    model.extract_feat = extract_mixed
    pipeline_config = copy.deepcopy(model.cfg.data.test.pipeline)
    pipeline_config[0]["type"] = "LoadImageFromWebcam"
    pipeline = Compose(pipeline_config)
    branch = find_interaction_branch(model.query_head)
    import numpy as np

    with torch.inference_mode():
        run_inference_batch(
            model, pipeline, [np.zeros((1080, 1920, 3), dtype=np.uint8)],
            device="cuda:0", class_names=CLASS_NAMES, score_thr=0.3, nms_iou=0.5,
        )
    torch.cuda.synchronize()
    LOG.info("HOI-DETR 模型预热完成：进程=%s", os.getpid())
    return model, pipeline, branch


def run_video(video, output, runtime, start=0, end=None, person_flags=None,
              batch_size=None, batch_workers=None):
    """使用已加载模型检测视频中的半开帧区间。"""
    LOG.info("HOI-DETR 检测开始：范围=%s:%s", start, end)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("无法读取视频")
    if start and not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        cap.release()
        raise ValueError("无法定位视频帧")
    gate = PersonGatePolicy(person_flags)
    model, pipeline, branch = runtime
    frames = []
    hand_frames = []
    batch_size = app_config.hoi_detr_batch_size() if batch_size is None else batch_size
    batch_workers = app_config.hoi_detr_batch_workers() if batch_workers is None else batch_workers
    if not 1 <= batch_size <= 8:
        raise ValueError("SKU_HOI_DETR_BATCH_SIZE 必须在 1 到 8 之间")
    if not 1 <= batch_workers <= 4:
        raise ValueError("SKU_HOI_DETR_BATCH_WORKERS 必须在 1 到 4 之间")
    thread_state = threading.local()

    def read_batches():
        """顺序读取视频，最后一批允许不足指定帧数。"""
        read_count = 0
        while True:
            images = []
            for _ in range(batch_size):
                if end is not None and start + read_count >= end:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                images.append(frame)
                read_count += 1
            if not images:
                return
            # 单批模式沿用原有人物门控，避免向 GPU 送入无人物的帧。
            decisions = (
                [gate.allow(len(frames) + i) for i in range(len(images))]
                if batch_workers == 1 else None
            )
            yield images, decisions

    def infer_batch(batch):
        """共享模型，使用线程独立的 CUDA 流并等待本批推理完成。"""
        images, decisions = batch
        selected = images if decisions is None else [
            image for image, allowed in zip(images, decisions) if allowed
        ]
        if not hasattr(thread_state, "stream"):
            thread_state.stream = torch.cuda.Stream(device="cuda:0")
        with torch.inference_mode(), torch.cuda.stream(thread_state.stream):
            output = run_inference_batch(
                model, pipeline, selected, device="cuda:0", class_names=CLASS_NAMES,
                score_thr=0.3, nms_iou=0.5,
            )
            thread_state.stream.synchronize()
        return len(images), decisions, output

    LOG.info("HOI-DETR 并发批次：批次数=%s 每批帧数=%s", batch_workers, batch_size)
    try:
        for size, decisions, results in ordered_parallel_map(infer_batch, read_batches(), batch_workers):
            with torch.inference_mode():
                # 预先推理全部帧，等前序批次完成后再应用门控，避免因过期状态漏检。
                if decisions is None:
                    decisions = [gate.allow(len(frames) + i) for i in range(size)]
                    # 多批次模式的结果包含跳过的帧，直接按原索引取用。
                    ordered_results = [
                        result if allowed else None for allowed, result in zip(decisions, results)
                    ]
                else:
                    batch = iter(results)
                    ordered_results = [next(batch) if allowed else None for allowed in decisions]
                for allowed, result in zip(decisions, ordered_results):
                    if not allowed:
                        frames.append([])
                        hand_frames.append([])
                        continue
                    dets, embeds = result
                    hands = [d for d in dets if d["class_id"] == 0]
                    hand_frames.append([{"box": [float(v) for v in hand["box"]]} for hand in hands])
                    objects = [d for d in dets if d["class_id"] == 1]
                    candidates = []
                    has_linked_candidate = False
                    for obj in objects:
                        links = [
                            call_interaction(
                                branch, embeds[hand["query_idx"]], embeds[obj["query_idx"]]
                            )
                            for hand in hands
                        ]
                        linked_indices = [i for i, (linked, _) in enumerate(links) if linked]
                        has_linked_candidate = has_linked_candidate or bool(linked_indices)
                        candidates.append(
                            {
                                "box": [float(v) for v in obj["box"]],
                                "score": float(obj["score"]),
                                "hand_indices": linked_indices,
                            }
                        )
                    frames.append(candidates)
                    gate.observe(has_linked_candidate)
    finally:
        cap.release()
    if not frames:
        raise ValueError("视频没有可解码的帧")
    if end is not None and len(frames) != end - start:
        raise ValueError("视频帧范围不完整：预期 %s 实际 %s" % (end - start, len(frames)))
    Path(output).write_text(
        json.dumps({"frames": frames, "hands": hand_frames, "person_gate": gate.summary()}),
        encoding="utf-8",
    )
    LOG.info("HOI-DETR 检测完成：帧数=%d", len(frames))


def serve():
    """使用行协议处理多个视频请求，模型在进程存活期间常驻。"""
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        runtime = load_runtime()
    protocol.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n")
    protocol.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                try:
                    run_video(request["video"], request["output"], runtime,
                              start=request["start"], end=request["end"],
                              batch_size=1, batch_workers=app_config.HOI_POOL_THREADS)
                finally:
                    # 只归还空闲的临时显存；runtime 持有的模型参数不会释放。
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            reply = {"ok": True, "pid": os.getpid()}
        except Exception as exc:
            LOG.exception("HOI-DETR 常驻进程处理失败")
            reply = {"error": str(exc), "pid": os.getpid()}
        protocol.write(json.dumps(reply) + "\n")
        protocol.flush()


def main():
    """兼容一次性 CLI，并提供常驻模式。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--person-flags")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve()
        return
    if not args.video or not args.output:
        parser.error("需要视频和输出路径")
    flags = None
    if args.person_flags:
        try:
            flags = json.loads(Path(args.person_flags).read_text(encoding="utf-8"))
            if not isinstance(flags, list) or not all(isinstance(v, bool) for v in flags):
                raise ValueError("人物检测数据无效")
        except Exception:
            LOG.exception("人物检测数据无效，将逐帧处理")
            flags = None
    run_video(args.video, args.output, load_runtime(), person_flags=flags)


if __name__ == "__main__":
    from app.logging import setup_logging

    setup_logging("hoi-detr")
    main()
