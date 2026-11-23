"""独立的 100DOH 商品识别接口及隔离运行的检测进程。"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import torch
from app.detection import area_detection
from app.config import VIDEO_DIR
from app.api.cabinet_validation import require_known_cabinet
from app import config
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.recognition.product_matching import (
    accept_ranking,
    prepare_catalog,
)
from app.recognition.hoi_mask_matching import rank_masked_track
from app.catalog.product_catalog import load_features
from app.api.video_storage import store_video
from fastapi import Form
from app.detection.person_gate_policy import PersonGatePolicy
from app.detection.person_gate_client import scan_person_gate
from app.tracking.motion_rank_quality import select_motion_rank_points
from app.detection.hand_area_gate import filter_hand_linked_candidates
from app.tracking.trajectory_100doh import ProductTracker, select_product_track

ROOT = config.ROOT
JOBS = config.JOBS
EXPERIMENT = config.DOH_EXPERIMENT
CHECKPOINT = config.DOH_CHECKPOINT
LOG = logging.getLogger("sku." + __name__)
RESOURCE_LOCK = threading.Lock()
router = APIRouter()


class Response100DOH(BaseModel):
    """100DOH 视频识别接口的响应数据。"""

    cabinet_id: str

    status: str
    product: Optional[str] = None
    confidence: Optional[float] = None
    vector_score: Optional[float] = None
    color_score: Optional[float] = None
    track_points: int = 0
    frames: int
    video_path: str
    duration: float


def _save_upload(upload, original, job, cabinet_id):
    """在指定柜子下保存或复用视频。"""
    return store_video(upload, original, job, cabinet_id, VIDEO_DIR)


def _box_area(box):
    """计算矩形框面积，负宽度或高度按零处理。"""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _linked_candidates(result):
    """筛选位置、面积有效且与手部关联的可携带物体，按置信度降序返回。"""
    candidates = []
    for link in result["links"]:
        hand = result["hands"][link["hand_index"]]
        obj = result["objects"][link["object_index"]]
        if hand["state"] != "portable_object":
            continue
        hb, ob = hand["box"], obj["box"]
        hc = ((hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2)
        oc = ((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2)
        hand_diag = math.hypot(hb[2] - hb[0], hb[3] - hb[1])
        if math.dist(hc, oc) > max(30.0, 1.5 * hand_diag) or _box_area(ob) > 4 * max(
            1.0, _box_area(hb)
        ):
            continue
        if not all(math.isfinite(v) for v in ob) or _box_area(ob) <= 0:
            continue
        candidates.append(
            {
                "box": ob,
                "score": obj["score"],
                "hand_score": hand["score"],
                "object_index": link["object_index"],
                "hand_indices": [link["hand_index"]],
            }
        )
    return sorted(candidates, key=lambda x: (x["score"], x["hand_score"]), reverse=True)


def _all_object_candidates(result, linked):
    """合并同一商品的手部关联，并加入未关联商品以延续已建立的轨迹。"""
    grouped = {}
    for candidate in linked:
        index = candidate["object_index"]
        if index in grouped:
            grouped[index]["hand_indices"].extend(candidate["hand_indices"])
        else:
            grouped[index] = dict(candidate)
    for index, obj in enumerate(result["objects"]):
        box = obj["box"]
        if (
            index not in grouped
            and len(box) == 4
            and all(math.isfinite(v) for v in box)
            and _box_area(box) > 0
        ):
            grouped[index] = {"box": box, "score": obj["score"], "hand_indices": []}
    return sorted(grouped.values(), key=lambda x: x["score"], reverse=True)


def _detect_worker(video, output, person_flags_path=None):
    """在隔离进程中运行 100DOH 检测，并将逐帧候选写入 JSON。

    参数：
        video: 视频路径；output: 输出 JSON 路径。
        person_flags_path: 可选的人物扫描标记文件，数据无效时回退为逐帧检测。
    """
    LOG.info("100DOH 检测开始")
    # 上游代码会修改 Python 模块路径，因此仅在检测进程中导入。
    sys.path.insert(0, str(EXPERIMENT))
    from test_100doh import infer, load_model

    if not torch.cuda.is_available():
        raise RuntimeError("100DOH 需要 CUDA GPU")
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    device = torch.device("cuda")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("无法读取视频")
    person_flags = None
    if person_flags_path:
        try:
            data = json.loads(Path(person_flags_path).read_text(encoding="utf-8"))
            expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (
                not isinstance(data, list)
                or not all(isinstance(v, bool) for v in data)
                or (expected > 0 and expected != len(data))
            ):
                raise ValueError("人体检测与视频帧数不一致")
            person_flags = data
        except Exception:
            LOG.exception("人物检测数据无效，将逐帧处理")
    gate = PersonGatePolicy(person_flags)
    if person_flags is not None and not any(person_flags):
        cap.release()
        output.write_text(
            json.dumps(
                {
                    "frames": len(person_flags),
                    "boxes": [None] * len(person_flags),
                    "hands": [[] for _ in person_flags],
                    "candidates": [[] for _ in person_flags],
                    "person_gate": {"sent_to_detector": 0, "skipped_no_person": len(person_flags)},
                }
            ),
            encoding="utf-8",
        )
        LOG.info("100DOH 跳过检测：帧数=%d", len(person_flags))
        return
    try:
        model, classes = load_model(CHECKPOINT, device)
    except BaseException:
        cap.release()
        raise
    boxes = []
    hands_by_frame = []
    candidates_by_frame = []
    workers = config.doh_workers()
    if not 1 <= workers <= 4:
        raise ValueError("SKU_100DOH_WORKERS 必须在 1 到 4 之间")
    thread_state = threading.local()

    def detect_frame(image):
        """在当前线程的 CUDA 流上检测一帧并返回商品框、手框和候选。"""
        # 每个工作线程使用自己的 CUDA 流；结果仍按视频顺序消费。
        if not hasattr(thread_state, "stream"):
            thread_state.stream = torch.cuda.Stream(device=device)
        with torch.inference_mode(), torch.cuda.stream(thread_state.stream):
            result = infer(model, classes, image, device, 0.5, 0.5)
            linked = _linked_candidates(result)
            selected = linked[0]["box"] if linked else None
            candidates = _all_object_candidates(result, linked)
            hands = [{"box": hand["box"]} for hand in result["hands"]]
            thread_state.stream.synchronize()
            return selected, hands, candidates

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="100doh-frame") as pool:
            while True:
                images = []
                for _ in range(workers * 2):
                    ok, image = cap.read()
                    if not ok:
                        break
                    images.append(image)
                if not images:
                    break
                decisions = [gate.allow(len(boxes) + i) for i in range(len(images))]
                selected = iter(
                    pool.map(
                        detect_frame,
                        [image for image, allowed in zip(images, decisions) if allowed],
                    )
                )
                for allowed in decisions:
                    box, hands, candidates = next(selected) if allowed else (None, [], [])
                    boxes.append(box)
                    hands_by_frame.append(hands)
                    candidates_by_frame.append(candidates)
                    if allowed:
                        gate.observe(box is not None)
    finally:
        cap.release()
    if not boxes:
        raise ValueError("视频没有可解码的帧")
    output.write_text(
        json.dumps(
            {
                "frames": len(boxes),
                "boxes": boxes,
                "hands": hands_by_frame,
                "candidates": candidates_by_frame,
                "person_gate": gate.summary(),
            }
        ),
        encoding="utf-8",
    )
    LOG.info("100DOH 检测完成：帧数=%d", len(boxes))


def _rank_video(video, boxes, model, cabinet_id, primary=None, state=None):
    """对所选商品轨迹的帧提取特征并生成商品排名。

    参数：
        video: 输入视频路径；boxes: 逐帧商品框。
        model: 商品向量模型；primary: 可选的预先选择的轨迹。
    返回：
        排名列表和实际参与评分的帧数；轨迹不足五点时不进行匹配。
    """
    if primary is None:
        tracker = ProductTracker()
        for i, box in enumerate(boxes, 1):
            tracker.step(i, box, [])
        primary = select_product_track(tracker.summary())
    if not primary:
        return [], 0
    # 仅对选中轨迹中实际检测到的帧进行评分。
    points = {pt["frame"]: pt["box"] for pt in primary["points"]}
    if len(points) < config.SCORE_WINDOW_FRAMES:
        return [], len(points)
    saved_features = load_features(cabinet_id)
    matching_catalog = prepare_catalog(saved_features)
    if not matching_catalog[0]:
        return [], 0
    ranking, count, _ = rank_masked_track(
        video, primary, model, cabinet_id,
        state if state is not None else SimpleNamespace(),
        saved_features=saved_features, matching_catalog=matching_catalog,
    )
    return ranking, count


@router.post(
    "/recognize-100doh",
    response_model=Response100DOH,
    tags=["商品识别"],
    summary="100DOH GPU 视频商品识别",
)
def recognize_100doh(
    request: Request, cabinet_id: str = Form(..., description="柜子编码"), video: UploadFile = File(..., description="待识别的视频文件")
):
    """通过独立检测进程、区域门控和轨迹评分识别上传视频中的商品。

    参数：
        request: 提供应用共享模型及状态的请求对象。
        video: 待识别的视频文件。
    返回：
        包含商品、分数、轨迹数量和耗时的 Response100DOH。
    """
    start = time.monotonic()
    original = (video.filename or "").replace("\\", "/").split("/")[-1]
    if (
        not original
        or original in (".", "..")
        or any(ord(c) < 32 for c in original)
        or not Path(original).suffix
        or not Path(original).stem
    ):
        video.file.close()
        raise HTTPException(status_code=400, detail="请提供有效的视频文件名（包含扩展名）")
    require_known_cabinet(cabinet_id, video)
    JOBS.mkdir(parents=True, exist_ok=True)
    with RESOURCE_LOCK, tempfile.TemporaryDirectory(prefix="job_100doh_", dir=JOBS) as tmp:
        try:
            job = Path(tmp)
            video_path = _save_upload(video, original, job, cabinet_id)
            probe = cv2.VideoCapture(str(video_path))
            try:
                if not probe.isOpened() or not probe.read()[0]:
                    raise ValueError("无法读取视频或视频为空")
                width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                probe.release()
            detection_file = job / "detections.json"
            person_flags = scan_person_gate(request.app.state, video_path)
            command = [
                sys.executable,
                "-m",
                "app.api.recognize_100doh",
                "--detect",
                str(video_path),
                str(detection_file),
            ]
            if person_flags is not None:
                person_file = job / "person_flags.json"
                person_file.write_text(json.dumps(person_flags), encoding="utf-8")
                command.extend(["--person-flags", str(person_file)])
            worker = subprocess.run(
                command, capture_output=True, text=True, timeout=900, cwd=str(ROOT)
            )
            if worker.returncode:
                LOG.error("100DOH 检测进程失败：%s", worker.stderr[-3000:])
                raise RuntimeError("100DOH 检测失败")
            detection = json.loads(detection_file.read_text(encoding="utf-8"))
            LOG.info("100DOH 人物检测统计：%s", detection.get("person_gate"))
            areas = area_detection.get_area_boxes(video_path, request.app.state)
            candidates, hand_area = filter_hand_linked_candidates(
                detection["hands"], detection["candidates"], areas
            )
            boxes = [frame[0]["box"] if frame else None for frame in candidates]
            product_track_ids = [
                frame[0]["area_track_id"] if frame else None for frame in candidates
            ]
            LOG.info("100DOH 手部区域筛选统计：%s", hand_area)
            primary, motion_summary = select_motion_rank_points(
                video_path, boxes, width, height, product_track_ids=product_track_ids
            )
            LOG.info("100DOH 运动验证统计：%s", motion_summary)
            ranking, count = (
                _rank_video(
                    video_path, [], request.app.state.dino, cabinet_id,
                    primary=primary, state=request.app.state,
                )
                if primary
                else ([], 0)
            )
            best = accept_ranking(ranking, count)
            return Response100DOH(
                cabinet_id=cabinet_id,
                status="ok" if best else "not_found",
                product=f"product_{best['sku']}" if best else None,
                confidence=round(best["combined"], 6) if best else None,
                vector_score=round(best["vector"], 6) if best else None,
                color_score=round(best["color"], 6) if best else None,
                track_points=count,
                frames=detection["frames"],
                video_path=str(video_path.relative_to(ROOT)),
                duration=round(time.monotonic() - start, 3),
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except subprocess.TimeoutExpired:
            LOG.exception("100DOH 检测超时")
            raise HTTPException(status_code=504, detail="100DOH 检测超时")
        except Exception:
            LOG.exception("100DOH 识别失败")
            raise HTTPException(status_code=500, detail="100DOH 识别失败，请查看容器日志")
        finally:
            video.file.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    from app.logging import setup_logging

    setup_logging("100doh")
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect", nargs=2, metavar=("VIDEO", "OUTPUT"), required=True)
    parser.add_argument("--person-flags")
    args = parser.parse_args()
    _detect_worker(Path(args.detect[0]), Path(args.detect[1]), args.person_flags)
