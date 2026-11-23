"""独立的 TensorRT 视频检测 HTTP 服务。"""

import logging
import base64
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import cv2

from detector import Detector
from bg0 import Bg0Segmenter

LOG = logging.getLogger("tensorrt_detector")
WORKSPACE = Path("/workspace").resolve()
MODEL_DIR = Path("/models")
LOCK = threading.Lock()
detector = None
bg0 = None


@asynccontextmanager
async def lifespan(app):
    """进程启动时加载并预热引擎。"""
    global detector, bg0
    detector = Detector(MODEL_DIR)
    detector.warmup()
    bg0 = Bg0Segmenter(MODEL_DIR, detector.logger)
    bg0.warmup()
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        bg0.close()
        detector.close()
        bg0 = None
        detector = None


app = FastAPI(title="独立 TensorRT 商品检测服务", lifespan=lifespan)


class DetectRequest(BaseModel):
    video_path: str


class SegmentPoint(BaseModel):
    frame: int = Field(ge=1, strict=True)
    box: list[int] = Field(min_length=4, max_length=4)


class SegmentRequest(BaseModel):
    video_path: str
    points: list[SegmentPoint] = Field(min_length=1, max_length=1000)
    threshold: float = Field(ge=0.0, le=1.0)


def video_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise HTTPException(status_code=400, detail="视频路径必须为 /workspace 下的绝对路径")
    path = path.resolve()
    if not path.is_relative_to(WORKSPACE) or not path.is_file():
        raise HTTPException(status_code=400, detail="视频必须是共享目录内的现有文件")
    return path


@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(getattr(app.state, "ready", False))}


@app.post("/detect")
def detect(request: DetectRequest):
    """读取共享目录内的视频，返回每帧货柜、手和商品检测结果。"""
    path = video_path(request.video_path)
    try:
        with LOCK:
            return detector.detect(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("TensorRT 视频检测失败")
        raise HTTPException(status_code=500, detail="检测失败，请查看服务日志") from exc


@app.post("/segment")
def segment(request: SegmentRequest):
    """批量处理同一轨迹的原始视频帧，仅返回商品框内的无损掩码。"""
    path = video_path(request.video_path)
    if len({point.frame for point in request.points}) != len(request.points):
        raise HTTPException(status_code=400, detail="帧号不能重复")
    try:
        with LOCK:
            started = time.monotonic()
            read_seconds = infer_seconds = encode_seconds = 0.0
            masks = []
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError("无法打开视频")
            try:
                for point in sorted(request.points, key=lambda item: item.frame):
                    step = time.monotonic()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, point.frame - 1)
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError("无法读取第 {} 帧".format(point.frame))
                    height, width = image.shape[:2]
                    x1, y1, x2, y2 = point.box
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError("第 {} 帧商品框无效".format(point.frame))
                    read_seconds += time.monotonic() - step
                    step = time.monotonic()
                    mask = bg0.segment(image, point.box, request.threshold)
                    infer_seconds += time.monotonic() - step
                    step = time.monotonic()
                    ok, encoded = cv2.imencode(".png", mask.astype("uint8") * 255)
                    if not ok:
                        raise RuntimeError("掩码编码失败")
                    masks.append({"frame": point.frame, "box": point.box,
                                  "mask_png": base64.b64encode(encoded.tobytes()).decode("ascii")})
                    encode_seconds += time.monotonic() - step
            finally:
                cap.release()
            timings = {"total_seconds": time.monotonic() - started, "read_seconds": read_seconds,
                       "segment_seconds": infer_seconds, "encode_seconds": encode_seconds}
            LOG.info("BG0 批量抠图：帧数=%s 耗时=%s", len(masks), timings)
            return {"masks": masks, "timings": timings}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("BG0 批量抠图失败")
        raise HTTPException(status_code=500, detail="BG0 批量抠图失败，请查看服务日志") from exc
