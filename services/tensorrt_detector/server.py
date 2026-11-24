"""独立的 TensorRT 视频检测 HTTP 服务。"""

import base64
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Queue

import cv2
from bg0 import Bg0Segmenter
from detector import Detector, hoi_input
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOG = logging.getLogger("tensorrt_detector")
WORKSPACE = Path("/workspace").resolve()
MODEL_DIR = Path("/models")
detector_pool = Queue()
bg0_pool = Queue()
detector = None
bg0 = None


@asynccontextmanager
async def lifespan(app):
    """进程启动时加载并预热引擎。"""
    global detector, bg0
    detectors = []
    segmenters = []
    for _ in range(2):
        model = Detector(MODEL_DIR)
        model.warmup()
        detectors.append(model)
        detector_pool.put(model)
    for _ in range(2):
        model = Bg0Segmenter(MODEL_DIR, detectors[0].logger)
        model.warmup()
        segmenters.append(model)
        bg0_pool.put(model)
    detector, bg0 = detectors[0], segmenters[0]
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        while not bg0_pool.empty():
            bg0_pool.get_nowait()
        while not detector_pool.empty():
            detector_pool.get_nowait()
        for model in segmenters:
            model.close()
        for model in detectors:
            model.close()
        bg0 = None
        detector = None


app = FastAPI(title="独立 TensorRT 商品检测服务", lifespan=lifespan)


class DetectRequest(BaseModel):
    video_path: str
    area_missing_max_frames: int = Field(default=375, ge=0, le=10000, strict=True)
    areas: list[list[int] | None]


class SegmentPoint(BaseModel):
    frame: int = Field(ge=1, strict=True)
    box: list[int] = Field(min_length=4, max_length=4)


class SegmentRequest(BaseModel):
    video_path: str
    points: list[SegmentPoint] = Field(min_length=1, max_length=1000)
    threshold: float = Field(ge=0.0, le=1.0)


class FrameRequest(BaseModel):
    image_path: str


class FrameSegmentRequest(FrameRequest):
    box: list[int] = Field(min_length=4, max_length=4)
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
        model = detector_pool.get()
        try:
            return model.detect(path, request.area_missing_max_frames, request.areas)
        finally:
            detector_pool.put(model)
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
        model = bg0_pool.get()
        try:
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
                        raise ValueError(f"无法读取第 {point.frame} 帧")
                    height, width = image.shape[:2]
                    x1, y1, x2, y2 = point.box
                    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                        raise ValueError(f"第 {point.frame} 帧商品框无效")
                    read_seconds += time.monotonic() - step
                    step = time.monotonic()
                    mask = model.segment(image, point.box, request.threshold)
                    infer_seconds += time.monotonic() - step
                    step = time.monotonic()
                    ok, encoded = cv2.imencode(".png", mask.astype("uint8") * 255)
                    if not ok:
                        raise RuntimeError("掩码编码失败")
                    masks.append(
                        {
                            "frame": point.frame,
                            "box": point.box,
                            "mask_png": base64.b64encode(encoded.tobytes()).decode("ascii"),
                        }
                    )
                    encode_seconds += time.monotonic() - step
            finally:
                cap.release()
            timings = {
                "total_seconds": time.monotonic() - started,
                "read_seconds": read_seconds,
                "segment_seconds": infer_seconds,
                "encode_seconds": encode_seconds,
            }
            LOG.info("BG0 批量抠图：帧数=%s 耗时=%s", len(masks), timings)
            return {"masks": masks, "timings": timings}
        finally:
            bg0_pool.put(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOG.exception("BG0 批量抠图失败")
        raise HTTPException(status_code=500, detail="BG0 批量抠图失败，请查看服务日志") from exc


@app.post("/frame/detect")
def detect_frame(request: FrameRequest):
    """从共享目录读取一帧，以独立 TensorRT 上下文执行检测。"""
    path = video_path(request.image_path)
    image = cv2.imread(str(path))
    if image is None or image.shape != (1080, 1920, 3):
        raise HTTPException(status_code=400, detail="视频帧无法读取或尺寸不符合要求")
    model = detector_pool.get()
    try:
        hands, products = model.decode(model.hoi.infer(hoi_input(image)))
        return {"hands": hands, "products": products}
    except Exception as exc:
        LOG.exception("TensorRT 单帧检测失败")
        raise HTTPException(status_code=500, detail="单帧检测失败") from exc
    finally:
        detector_pool.put(model)


@app.post("/frame/segment")
def segment_frame(request: FrameSegmentRequest):
    """对一帧的商品框执行 BG0 并返回框内掩码。"""
    path = video_path(request.image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=400, detail="视频帧无法读取")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = request.box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise HTTPException(status_code=400, detail="商品框无效")
    model = bg0_pool.get()
    try:
        mask = model.segment(image, request.box, request.threshold)
        ok, encoded = cv2.imencode(".png", mask.astype("uint8") * 255)
        if not ok:
            raise RuntimeError("掩码编码失败")
        return {"mask_png": base64.b64encode(encoded.tobytes()).decode("ascii")}
    except Exception as exc:
        LOG.exception("BG0 单帧抠图失败")
        raise HTTPException(status_code=500, detail="单帧抠图失败") from exc
    finally:
        bg0_pool.put(model)
