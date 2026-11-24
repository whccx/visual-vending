"""逐帧检测柜内区域，并可选生成带标注的视频。"""

import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app import config

MODEL = config.area_model()


def load_model():
    """从 SKU_AREA_MODEL 指定的路径加载 YOLO 模型，文件不存在时抛出异常。"""
    from ultralytics import YOLO

    if not MODEL.is_file():
        raise FileNotFoundError(f"柜内区域模型不存在: {MODEL}")
    return YOLO(str(MODEL))


def detect_frame_box(frame, model, confidence=None):
    """返回分割掩码的外接框和置信度，兼容原有调用方。"""
    _, box, score = detect_frame_mask_with_confidence(frame, model, confidence)
    return box, score


def detect_frame_mask(frame, model, confidence=None):
    """检测单帧区域掩码，返回（布尔掩码, 掩码外接框）；未检出时均为 None。

    区域模型为 seg 模型时取置信度最高的实例掩码并还原到原帧分辨率；
    外接框仅供日志与对角线度量使用，判定应以掩码本身为准。
    """
    mask, box, _ = detect_frame_mask_with_confidence(frame, model, confidence)
    return mask, box


def detect_frame_mask_with_confidence(frame, model, confidence=None):
    """统一运行货柜分割并返回掩码、外接框和置信度。"""
    height, width = frame.shape[:2]
    prediction = model.predict(
        frame, imgsz=640, conf=config.area_confidence() if confidence is None else confidence,
        device=0, verbose=False,
    )[0]
    if not len(prediction.boxes) or prediction.masks is None:
        return None, None, None
    confs = prediction.boxes.conf.cpu().tolist()
    best = max(range(len(confs)), key=lambda index: confs[index])
    mask = prediction.masks.data[best].cpu().numpy()
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask > 0.5
    if not mask.any():
        return None, None, None
    ys, xs = np.where(mask)
    return mask, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], confs[best]


def detect_video_masks(path, model):
    """逐帧分割货柜，紧凑保存掩码以便多个识别阶段复用。"""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("无法打开视频")
    areas, packed_masks = [], []
    shape = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if shape is None:
                shape = frame.shape[:2]
            elif frame.shape[:2] != shape:
                raise ValueError("视频分辨率中途变化")
            mask, box = detect_frame_mask(frame, model)
            areas.append(box)
            packed_masks.append(np.packbits(mask.reshape(-1)) if mask is not None else None)
    finally:
        capture.release()
    if not areas:
        raise ValueError("视频没有可解码的帧")
    return areas, packed_masks, shape


def get_area_boxes(path, app_state):
    """使用应用共享模型检测视频，并把四点多边形转换为逐帧矩形框。

    参数：
        path: 视频路径。
        app_state: 保存常驻区域模型的应用状态；调用方负责持有资源锁。
    返回：
        每帧的 [左、上、右、下] 坐标列表，未检测到时对应 None。
    """
    if not hasattr(app_state, "area_model"):
        app_state.area_model = load_model()
    result = detect_video(path, app_state.area_model, config.area_confidence())
    boxes = []
    for row in result["frames"]:
        polygon = row["polygon"]
        if not polygon:
            boxes.append(None)
            continue
        x, y, width, height = cv2.boundingRect(np.asarray(polygon, dtype=np.int32))
        boxes.append([x, y, x + width, y + height])
    return boxes


def detect_video(path, model, confidence, intermediate=None, output=None):
    """逐帧调用区域模型，并在指定中间路径时生成标注视频。

    参数：
        path: 输入视频路径。
        model: 提供 predict 方法的区域检测模型。
        confidence: 模型检测置信度阈值。
        intermediate: 可选的临时标注视频路径。
        output: 指定中间路径时用于编码最终视频的输出路径。
    返回：
        包含视频宽高、帧率以及逐帧多边形和置信度的字典。
    异常：
        视频无法读取、参数无效或标注视频编码失败时抛出异常。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("无法打开视频")
    writer = None
    frames = []
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("无法读取视频尺寸或帧率")
        if intermediate is not None:
            writer = cv2.VideoWriter(
                str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("无法创建标注视频")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError("暂不支持视频分辨率中途变化")
            mask, box, score = detect_frame_mask_with_confidence(frame, model, confidence)
            number = len(frames) + 1
            row = {
                "frame": number,
                "time_seconds": round((number - 1) / fps, 3),
                "polygon": None,
                "confidence": None,
            }
            if box is not None:
                x1, y1, x2, y2 = box
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                row["polygon"] = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()
                row["confidence"] = round(score, 6)
                if writer is not None:
                    overlay = frame.copy()
                    overlay[mask] = (60, 220, 80)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                    cv2.putText(
                        frame,
                        f"area {score:.2f}",
                        (x1, max(35, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            frames.append(row)
            if writer is not None:
                if row["polygon"] is None:
                    cv2.putText(
                        frame,
                        "NO AREA DETECTED",
                        (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(frame)
        if not frames:
            raise ValueError("视频没有可解码的帧")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if intermediate is not None:
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        result = subprocess.run(
            [
                "/usr/bin/ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-i",
                str(intermediate),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if result.returncode:
            raise RuntimeError("标注视频编码失败: " + result.stderr[-1000:])
    return {"width": width, "height": height, "fps": round(fps, 6), "frames": frames}
