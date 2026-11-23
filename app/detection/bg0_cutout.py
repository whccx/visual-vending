"""BG0 Full 的原始帧 CUDA 抠图，独立于浏览器运行。"""

import math
import sys

import cv2
import numpy as np
import torch  # noqa: F401  # 加载 PyTorch 的 CUDA/cuDNN 库，供 ONNX Runtime 使用。

from app import config


def refinement_box(mask: np.ndarray, width: int, height: int):
    """按 BG0 quality 模式寻找带边距的主体二次推理区域。"""
    ys, xs = np.where(mask >= 0.1)
    if not len(xs):
        return None
    mh, mw = mask.shape
    left = math.floor(int(xs.min()) * width / mw)
    right = math.ceil((int(xs.max()) + 1) * width / mw)
    top = math.floor(int(ys.min()) * height / mh)
    bottom = math.ceil((int(ys.max()) + 1) * height / mh)
    padding = math.ceil(max(right - left, bottom - top) * 0.12)
    left, top = max(0, left - padding), max(0, top - padding)
    right, bottom = min(width, right + padding), min(height, bottom + padding)
    area = (right - left) * (bottom - top)
    total = width * height
    if right - left < 32 or bottom - top < 32 or area >= total * 0.85:
        return None
    foreground = len(xs) * width / mw * height / mh
    if area >= total * 0.8 and foreground / area < 0.4:
        return None
    return left, top, right, bottom


def crop_track_mask(mask: np.ndarray, box: tuple, threshold: float) -> np.ndarray:
    """从原图 alpha 提取选中轨迹对应区域，不扩大商品框。"""
    x1, y1, x2, y2 = box
    height, width = mask.shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("商品框超出原始帧")
    return np.ascontiguousarray(mask[y1:y2, x1:x2] >= threshold)


class Bg0Cutout:
    """懒加载 CUDA 会话，避免每帧重复加载模型。"""

    def __init__(self):
        if not config.BG0_MODEL.is_file():
            raise RuntimeError(f"BG0 模型不存在：{config.BG0_MODEL}")
        sys.path.insert(0, str(config.BG0_RUNTIME))
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("缺少 BG0 ONNX Runtime GPU 依赖") from exc
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("BG0 缺少 ONNX Runtime CUDA 执行器")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(config.BG0_MODEL), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("BG0 CUDA 初始化失败，拒绝静默使用 CPU")
        self.input = self.session.get_inputs()[0].name

    def _infer(self, image_bgr: np.ndarray) -> np.ndarray:
        size = config.BG0_INPUT_SIZE
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        data = rgb.astype(np.float32) / 255.0
        data = (data - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        tensor = np.ascontiguousarray(data.transpose(2, 0, 1)[None])
        output = np.asarray(self.session.run(None, {self.input: tensor})[-1]).squeeze()
        if output.ndim != 2 or not np.isfinite(output).all():
            raise ValueError("BG0 返回无效掩码")
        return 1.0 / (1.0 + np.exp(-np.clip(output, -80, 80)))

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """预测完整帧 alpha，并按 BG0 quality 策略进行局部精修。"""
        height, width = image_bgr.shape[:2]
        base = self._infer(image_bgr)
        alpha = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
        scale = min(1.0, 1024 / max(width, height))
        sw, sh = round(width * scale), round(height * scale)
        box = refinement_box(base, sw, sh)
        if box:
            source = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            left, top, right, bottom = box
            refined = self._infer(source[top:bottom, left:right])
            if np.isfinite(refined).all() and refined.max() >= 0.05:
                x1, y1 = math.floor(left * width / sw), math.floor(top * height / sh)
                x2, y2 = math.ceil(right * width / sw), math.ceil(bottom * height / sh)
                alpha[y1:y2, x1:x2] = cv2.resize(
                    refined, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR
                )
        return alpha
