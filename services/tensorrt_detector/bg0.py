"""使用纯 TensorRT 执行 BG0 的整帧与局部精修抠图。"""

import math

import cv2
import numpy as np

from detector import Engine


def refinement_box(mask, width, height):
    """与 SKU 原 BG0 实现保持一致的局部精修区域。"""
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
    area, total = (right - left) * (bottom - top), width * height
    if right - left < 32 or bottom - top < 32 or area >= total * 0.85:
        return None
    foreground = len(xs) * width / mw * height / mh
    if area >= total * 0.8 and foreground / area < 0.4:
        return None
    return left, top, right, bottom


class Bg0Segmenter:
    """保持 BG0 引擎常驻，仅返回指定商品框的二值掩码。"""

    def __init__(self, model_dir, logger):
        self.engine = Engine(model_dir / "bg0_strong.engine", logger)

    def infer(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        normalized = (rgb.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        logits = self.engine.infer(tensor)["output_image"].squeeze()
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError("BG0 返回无效掩码")
        return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))

    def segment(self, image, box, threshold):
        height, width = image.shape[:2]
        base = self.infer(image)
        alpha = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)
        scale = min(1.0, 1024 / max(width, height))
        sw, sh = round(width * scale), round(height * scale)
        refined_box = refinement_box(base, sw, sh)
        if refined_box:
            source = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
            left, top, right, bottom = refined_box
            refined = self.infer(source[top:bottom, left:right])
            if np.isfinite(refined).all() and refined.max() >= 0.05:
                x1, y1 = math.floor(left * width / sw), math.floor(top * height / sh)
                x2, y2 = math.ceil(right * width / sw), math.ceil(bottom * height / sh)
                alpha[y1:y2, x1:x2] = cv2.resize(
                    refined, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR
                )
        x1, y1, x2, y2 = box
        return np.ascontiguousarray(alpha[y1:y2, x1:x2] >= threshold)

    def warmup(self):
        self.infer(np.zeros((512, 512, 3), dtype=np.uint8))

    def close(self):
        self.engine.close()
