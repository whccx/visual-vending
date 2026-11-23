"""在最终裁剪前，以包装文字判断是否旋转原始帧。"""

import sys

import cv2
import numpy as np

from app import config

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def load_orientation_ocr():
    """加载本机 OCR；模型与运行环境由部署配置提供。"""
    for directory in (config.BG0_RUNTIME, config.HOI_ORIENTATION_OCR_DIR):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def choose_text_rotation(image, box, ocr):
    """用 ROI 临时预览判断方向，最终裁剪仍在原帧旋转后进行。"""
    x1, y1, x2, y2 = box
    preview = image[y1:y2, x1:x2]
    if not preview.size:
        raise ValueError("方向判断商品区域为空")
    scale = config.HOI_ORIENTATION_OCR_TARGET_SIZE / max(preview.shape[:2])
    if scale > 1:
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scores = {}
    for angle, code in ROTATIONS.items():
        rotated = preview if code is None else cv2.rotate(preview, code)
        lines, _ = ocr(rotated)
        evidence = []
        for points, text, confidence in lines or []:
            points = np.asarray(points)
            if points.shape != (4, 2):
                continue
            width, height = float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1]))
            count = sum(character.isalnum() for character in text)
            if (count >= config.HOI_ORIENTATION_OCR_MIN_CHARS
                    and confidence >= config.HOI_ORIENTATION_OCR_MIN_CONFIDENCE
                    and width >= config.HOI_ORIENTATION_TEXT_ASPECT * height):
                evidence.append(float(confidence))
        scores[angle] = max(evidence, default=0.0)
    ranked = sorted(scores, key=lambda angle: (-scores[angle], angle))
    if (scores[ranked[0]] < config.HOI_ORIENTATION_OCR_MIN_CONFIDENCE
            or scores[ranked[0]] - scores[ranked[1]] < config.HOI_ORIENTATION_OCR_MIN_GAP):
        return 0, {"reason": "uncertain", "scores": scores}
    return ranked[0], {"reason": "text", "scores": scores}


def rotate_frame_and_box(image, box, angle):
    """以 90° 倍数旋转整帧和左闭右开商品框，确保原始像素不被裁掉。"""
    if angle not in ROTATIONS:
        raise ValueError("旋转角度必须是 0、90、180 或 270 度")
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("商品框超出原始帧")
    if angle == 0:
        return image, box
    if angle == 90:
        result_box = (height - y2, x1, height - y1, x2)
    elif angle == 180:
        result_box = (width - x2, height - y2, width - x1, height - y1)
    else:
        result_box = (y1, width - x2, y2, width - x1)
    return cv2.rotate(image, ROTATIONS[angle]), result_box
