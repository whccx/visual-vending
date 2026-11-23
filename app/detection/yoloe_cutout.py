"""使用独立 YOLOE 模型按文本提示生成透明背景图片。"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app import config

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
AUTO_PROMPTS = (
    "person", "mascot", "paper", "hand", "animal", "dog", "cat", "bird",
    "tree", "flower", "bottle", "food", "toy", "object",
)
AUTO_ALIASES = {"auto", "foreground", "foreground subject", "subject", "主体", "前景"}


class TargetNotFound(ValueError):
    """图片中没有符合提示词的分割目标。"""


def decode_image(data):
    """读取图片并保留原透明度和 EXIF 方向。"""
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("图片为空或超过 10 MB")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("图片不能超过 2000 万像素")
            return np.array(ImageOps.exif_transpose(source).convert("RGBA"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片，请上传有效图片") from exc


def load_model():
    """独立加载 YOLOE 和本地文本编码器，不改变视频模型。"""
    import torch
    from ultralytics import YOLOE
    from ultralytics.nn.text_model import MobileCLIPTS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    model = YOLOE(str(config.HELD_PRODUCT_MODEL))
    model.model.to("cuda").eval()
    model.model.clip_model = MobileCLIPTS(
        torch.device("cuda"), weight=str(config.HELD_PRODUCT_DIR / "mobileclip2_b.ts")
    )
    return model


def clean_mask(mask):
    """去除细小孤立片段并填充封闭空洞，保留多个主要主体。"""
    mask = mask.astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask.astype(bool)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest = int(sizes.argmax()) + 1
    max_area = int(sizes.max())
    keep = np.zeros(count, dtype=bool)
    keep[1:] = sizes >= max_area * 0.15
    # 靠近主体的小块可能是配件；远离主体的小块通常是背景残片。
    distance = cv2.distanceTransform((components != largest).astype(np.uint8), cv2.DIST_L2, 3)
    max_distance = min(mask.shape) * 0.02
    for component in range(1, count):
        if not keep[component] and sizes[component - 1] >= max_area * 0.005:
            if float(distance[components == component].min()) <= max_distance:
                keep[component] = True
    cleaned = keep[components].astype(np.uint8)
    flood = np.pad(cleaned, 1)
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return cleaned.astype(bool) | holes


def cutout(data, prompt, model):
    """返回原图尺寸的透明 PNG，填充主体内部空洞并去除碎片。"""
    prompt = prompt.strip()
    if not prompt or len(prompt) > 200:
        raise ValueError("提示词不能为空且不能超过 200 个字符")
    rgba = decode_image(data)
    # 将原有透明像素合成到黑色背景，以匹配项目中的参考图预处理。
    rgb = (rgba[:, :, :3].astype(np.float32) * (rgba[:, :, 3:4] / 255)).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    labels = list(AUTO_PROMPTS) if prompt.lower() in AUTO_ALIASES else [prompt]
    embeddings = model.model.get_text_pe(labels, cache_clip_model=True)
    model.set_classes(labels, embeddings=embeddings)
    result = model.predict(
        bgr, device=0, imgsz=640, conf=0.20, retina_masks=True, max_det=30, verbose=False
    )[0]
    if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
        raise TargetNotFound("未检测到与提示词匹配的目标")
    masks = result.masks.data.detach() > 0.5
    areas = masks.flatten(1).sum(dim=1)
    if not len(areas) or int(areas.max().item()) == 0:
        raise TargetNotFound("未检测到有效的目标分割区域")
    # 忽略远处的小目标，保留画面中的多个主要主体。
    selected = areas >= areas.max() * 0.15
    mask = masks[selected].any(dim=0).cpu().numpy()
    height, width = rgba.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    if not mask.any():
        raise TargetNotFound("未检测到有效的目标分割区域")
    if prompt.lower() in AUTO_ALIASES:
        # 与动物或卡通人偶相关的附属物容易被目标分割漏掉。
        selected_classes = result.boxes.cls[selected].detach().cpu().tolist()
        selected_labels = {labels[int(index)] for index in selected_classes}
        if selected_labels.intersection({"mascot", "animal", "dog", "cat", "bird", "toy"}):
            from app.detection.foreground_refine import refine

            mask |= refine(rgba)
    mask = clean_mask(mask)
    rgba[:, :, 3] = np.where(mask, 255, 0)
    output = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(output, format="PNG")
    return output.getvalue()
