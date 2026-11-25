"""ViT-7B ArcFace 的参考图建模、五帧向量评分和保守确认。"""

import math
import threading
from statistics import mean, pstdev

import cv2
import numpy as np

from app import config


class ArcFaceEncoder:
    """每个 worker 共享同一个 BF16 模型，串行执行向量前向。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None

    def encode(self, pixels):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel

        with self._lock, torch.inference_mode():
            if self._model is None:
                model = AutoModel.from_pretrained(
                    str(config.ARCFACE_MODEL), local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                ).eval()
                if model.config.hidden_size != config.ARCFACE_DIMS:
                    raise ValueError("ViT-7B ArcFace 模型向量维度不符")
                torch.cuda.empty_cache()
                self._model = model.cuda()
            output = self._model(pixel_values=pixels.to(device="cuda", dtype=torch.bfloat16))
            return F.normalize(output.pooler_output.float(), dim=1)

    def references(self, images: list[bytes]) -> np.ndarray:
        from app.catalog.product_catalog import reference_batches

        return np.concatenate([
            self.encode(pixels).cpu().numpy() for pixels, _ in reference_batches(images)
        ])

    def release(self) -> None:
        with self._lock:
            self._model = None


def geometric_windows(points):
    """按商品框总面积排序，只使用连续的完整窗口。"""
    boxes = {}
    for point in points:
        x1, y1, x2, y2 = point["box"]
        if x1 < 19.2 or y1 < 10.8 or x2 > 1900.8 or y2 > 1069.2:
            continue
        box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
        if box[2] > box[0] and box[3] > box[1]:
            boxes[point["frame"]] = box
    areas = {frame: (x2 - x1) * (y2 - y1) for frame, (x1, y1, x2, y2) in boxes.items()}
    size = config.SCORE_WINDOW_FRAMES
    windows = [tuple(range(frame, frame + size)) for frame in boxes]
    windows = [window for window in windows if all(frame in boxes for frame in window)]
    windows.sort(key=lambda window: (-sum(areas[frame] for frame in window), window[0]))
    return windows, boxes


def score_window(crops, encoder: ArcFaceEncoder, catalog):
    """用同一批去手裁剪计算每帧的 SKU 向量分与包装色彩分。"""
    import torch
    import torch.nn.functional as F

    from app.recognition.full_color import descriptor
    from app.recognition.hoi_mask_matching import _enhance_crop

    vectors, centroids, colors = catalog
    skus = tuple(vectors)
    mean_rgb = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std_rgb = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    pixels, color_rows = [], []
    with torch.inference_mode():
        for bgr, mask in crops:
            raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            enhanced = _enhance_crop(raw) * mask[:, :, None]
            tensor = torch.from_numpy(enhanced).permute(2, 0, 1)
            resized = F.interpolate(
                tensor[None].float(), (224, 224), mode="bilinear", align_corners=False
            ).round()
            pixels.append(resized)
            color_rows.append(descriptor(
                torch.from_numpy(raw).permute(2, 0, 1).contiguous().cuda(),
                torch.from_numpy(np.ascontiguousarray(mask)).cuda(),
            ))
        query = encoder.encode((torch.cat(pixels) / 255 - mean_rgb) / std_rgb)
        full_color = torch.stack(color_rows)
        vector_scores = torch.stack([
            torch.maximum(query @ centroids[sku], (query @ vectors[sku].T).amax(1))
            for sku in skus
        ], dim=1).cpu().numpy()
        color_scores = torch.stack([
            torch.sqrt(full_color[:, None] * colors[sku][None]).sum(2).amax(1)
            for sku in skus
        ], dim=1).cpu().numpy()
    return skus, vector_scores, color_scores


def confirm_window(skus, vector_scores, color_scores):
    """按统一证据阈值返回可确认的 SKU 和用于日志的第一名证据。"""
    if (
        len(skus) < 3 or len(vector_scores) != config.SCORE_WINDOW_FRAMES
        or not np.isfinite(vector_scores).all()
    ):
        return None, {}
    averages = vector_scores.mean(axis=0, dtype=np.float64).tolist()
    order = sorted(range(len(skus)), key=lambda index: (-averages[index], skus[index]))
    leader = order[0]
    score = averages[leader]
    gap = score - averages[order[1]]
    competitors = [averages[index] for index in order[1:]]
    sigma = pstdev(competitors)
    separation = (score - mean(competitors)) / sigma if sigma > 1e-9 else 0.0
    relative_gap = gap / score if score > 0 else 0.0
    info = {
        "sku": skus[leader], "向量分": score, "前两名差": gap,
        "标准化分离度": separation, "相对分差": relative_gap,
    }
    if score < config.ARCFACE_MIN_VECTOR:
        return None, info
    if (
        score >= config.ARCFACE_HIGH_VECTOR and gap >= config.ARCFACE_MIN_MARGIN
        and separation >= config.ARCFACE_HIGH_SEPARATION
    ) or relative_gap >= config.ARCFACE_RELATIVE_MARGIN:
        return skus[leader], info
    if gap < config.ARCFACE_CONFIRM_MARGIN or not np.isfinite(color_scores).all():
        return None, info
    contenders = np.delete(vector_scores, leader, axis=1)
    frame_gap = float((vector_scores[:, leader] - contenders.max(axis=1)).min())
    mean_colors = color_scores.mean(axis=0, dtype=np.float64)
    color = float(mean_colors[leader])
    color_rank = 1 + int((mean_colors > color).sum())
    info.update(逐帧最小差=frame_gap, 颜色分=color, 颜色名次=color_rank)
    if (
        frame_gap >= config.ARCFACE_MIN_MARGIN
        and color >= config.ARCFACE_CONFIRM_COLOR
        and color_rank <= config.ARCFACE_CONFIRM_COLOR_RANK
    ):
        return skus[leader], info
    return None, info
