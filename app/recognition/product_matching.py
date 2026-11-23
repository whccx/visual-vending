"""三条识别路径共用的商品特征匹配与可信判定。"""

import logging
import math
from decimal import Decimal

import numpy as np
import torch
import torch.nn.functional as F

from app import config
from app.catalog.product_catalog import hist
from app.recognition.full_color import descriptor

LOG = logging.getLogger("sku." + __name__)


def prepare_catalog(catalog):
    """将商品库特征、逐商品中心向量和颜色特征转移到 GPU。

    参数：
        catalog: 以商品标识为键，包含 features 和 full_colors 的字典。
    返回：
        依次包含参考特征、归一化中心向量和颜色特征字典的元组。
    """
    features, centroids, colors = {}, {}, {}
    for sku, entry in catalog.items():
        center = None
        for index, ref in enumerate(entry["features"]):
            center = ref.copy() if center is None else (center * index + ref) / (index + 1)
            center /= np.linalg.norm(center)
        features[sku] = torch.from_numpy(entry["features"]).cuda()
        centroids[sku] = torch.from_numpy(center).cuda()
        colors[sku] = torch.from_numpy(entry["full_colors"]).cuda()
    return features, centroids, colors


def score_crop(
    crop, mask, model, catalog, mean, std, *, reference_hues=None, hue_rgb=None, hue_mask=None,
):
    """对掩码裁剪图提取向量及颜色特征，并计算各商品的相似度。

    参数：
        crop, mask: RGB 裁剪张量及其可见区域掩码。
        model: 接收 pixel_values 并返回 pooler_output 的特征模型。
        catalog: prepare_catalog 返回的商品特征元组。
        mean, std: 模型输入归一化参数。
    返回：
        查询向量及每个商品的（向量分数，颜色分数）字典。
    """
    features, centroids, colors = catalog
    masked = crop * mask.unsqueeze(0)
    resized = (
        F.interpolate(masked[None].float(), size=(224, 224), mode="bilinear", align_corners=False)
        .round()
        .div_(255)
    )
    query = F.normalize(model(pixel_values=(resized - mean) / std).pooler_output, dim=1)[0]
    full_color = descriptor(crop, mask) if reference_hues is None else None
    scores = {}
    for sku in features:
        vector = torch.maximum(query @ centroids[sku], (features[sku] @ query).max())
        color = torch.sqrt(colors[sku] * full_color).sum(dim=1).max() if full_color is not None else 0.0
        scores[sku] = (float(vector), float(color))
    if reference_hues is not None:
        if hue_rgb is None:
            hue_rgb = crop.permute(1, 2, 0).contiguous().cpu().numpy()
        if hue_mask is None:
            hue_mask = mask.contiguous().cpu().numpy()
        scores = hue_frame_scores(hue_rgb, hue_mask, scores, reference_hues)
    return query, scores


def rank_scores(scores, skus):
    """对多帧分数求平均并按配置的向量和颜色权重计算降序排名。

    参数：
        scores: 每帧以商品标识索引向量分数和颜色分数的字典列表。
        skus: 待排名的商品标识。
    返回：
        包含商品标识、两项平均分和综合分的列表；同分按标识排序。
    """
    result = []
    vector_weight, color_weight = config.vector_weight(), config.color_weight()
    for sku in skus:
        if scores:
            vector = float(np.mean([score[sku][0] for score in scores]))
            color = float(np.mean([score[sku][1] for score in scores]))
            result.append(
                {
                    "sku": sku,
                    "vector": vector,
                    "color": color,
                    "combined": vector_weight * vector + color_weight * color,
                }
            )
    result.sort(key=lambda item: (-item["combined"], item["sku"]))
    return result


def hue_frame_scores(rgb, mask, scores, reference_hues):
    """保留向量分，用商品库现有 36 维色相特征重新计算颜色分。"""
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    alpha = np.ascontiguousarray(mask.astype(np.uint8) * 255)
    query = hist(bgr, alpha)
    return {
        sku: (
            float(vector),
            float(np.sqrt(np.maximum(reference_hues[sku] * query, 0)).sum(axis=1).max()),
        )
        for sku, (vector, _) in scores.items()
    }


def select_scoring_window(scored_frames, skus):
    """从连续有效帧窗口中选综合分前两名间隔最大的窗口。"""
    by_frame = dict(scored_frames)
    size = config.SCORE_WINDOW_FRAMES
    candidates = []
    for start in sorted(by_frame):
        frames = list(range(start, start + size))
        if any(frame not in by_frame for frame in frames):
            continue
        ranking = rank_scores([by_frame[frame] for frame in frames], skus)
        if not ranking:
            continue
        margin = ranking[0]["combined"] - ranking[1]["combined"] if len(ranking) > 1 else 0.0
        candidates.append((margin, -start, frames, ranking))
    if not candidates:
        return [], []
    _, _, frames, ranking = max(candidates, key=lambda item: (item[0], item[1]))
    return frames, ranking


def accept_ranking(ranking, count):
    """根据轨迹点数、向量分数、颜色分数及前两名差距筛选最佳商品。

    参数：
        ranking: rank_scores 生成的降序候选列表。
        count: 参与识别的轨迹点数。
    返回：
        满足门槛的第一名候选；否则返回 None。
    """
    if not ranking or count < config.SCORE_WINDOW_FRAMES:
        return None
    best = ranking[0]
    vector, color = float(best["vector"]), float(best["color"])
    if not math.isfinite(vector) or vector < config.min_vector_score():
        return None
    if not math.isfinite(color) or color < config.min_color_score():
        return None
    # 多个商品的色相分都接近满分时，颜色无法提供有效的区分证据。
    if color >= config.SCORE_AMBIGUOUS_HUE_THRESHOLD:
        similar_colors = sum(
            math.isfinite(float(item["color"]))
            and float(item["color"]) >= config.SCORE_AMBIGUOUS_HUE_THRESHOLD
            for item in ranking
        )
        if similar_colors >= config.SCORE_AMBIGUOUS_HUE_COUNT:
            LOG.info("商品匹配拒识：%s 个候选商品色相高度相似", similar_colors)
            return None
    if len(ranking) > 1:
        first, second = float(best["combined"]), float(ranking[1]["combined"])
        if (
            not math.isfinite(first)
            or not math.isfinite(second)
            or Decimal(str(first)) - Decimal(str(second)) < Decimal("0.03")
        ):
            return None
    return best
