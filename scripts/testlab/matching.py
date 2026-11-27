"""三条识别路径共用的商品特征匹配与可信判定。"""

import logging
import math
import os
from collections.abc import Collection
from decimal import Decimal
from typing import Literal

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
    color_crop=None,
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
    full_color = descriptor(crop if color_crop is None else color_crop, mask) if reference_hues is None else None
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


def select_scoring_window(scored_frames, skus, *, sku_scope: Collection[int] | None = None):
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
        best = ranking[0]["sku"]
        if sku_scope is not None and best not in sku_scope:
            # 范围外商品在窗口的每帧都必须有更强的向量证据。
            if not all(
                by_frame[frame][best][0] > by_frame[frame][sku][0]
                for frame in frames
                for sku in sku_scope
            ):
                continue
        margin = ranking[0]["combined"] - ranking[1]["combined"] if len(ranking) > 1 else 0.0
        candidates.append((margin, -start, frames, ranking))
    if not candidates:
        return [], []
    _, _, frames, ranking = max(candidates, key=lambda item: (item[0], item[1]))
    return frames, ranking


def ranking_rejection_reason(
    ranking, count, *, floor_first=None
) -> Literal["insufficient_evidence", "unmatched_scores", "ambiguous_margin"] | None:
    """按现有匹配规则返回拒识原因，满足所有门槛时返回 None。

    与线上规则的差异仅在判定顺序可调：floor_first 为真时先检查第一名
    地板分再检查前两名分差，让"库内没有像的商品"优先报 unmatched_scores。
    """
    if floor_first is None:
        floor_first = os.environ.get("TESTLAB_FLOOR_FIRST", "0") == "1"
    if not ranking or count < config.SCORE_WINDOW_FRAMES:
        return "insufficient_evidence"
    best = ranking[0]
    vector, color = float(best["vector"]), float(best["color"])
    first = float(best["combined"])
    second = float(ranking[1]["combined"]) if len(ranking) > 1 else None
    if not all(math.isfinite(score) for score in (vector, color, first)):
        return "insufficient_evidence"
    if second is not None and not math.isfinite(second):
        return "insufficient_evidence"
    if floor_first and (vector < config.min_vector_score() or color < config.min_color_score()):
        return "unmatched_scores"
    if (
        second is not None
        and Decimal(str(first)) - Decimal(str(second)) < Decimal(str(config.SCORE_MIN_MARGIN))
    ):
        if os.environ.get("TESTLAB_AMBIGUOUS_UNMATCHED", "0") == "1":
            # 前两名分差过小说明库内没有能可靠胜出的商品，按柜中无此SKU告警处理。
            return "unmatched_scores"
        return "ambiguous_margin"
    if vector < config.min_vector_score() or color < config.min_color_score():
        return "unmatched_scores"
    # 多个商品的色相分都接近满分时，颜色无法提供有效的区分证据。
    if color >= config.SCORE_AMBIGUOUS_HUE_THRESHOLD:
        similar_colors = sum(
            math.isfinite(float(item["color"]))
            and float(item["color"]) >= config.SCORE_AMBIGUOUS_HUE_THRESHOLD
            for item in ranking
        )
        if similar_colors >= config.SCORE_AMBIGUOUS_HUE_COUNT:
            LOG.info("商品匹配拒识：%s 个候选商品色相高度相似", similar_colors)
            return "insufficient_evidence"
    return None


def accept_ranking(ranking, count):
    """返回满足匹配门槛的第一名候选。"""
    return ranking[0] if ranking_rejection_reason(ranking, count) is None else None
