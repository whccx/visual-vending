"""对解码后的 BGR 帧执行 HOI-DETR 推理并保留逐帧关联。"""

import numpy as np
import torch
from mmcv.ops import batched_nms
from mmcv.parallel import collate, scatter
from mmdet.core import bbox_cxcywh_to_xyxy


def run_inference_batch(
    model, test_pipeline, images, device, class_names, score_thr=0.3, nms_iou=0.5
):
    """按图像尺寸分组推理，并按原始输入顺序返回检测结果。

    参数：
        model: 提供特征提取和查询头的 HOI-DETR 模型。
        test_pipeline: 将输入图像转换为模型输入的测试流水线。
        images: 已解码的 BGR 图像列表。
        device: 模型所在设备。
        class_names: 按类别编号索引的类别名称。
        score_thr: 保留检测结果的最低分数。
        nms_iou: Soft-NMS 的重叠阈值。
    返回：
        每帧的检测列表和查询嵌入组成的列表；没有输入时返回空列表。
    """
    if not images:
        return []
    prepared = []
    for image in images:
        data = scatter(collate([test_pipeline(dict(img=image))], samples_per_gpu=1), [device])[0]
        tensor = data["img"]
        if isinstance(tensor, list):
            tensor = tensor[0]
        metas = data["img_metas"]
        if isinstance(metas, list) and isinstance(metas[0], list):
            metas = metas[0]
        for meta in metas:
            meta["batch_input_shape"] = tuple(tensor.shape[-2:])
        prepared.append((tensor, metas[0]))
    # 仅将张量尺寸相同的图像合批，并在返回时恢复输入顺序。
    groups = {}
    for idx, (tensor, meta) in enumerate(prepared):
        groups.setdefault(tuple(tensor.shape[1:]), []).append(idx)
    results = [None] * len(images)
    with torch.no_grad():
        for indices in groups.values():
            tensor = torch.cat([prepared[i][0] for i in indices], dim=0)
            metas = [prepared[i][1] for i in indices]
            feats = model.extract_feat(tensor)
            outs, hs = model.query_head(feats, metas, return_hs=True)
            for j, index in enumerate(indices):
                cls_logits, dec_boxes, embeddings = outs[0][-1][j], outs[1][-1][j], hs[-1][j]
                scores = cls_logits.sigmoid()
                flat_scores, flat_idx = scores.view(-1).topk(
                    model.query_head.test_cfg.get("max_per_img", 300)
                )
                flat_labels = flat_idx % model.query_head.num_classes
                flat_qidx = flat_idx // model.query_head.num_classes
                h, w = metas[j]["img_shape"][:2]
                boxes = bbox_cxcywh_to_xyxy(dec_boxes[flat_qidx]) * dec_boxes.new_tensor(
                    [w, h, w, h]
                )
                _, keep = batched_nms(
                    boxes,
                    flat_scores,
                    flat_labels,
                    dict(type="soft_nms", iou_threshold=nms_iou, min_score=score_thr),
                )
                scale = np.array(metas[j].get("scale_factor", 1.0), dtype=np.float32).ravel()
                if scale.size == 1:
                    scale = np.tile(scale, 4)
                elif scale.size == 2:
                    scale = np.array([scale[0], scale[1], scale[0], scale[1]])
                detections = []
                for k in keep.tolist():
                    score = flat_scores[k].item()
                    if score < score_thr:
                        continue
                    cls_id, query_idx = flat_labels[k].item(), flat_qidx[k].item()
                    box = boxes[k].cpu().numpy() / scale
                    detections.append(
                        dict(
                            box=box,
                            score=score,
                            class_id=cls_id,
                            class_name=class_names[cls_id],
                            query_idx=query_idx,
                            center=(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)),
                        )
                    )
                results[index] = (detections, embeddings)
    return results
