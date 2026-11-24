"""对 HOI 最终轨迹进行前景、人手分割及商品匹配。"""

import io
import json
import logging
import shutil
import time
from collections.abc import Collection
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

from app import config
from app.catalog.product_catalog import load_features
from app.detection import bg0_cutout, foreground_refine, yoloe_cutout
from app.recognition.frame_orientation import (
    choose_text_rotation,
    load_orientation_ocr,
    rotate_frame_and_box,
)
from app.recognition.product_matching import (
    prepare_catalog,
    score_crop,
    select_scoring_window as _select_scoring_window,
)

LOG = logging.getLogger("sku." + __name__)


def _enhance_crop(rgb):
    """按配置先提高饱和度，再以中灰为中心调整对比度。"""
    saturated = ImageEnhance.Color(Image.fromarray(rgb)).enhance(config.HOI_MATCH_SATURATION)
    pixels = np.asarray(saturated, dtype=np.float32)
    adjusted = (pixels - 127.5) * config.HOI_MATCH_CONTRAST + 127.5
    return np.ascontiguousarray(np.clip(adjusted, 0, 255).astype(np.uint8))


def _segment_crop(crop_bgr, state):
    """分割前景并扣除人手；人手未检出时标记为不确定。"""
    rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
    foreground = foreground_refine.refine(rgba)
    if foreground.shape != crop_bgr.shape[:2] or not foreground.any():
        raise ValueError("前景掩码为空或尺寸不匹配")
    ok, encoded = cv2.imencode(".png", crop_bgr)
    if not ok:
        raise ValueError("裁剪图编码失败")
    if not hasattr(state, "cutout_model"):
        state.cutout_model = yoloe_cutout.load_model()
    try:
        # 直接调用共享模型，避免在持有接口锁时回调本服务造成死锁。
        hand_png = yoloe_cutout.cutout(encoded.tobytes(), "hand", state.cutout_model)
        hand = np.array(Image.open(io.BytesIO(hand_png)).convert("RGBA"))[:, :, 3] > 0
        if hand.shape != foreground.shape:
            raise ValueError("人手掩码尺寸不匹配")
        found = True
    except yoloe_cutout.TargetNotFound:
        hand = np.zeros_like(foreground)
        found = False
    mask = foreground & ~hand
    return mask, found, float(mask.sum() / foreground.sum())


def _segment_frame(image, box, state):
    """对原图进行 BG0 分割，仅保留当前轨迹商品框内的掩码。"""
    if not hasattr(state, "bg0_model"):
        state.bg0_model = bg0_cutout.Bg0Cutout()
    alpha = state.bg0_model.segment(image)
    if alpha.shape != image.shape[:2] or not np.isfinite(alpha).all():
        raise ValueError("BG0 原图掩码无效或尺寸不匹配")
    return bg0_cutout.crop_track_mask(alpha, box, config.BG0_MASK_THRESHOLD)


def _detect_frame_hands(image, state):
    """在整帧上用 MediaPipe 按 4 个方向检测手，返回映射回原图的关键点集合。

    裁剪图缺少上下文时 MediaPipe 难以检出握持的手；整帧检测可靠得多。
    """
    import mediapipe as mp

    from app.detection.held_product_detect import load_gpu_hands

    if not hasattr(state, "mp_hands"):
        state.mp_hands = load_gpu_hands()
    height, width = image.shape[:2]
    found = []
    for k, rotation in enumerate(
        (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ):
        frame = image if rotation is None else cv2.rotate(image, rotation)
        rh, rw = frame.shape[:2]
        result = state.mp_hands.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )
        for hand in result.hand_landmarks:
            pts = []
            for vertex in hand:
                x, y = vertex.x * rw, vertex.y * rh
                if k == 1:
                    x, y = y, height - x
                elif k == 2:
                    x, y = width - x, height - y
                elif k == 3:
                    x, y = width - y, x
                pts.append((x, y))
            found.append(np.array(pts))
    return found


def _hand_region_mask(pts, shape):
    """将手部关键点凸包填充为区域掩码，并外扩覆盖超出关键点的手指。"""
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    radius = int(min(25.0, max(8.0, 0.12 * float(np.hypot(x2 - x1, y2 - y1)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    hull = cv2.convexHull(pts.astype(np.int32))
    filled = cv2.fillConvexPoly(np.zeros(shape[:2], np.uint8), hull, 1)
    return cv2.dilate(filled, kernel) > 0


def _remove_hand_from_mask(image, box, foreground, state):
    """在商品框周围检测手，再从 BG0 的商品候选掩码中扣除。

    优先整帧 MediaPipe 关键点；未检出时回退 YOLOE 文本提示分割。
    """
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]
    padding = config.HOI_HAND_CROP_PADDING_RATIO
    px, py = int(np.ceil((x2 - x1) * padding)), int(np.ceil((y2 - y1) * padding))
    left, top = max(0, x1 - px), max(0, y1 - py)
    right, bottom = min(width, x2 + px), min(height, y2 + py)
    frame_hand = np.zeros(image.shape[:2], dtype=bool)
    try:
        detected = _detect_frame_hands(image, state)
        LOG.info("HOI 商品框 %s MediaPipe 检出手数=%s", tuple(box), len(detected))
        for pts in detected:
            center = pts.mean(axis=0)
            if not (left <= center[0] <= right and top <= center[1] <= bottom):
                continue
            frame_hand |= _hand_region_mask(pts, image.shape[:2])
        # 商品掩码为裁剪尺寸；全帧手部掩码需对齐到同一坐标系。
        hand = frame_hand[y1:y2, x1:x2]
    except Exception as exc:
        LOG.warning("HOI MediaPipe 手部检测失败，回退 YOLOE：%s", exc)
        hand = np.zeros_like(foreground, dtype=bool)
    if not hand.any():
        area = image[top:bottom, left:right]
        ok, encoded = cv2.imencode(".png", area)
        if not ok:
            raise ValueError("手部分割图编码失败")
        if not hasattr(state, "cutout_model"):
            state.cutout_model = yoloe_cutout.load_model()
        try:
            hand_png = yoloe_cutout.cutout(encoded.tobytes(), "hand", state.cutout_model)
            hand_area = np.asarray(Image.open(io.BytesIO(hand_png)).convert("RGBA"))[:, :, 3] > 0
            if hand_area.shape != area.shape[:2]:
                raise ValueError("人手掩码尺寸不匹配")
            hand = hand_area[y1 - top : y2 - top, x1 - left : x2 - left]
        except yoloe_cutout.TargetNotFound:
            hand = np.zeros_like(foreground, dtype=bool)
    remaining = foreground & ~hand
    if float(remaining.sum()) < 0.5 * float(foreground.sum()):
        # 重度握持时手与商品像素互锁，强行扣除会切掉商品主体；保留原掩码。
        return foreground, np.zeros_like(foreground, dtype=bool), False
    return remaining, hand, bool(hand.any())


def _propagate_mask(source, target, cap):
    """将参考帧掩码映射到目标帧，并检查前后向光流一致性。"""
    sx1, sy1, sx2, sy2 = source["box"]
    tx1, ty1, tx2, ty2 = target["box"]
    images = []
    for item in (source, target):
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"] - 1)
        ok, image = cap.read()
        if not ok:
            return None
        images.append(image)
    h, w = images[0].shape[:2]
    left, top = max(0, min(sx1, tx1) - 80), max(0, min(sy1, ty1) - 80)
    right, bottom = min(w, max(sx2, tx2) + 80), min(h, max(sy2, ty2) + 80)
    if right <= left or bottom <= top:
        return None
    source_gray = cv2.cvtColor(images[0][top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(images[1][top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    options = (None, 0.5, 5, 31, 5, 7, 1.5, 0)
    back = cv2.calcOpticalFlowFarneback(target_gray, source_gray, *options)
    forward = cv2.calcOpticalFlowFarneback(source_gray, target_gray, *options)
    yy, xx = np.indices(target_gray.shape, dtype=np.float32)
    map_x, map_y = xx + back[:, :, 0], yy + back[:, :, 1]
    source_mask = np.zeros(target_gray.shape, dtype=np.uint8)
    source_mask[sy1 - top : sy2 - top, sx1 - left : sx2 - left] = source["mask"] * 255
    warped = cv2.remap(source_mask, map_x, map_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT) > 0
    forward_x = cv2.remap(forward[:, :, 0], map_x, map_y, cv2.INTER_LINEAR)
    forward_y = cv2.remap(forward[:, :, 1], map_x, map_y, cv2.INTER_LINEAR)
    error = np.hypot(back[:, :, 0] + forward_x, back[:, :, 1] + forward_y)
    candidate = warped[ty1 - top : ty2 - top, tx1 - left : tx2 - left]
    error = error[ty1 - top : ty2 - top, tx1 - left : tx2 - left]
    if not candidate.any():
        return None
    consistent = error <= config.HOI_MASK_FLOW_MAX_ERROR
    if float(consistent[candidate].mean()) < config.HOI_MASK_FLOW_MIN_CONSISTENCY:
        return None
    # 一致性用于整帧验收，不逐像素腐蚀，否则会删去遮挡附近的商品细节。
    return candidate


def _repair_masks(items, cap):
    """对面积异常的帧尝试使用同轨迹可靠参考帧修复。"""
    valid = [item for item in items if item.get("mask") is not None and item["mask"].any()]
    if not valid:
        return
    fractions = [float(item["mask"].mean()) for item in valid]
    threshold = max(config.HOI_MASK_MIN_FRACTION,
                    float(np.median(fractions)) * config.HOI_MASK_RELATIVE_FRACTION)
    # 有人手分割的帧作为参考；避免用未检出人手的帧传播手部。
    anchors = [item for item in valid if item["hand_found"] and item["mask"].mean() >= threshold]
    for item in items:
        mask = item.get("mask")
        if mask is not None and float(mask.mean()) >= threshold:
            item["method"] = "直接分割"
            continue
        nearby = [anchor for anchor in anchors
                  if 0 < abs(anchor["frame"] - item["frame"]) <= config.HOI_MASK_MAX_FRAME_GAP]
        nearby.sort(key=lambda anchor: (-int(anchor["mask"].sum()),
                                       abs(anchor["frame"] - item["frame"])))
        repaired = None
        for anchor in nearby:
            repaired = _propagate_mask(anchor, item, cap)
            if repaired is not None and float(repaired.mean()) >= config.HOI_MASK_MIN_FRACTION:
                item.pop("error", None)
                item.update(mask=repaired, method="光流修复", reference_frame=anchor["frame"])
                break
        else:
            item.update(mask=None, method="跳过", error="掩码质量不足且光流修复失败")


def _reject_small_mask(item):
    """最终掩码不足阈值或不存在时，跳过该帧，不进行商品比对。

    扣除手部会刻意缩小掩码；质量门槛以扣除前的 BG0 占比为准。
    """
    mask = item.get("mask")
    raw_fraction = float(mask.mean()) if mask is not None and mask.size else 0.0
    fraction = item.get("bg0_fraction", raw_fraction)
    item["mask_fraction"] = raw_fraction
    if mask is not None and mask.any() and fraction >= config.HOI_MASK_MATCH_MIN_FRACTION:
        return
    reason = item.get("error") if mask is None or not mask.any() else None
    if not reason:
        reason = f"商品掩码占比 {fraction:.3f} 低于阈值 {config.HOI_MASK_MATCH_MIN_FRACTION:.3f}"
    item.update(mask=None, method="跳过", error=reason)
    LOG.info("HOI 第 %s 帧跳过比对：%s", item["frame"], reason)


def rank_masked_track(
    video, primary, model, cabinet_id, state, save_debug=False,
    saved_features=None, matching_catalog=None, use_trt_bg0=False,
    *, sku_scope: Collection[int] | None = None,
):
    """仅处理最终选中轨迹，分割或修复后计算商品匹配分数。"""
    points = {int(point["frame"]): point["box"] for point in (primary or {}).get("points", [])}
    debug_dir = None
    if save_debug:
        parent = config.ROOT / "tests" / "result" / "hoi_mask"
        parent.mkdir(parents=True, exist_ok=True)
        # 目录名用视频名+月日-时分秒（1120-203009 即 11月20日 20:30:09）；同秒重跑时追加序号。
        stem = Path(video).stem
        stamp = time.strftime("%m%d-%H%M%S")
        debug_dir = parent / f"{stem}_{stamp}"
        serial = 1
        while debug_dir.exists():
            debug_dir = parent / f"{stem}_{stamp}-{serial}"
            serial += 1
        debug_dir.mkdir(parents=True)
        for name in ("original_crops", "crops", "bg0_masks", "hand_masks", "masks", "masked", "enhanced", "selected_enhanced"):
            (debug_dir / name).mkdir()
    if not points:
        return [], 0, debug_dir
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("无法读取视频")
    items, failures = [], []
    try:
        remote_masks = None
        if use_trt_bg0 and not config.HOI_ORIENTATION_ENABLED:
            from app.detection.hoi_tensorrt_client import segment_video

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            batch = []
            for frame, box in sorted(points.items()):
                x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
                x2, y2 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
                if x2 > x1 and y2 > y1:
                    batch.append({"frame": frame, "box": [x1, y1, x2, y2]})
            remote_masks = segment_video(video, batch, config.BG0_MASK_THRESHOLD)
        elif use_trt_bg0:
            LOG.info("启用方向旋转，使用原 BG0 路径以保持旋转后的坐标一致")
        for frame, box in sorted(points.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"无法读取视频第 {frame} 帧")
            h, w = image.shape[:2]
            x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
            x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
            if x2 <= x1 or y2 <= y1:
                failures.append({"frame": frame, "reason": "商品框无效"})
                continue
            original_box = (x1, y1, x2, y2)
            name = f"{frame:06d}.png"
            if debug_dir and not cv2.imwrite(str(debug_dir / "original_crops" / name), image[y1:y2, x1:x2]):
                raise OSError(f"第 {frame} 帧原始裁剪图保存失败")
            angle, orientation = 0, {"reason": "disabled"}
            if config.HOI_ORIENTATION_ENABLED:
                try:
                    if not hasattr(state, "hoi_orientation_ocr"):
                        state.hoi_orientation_ocr = load_orientation_ocr()
                    angle, orientation = choose_text_rotation(image, original_box, state.hoi_orientation_ocr)
                except Exception as exc:
                    # 方向证据不可用时保留原图；不要因 OCR 失败阻断商品识别。
                    LOG.warning("HOI 第 %s 帧方向判断失败，保留原方向：%s", frame, exc)
                    angle, orientation = 0, {"reason": "unavailable"}
            image, transformed_box = rotate_frame_and_box(image, original_box, angle)
            x1, y1, x2, y2 = transformed_box
            crop = image[y1:y2, x1:x2].copy()
            item = {"frame": frame, "box": transformed_box, "original_box": original_box,
                    "rotation": angle, "orientation": orientation,
                    "crop": crop, "mask": None, "hand_found": False}
            if debug_dir and not cv2.imwrite(str(debug_dir / "crops" / f"{frame:06d}.png"), crop):
                raise OSError(f"第 {frame} 帧裁剪图保存失败")
            try:
                if remote_masks is None:
                    item["mask"] = _segment_frame(image, item["box"], state)
                    item["method"] = "BG0 原图 CUDA 分割"
                else:
                    item["mask"] = remote_masks[frame]
                    item["method"] = "BG0 原图 TensorRT 分割"
                if debug_dir:
                    cv2.imwrite(str(debug_dir / "bg0_masks" / f"{frame:06d}.png"),
                                item["mask"].astype(np.uint8) * 255)
                # BG0 阶段不足 40% 的帧直接放弃，不再调用人手分割。
                item["bg0_fraction"] = float(item["mask"].mean())
                if (config.HOI_REMOVE_HAND and
                        item["bg0_fraction"] >= config.HOI_MASK_MATCH_MIN_FRACTION):
                    item["mask"], hand, item["hand_found"] = _remove_hand_from_mask(
                        image, item["box"], item["mask"], state
                    )
                    item["method"] = "BG0 原图分割及手部扣除"
                    if debug_dir:
                        cv2.imwrite(str(debug_dir / "hand_masks" / f"{frame:06d}.png"),
                                    hand.astype(np.uint8) * 255)
            except ValueError as exc:
                item["error"] = str(exc)
                LOG.warning("HOI 第 %s 帧分割失败：%s", frame, exc)
            items.append(item)
    finally:
        cap.release()
    for item in items:
        _reject_small_mask(item)
    if saved_features is None:
        saved_features = load_features(cabinet_id)
    catalog = matching_catalog if matching_catalog is not None else prepare_catalog(saved_features)
    scores = []
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
    frame_report = []
    for item in items:
        frame, mask, crop = item["frame"], item.get("mask"), item["crop"]
        detail = {key: item[key] for key in ("frame", "method", "rotation", "orientation", "original_box", "box", "hand_found", "remaining_ratio",
                                             "reference_frame", "error",
                                             "mask_fraction") if key in item}
        if mask is None or not mask.any():
            failures.append({"frame": frame, "reason": detail.get("error", "掩码为空")})
            frame_report.append(detail)
            continue
        detail["mask_fraction"] = float(mask.mean())
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        enhanced = _enhance_crop(rgb)
        if debug_dir:
            name = f"{frame:06d}.png"
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = mask.astype(np.uint8) * 255
            Image.fromarray(rgba).save(debug_dir / "masked" / name)
            enhanced_rgba = np.dstack((enhanced, rgba[:, :, 3]))
            Image.fromarray(enhanced_rgba).save(debug_dir / "enhanced" / name)
            if not cv2.imwrite(str(debug_dir / "masks" / name), rgba[:, :, 3]):
                raise OSError(f"第 {frame} 帧掩码保存失败")
        if catalog[0]:
            crop_tensor = torch.from_numpy(enhanced).cuda().permute(2, 0, 1)
            color_tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1)
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).cuda()
            with torch.inference_mode():
                _, frame_scores = score_crop(
                    crop_tensor, mask_tensor, model, catalog, mean, std,
                    color_crop=color_tensor,
                )
            scores.append((frame, frame_scores))
        frame_report.append(detail)
    selected_frames, ranking = _select_scoring_window(scores, catalog[0], sku_scope=sku_scope)
    if debug_dir:
        for frame in selected_frames:
            name = f"{frame:06d}.png"
            shutil.copyfile(debug_dir / "enhanced" / name, debug_dir / "selected_enhanced" / name)
    if debug_dir:
        report = {"selected_frames": sorted(points), "processed_frames": len(scores),
                  "scored_frames": len(selected_frames), "matching_window_frames": selected_frames,
                  "failures": failures, "frames": frame_report, "ranking": ranking}
        (debug_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    LOG.info("HOI 掩码评分：轨迹帧=%s 处理帧=%s 比对窗口=%s 跳过=%s", len(points), len(scores), selected_frames, len(failures))
    return ranking, len(selected_frames), debug_dir
