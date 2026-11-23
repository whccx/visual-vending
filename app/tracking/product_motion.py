"""计算候选物体相对场景的像素运动，避免将检测框抖动误判为运动。

即使检测器漏掉某一帧，仍使用解码后的相邻视频帧。
无法验证的运动会被排除。
"""

import cv2
import numpy as np


def _box(box, sx, sy, w, h):
    """缩放检测框并将坐标限制在图像范围内。"""
    x1, y1, x2, y2 = box
    return (
        max(0, min(w, int(round(x1 * sx)))),
        max(0, min(h, int(round(y1 * sy)))),
        max(0, min(w, int(round(x2 * sx)))),
        max(0, min(h, int(round(y2 * sy)))),
    )


def _appearance(image, box):
    """提取检测框内的 HSV 直方图，用于比较外观。"""
    x1, y1, x2, y2 = box
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(cv2.resize(crop, (80, 80)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)


def appearance_distance(a, b):
    """计算两个外观直方图之间的距离；缺少特征时返回空值。"""
    if a is None or b is None:
        return None
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _flow(src, dst, mask):
    """用前后向光流估计掩码区域的中位位移。"""
    pts = cv2.goodFeaturesToTrack(
        src, maxCorners=150, qualityLevel=0.01, minDistance=4, mask=mask, blockSize=5
    )
    if pts is None or len(pts) < 4:
        return None
    opts = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03),
    )
    target, status, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, **opts)
    if target is None:
        return None
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(dst, src, target, None, **opts)
    if back is None:
        return None
    valid = (
        (status[:, 0] > 0)
        & (back_status[:, 0] > 0)
        & (np.linalg.norm(back[:, 0] - pts[:, 0], axis=1) < 1.5)
    )
    if int(valid.sum()) < 4:
        return None
    return np.median((target - pts)[valid, 0], axis=0)


def _background(src, dst, box):
    """分网格估计背景位移，减少其他移动物体对相机运动估计的影响。"""
    h, w = src.shape
    x1, y1, x2, y2 = box
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[max(0, y1 - 10) : min(h, y2 + 10), max(0, x1 - 10) : min(w, x2 + 10)] = 0
    # 强角点可能集中在其他移动物体上。对各个网格分别采样并各计一票，
    # 避免单个移动物体主导相机运动估计。
    points, cell_ids = [], []
    for gy in range(5):
        ya, yb = gy * h // 5, (gy + 1) * h // 5
        for gx in range(8):
            xa, xb = gx * w // 8, (gx + 1) * w // 8
            block = src[ya:yb, xa:xb]
            block_mask = mask[ya:yb, xa:xb]
            if block.size == 0 or not np.any(block_mask):
                continue
            found = cv2.goodFeaturesToTrack(
                block, maxCorners=8, qualityLevel=0.01, minDistance=4, mask=block_mask, blockSize=5
            )
            if found is None:
                continue
            found = found + np.array([[[xa, ya]]], dtype=np.float32)
            points.append(found)
            cell_ids.extend([gy * 8 + gx] * len(found))
    if not points:
        return None
    pts = np.concatenate(points, axis=0)
    opts = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03),
    )
    target, status, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, **opts)
    if target is None:
        return None
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(dst, src, target, None, **opts)
    if back is None:
        return None
    valid = (
        (status[:, 0] > 0)
        & (back_status[:, 0] > 0)
        & (np.linalg.norm(back[:, 0] - pts[:, 0], axis=1) < 1.5)
    )
    cell_ids = np.asarray(cell_ids)
    vectors = (target - pts)[:, 0]
    medians = [
        np.median(vectors[valid & (cell_ids == cell)], axis=0)
        for cell in np.unique(cell_ids[valid])
        if np.sum(valid & (cell_ids == cell)) >= 2
    ]
    if len(medians) < 3:
        return None
    return np.median(np.asarray(medians), axis=0)


def _local_motion(src, dst, box, scale):
    """扣除背景位移，计算目标相对场景的运动幅度。"""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros(src.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    obj = _flow(src, dst, mask)
    bg = _background(src, dst, box)
    if obj is None or bg is None:
        return None
    return float(np.linalg.norm(obj - bg) / scale)


def _orb_data(gray, box, cache, key):
    """提取并缓存检测框内的 ORB 特征点和描述子。"""
    if key in cache:
        return cache[key]
    x1, y1, x2, y2 = box
    if x2 - x1 < 16 or y2 - y1 < 16:
        cache[key] = None
        return None
    orb = cv2.ORB_create(nfeatures=400, edgeThreshold=7, patchSize=15, fastThreshold=10)
    points, descriptors = orb.detectAndCompute(gray[y1:y2, x1:x2], None)
    if descriptors is None or len(points) < 5:
        cache[key] = None
        return None
    coords = np.array([p.pt for p in points], dtype=np.float32) + np.array(
        [x1, y1], dtype=np.float32
    )
    cache[key] = (coords, descriptors)
    return cache[key]


def _feature_motion(a, box_a, b, box_b, cache, key_a, key_b, scale):
    """使用图像特征对应关系，不设置位移上限。"""
    feat_a = _orb_data(a, box_a, cache, key_a)
    feat_b = _orb_data(b, box_b, cache, key_b)
    obj = feature_correspondence(feat_a, feat_b)
    if obj is None:
        return None
    bg = _background(a, b, box_a)
    if bg is None:
        return None
    return float(np.linalg.norm(obj - bg) / scale)


def feature_correspondence(feat_a, feat_b):
    """返回稳健的图像特征位移；匹配不明确时返回 None。"""
    if feat_a is None or feat_b is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(feat_a[1], feat_b[1], k=2)
    good = [
        pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
    ]
    # 重复纹理不能算作多个独立的特征对应点。
    unique = {}
    for match in good:
        if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
            unique[match.trainIdx] = match
    good = list(unique.values())
    if len(good) < 5:
        return None
    src = np.array([feat_a[0][m.queryIdx] for m in good], dtype=np.float32)
    dst = np.array([feat_b[0][m.trainIdx] for m in good], dtype=np.float32)
    transform, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if transform is None or inliers is None:
        return None
    valid = inliers.ravel().astype(bool)
    if int(valid.sum()) < 5 or float(valid.mean()) < 0.5:
        return None
    return np.median(dst[valid] - src[valid], axis=0)


def inspect_video_motion(video_path, frames, min_motion_px=1.0):
    """返回与输入帧对齐的运动候选和核查数据。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("无法读取视频以确认物体位移")
    grays, prepared, scales = [], [], []
    needed = set()
    for index, candidates in enumerate(frames):
        if candidates:
            needed.update(i for i in (index - 1, index, index + 1) if 0 <= i < len(frames))
    original_shape = None
    try:
        for index, candidates in enumerate(frames):
            ok, image = cap.read()
            if not ok:
                raise ValueError("视频帧数与检测结果不一致")
            h, w = image.shape[:2]
            if original_shape is None:
                original_shape = (h, w)
            elif original_shape != (h, w):
                raise ValueError("暂不支持视频分辨率中途变化")
            if index not in needed:
                grays.append(None)
                scales.append(None)
                prepared.append([])
                continue
            scale = min(1.0, 960.0 / w)
            small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            scales.append(scale)
            row = []
            for candidate in candidates:
                bounds = _box(
                    candidate["box"],
                    small.shape[1] / w,
                    small.shape[0] / h,
                    small.shape[1],
                    small.shape[0],
                )
                row.append(
                    {
                        "candidate": candidate,
                        "bounds": bounds,
                        "appearance": _appearance(small, bounds),
                    }
                )
            prepared.append(row)
    finally:
        cap.release()
    assert len(grays) == len(frames)
    result = [[] for _ in frames]
    audit = []
    orb_cache = {}
    for index, candidates in enumerate(prepared):
        for candidate_index, item in enumerate(candidates):
            local = []
            scale = scales[index]
            for other_index in (index - 1, index + 1):
                if 0 <= other_index < len(grays):
                    value = _local_motion(grays[index], grays[other_index], item["bounds"], scale)
                    if value is not None:
                        local.append(value)
            best = max(local) if local else None
            source = "adjacent_optical_flow" if best is not None else "unknown"
            # 光流可能跟丢快速移动的物体；若在其他位置再次检测到，
            # 则通过图像特征核查，不限制位移距离。
            if best is None:
                for gap in range(1, len(frames)):
                    found = False
                    for other_index in (index - gap, index + gap):
                        if not 0 <= other_index < len(prepared):
                            continue
                        for other_j, other in enumerate(prepared[other_index]):
                            distance = appearance_distance(item["appearance"], other["appearance"])
                            if distance is None or distance > 0.48:
                                continue
                            value = _feature_motion(
                                grays[index],
                                item["bounds"],
                                grays[other_index],
                                other["bounds"],
                                orb_cache,
                                (index, candidate_index),
                                (other_index, other_j),
                                scale,
                            )
                            if value is not None:
                                if best is None or value > best:
                                    best, source = value, "matched_object_features"
                                found = True
                                if value >= min_motion_px:
                                    break
                        if found:
                            break
                    if found:
                        break
            moving = best is not None and best >= min_motion_px
            row = {
                "frame": index + 1,
                "box": item["candidate"]["box"],
                "motion_px": best,
                "motion_source": source,
                "status": "moving" if moving else ("stationary" if best is not None else "unknown"),
            }
            audit.append(row)
            if moving:
                entry = dict(item["candidate"])
                entry["_appearance"] = item["appearance"]
                entry["_motion_px"] = best
                entry["_features"] = _orb_data(
                    grays[index], item["bounds"], orb_cache, (index, candidate_index)
                )
                result[index].append(entry)
    return result, audit
