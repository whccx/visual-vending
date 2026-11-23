"""摄像头遮挡判定：开门画面被挡黑后恢复或黑得反常。

复用 HOI 检测已有的逐帧货柜框做状态机，只在确认过货柜检出之后统计黑帧；
黑帧占比与画面相似检查仅需 CPU 解码视频，不引入额外模型推理。
"""

import logging
from collections import deque

import cv2
import numpy as np

from app import config

LOG = logging.getLogger("sku." + __name__)

# dHash 缩放宽度；9 列灰度图产生 64 位水平差值哈希。
HASH_WIDTH = 9


def dark_ratio(gray, pixel_max):
    """返回灰度图中低于 pixel_max 的暗像素占比。"""
    return float((gray < pixel_max).mean())


def dhash(gray):
    """把灰度图缩放后取水平差值，得到 64 位感知哈希。"""
    small = cv2.resize(gray, (HASH_WIDTH, HASH_WIDTH - 1), interpolation=cv2.INTER_AREA)
    bits = np.packbits(small[:, 1:] > small[:, :-1])
    return int.from_bytes(bits.tobytes(), "big")


def hamming_distance(left, right):
    """返回两个哈希的汉明距离。"""
    return (left ^ right).bit_count()


def blocked_from_observations(observations, required):
    """根据逐帧 (有货柜, 画面黑) 观测归类遮挡事件。

    返回：
        "none"：未确认过货柜检出或未出现黑帧事件，全程黑或暗光场景属此类；
        "recovered"：确认正常态后出现连续"无货柜且黑帧"，且其后连续
        required 帧画面变亮，即"后续画面再次出现变化"；
        "persisted"：黑帧事件后直到结尾画面仍是黑帧——可能是镜头被挡
        到结尾，也可能是正常关门灯灭，由调用方结合亮度进一步区分。

    黑帧中重新检出货柜（哪怕连续多帧）只是镜头缝隙的误检，不算恢复；
    开局就是黑帧、从未确认货柜检出的视频不产生事件。
    """
    present_streak = 0
    dark_streak = 0
    change_streak = 0
    saw_present = False
    triggered = False
    for present, dark in observations:
        if triggered:
            # "后续画面再次出现变化"：画面重新变亮；黑帧中的货柜误检不算。
            if not dark:
                change_streak += 1
                if change_streak >= required:
                    return "recovered"
            else:
                change_streak = 0
            continue
        if present:
            present_streak += 1
            dark_streak = 0
            if present_streak >= required:
                saw_present = True
        else:
            present_streak = 0
            dark_streak = dark_streak + 1 if dark else 0
            if saw_present and dark_streak >= required:
                triggered = True
    return "persisted" if triggered else "none"


def scenes_similar(start_hashes, end_hashes, start_mean, end_mean):
    """开局段与结束段结构相似且整体明暗接近时视为画面恢复。

    结构用 dHash 汉明距离判断；纯结构哈希会把黑帧与简单亮场景误判相似，
    再用平均灰度差兜住"镜头仍被挡黑"的结束段。
    """
    if not start_hashes or not end_hashes:
        return False
    distance = min(hamming_distance(left, right) for left in start_hashes for right in end_hashes)
    if distance > config.hoi_block_hash_distance():
        return False
    return abs(start_mean - end_mean) <= config.hoi_block_brightness_delta()


def judge_camera_blocked(video_path, areas):
    """综合黑帧状态机与结束段相似度，判定摄像头是否被遮挡。

    正常开关门是 无→有→无，外置摄像头开局无货柜、柜内摄像头开局即可
    检出货柜，因此不预设开局状态。确认过货柜检出后出现连续"画面黑且
    无货柜"即为遮挡事件，之后分两种情况：画面重新变亮并恢复到开局画面
    （结束段相似开局段）则判遮挡；黑帧持续到结尾时，与开局黑段（关门
    灯灭基准）比较亮度——结尾明显更黑说明镜头被挡而非关门，判遮挡。
    仅对确认正常态之后的无货柜帧计算黑帧占比，结束段哈希只在正常态
    确认后采样。

    参数：
        video_path: 与 areas 逐帧对齐的视频路径。
        areas: HOI 检测返回的逐帧货柜框，未检出为 None。
    返回：
        遮挡事件成立且恢复或结尾黑得反常时为 True。
    异常：
        视频无法打开时抛出 ValueError。
    """
    total = len(areas)
    if not total:
        return False
    required = config.hoi_block_confirm_frames()
    sample = config.hoi_block_sample_frames()
    pixel_max = config.hoi_block_dark_pixel_max()
    frame_ratio = config.hoi_block_dark_frame_ratio()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("无法打开视频")
    start_hashes = []
    start_means = []
    start_dark = []
    end_hashes = deque(maxlen=sample)
    end_means = deque(maxlen=sample)
    observations = []
    present_streak = 0
    saw_present = False
    index = 0
    try:
        while index < total:
            ok, frame = capture.read()
            if not ok:
                LOG.warning("遮挡判定视频在第 %s 帧提前结束", index + 1)
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(start_hashes) < sample:
                start_hashes.append(dhash(gray))
                start_means.append(float(gray.mean()))
                start_dark.append(dark_ratio(gray, pixel_max) >= frame_ratio)
            present = areas[index] is not None
            # 正常态确认后的帧都按真实亮度记黑帧标记：黑帧事件与"变亮恢复"
            # 均以实际画面为准，货柜误检出的黑帧不能伪装成亮帧。
            dark = saw_present and dark_ratio(gray, pixel_max) >= frame_ratio
            observations.append((present, dark))
            if present:
                present_streak += 1
                if present_streak >= required:
                    saw_present = True
            else:
                present_streak = 0
            # 结束段取视频最后 sample 帧，正常态确认后逐帧滚动采样。
            if saw_present:
                end_hashes.append(dhash(gray))
                end_means.append(float(gray.mean()))
            index += 1
    finally:
        capture.release()
    event = blocked_from_observations(observations, required)
    if event == "none":
        return False
    start_mean = float(np.mean(start_means))
    end_mean = float(np.mean(end_means))
    if event == "recovered":
        similar = scenes_similar(start_hashes, end_hashes, start_mean, end_mean)
        LOG.info(
            "HOI 遮挡判定：黑帧事件后画面恢复，开局/结束段最小哈希距离与灰度差判定=%s"
            "（灰度差 %.1f，阈值 %s）",
            similar, abs(start_mean - end_mean), config.hoi_block_brightness_delta(),
        )
        return similar
    # persisted：黑帧持续到结尾。开局黑段是该相机关门灯灭的基准亮度，
    # 结尾比基准还黑得多不可能是灯灭关门，只能是镜头被挡。
    if not end_means or sum(start_dark) * 2 < len(start_dark):
        LOG.info("HOI 遮挡判定：黑帧事件持续到结尾，开局无黑段基准，不判遮挡")
        return False
    verdict = end_mean < start_mean * config.hoi_block_tail_ratio()
    LOG.info(
        "HOI 遮挡判定：黑帧事件持续到结尾，开局平均亮度 %.1f，结尾平均亮度 %.1f"
        "（阈值 %.1f），判定=%s",
        start_mean, end_mean, start_mean * config.hoi_block_tail_ratio(), verdict,
    )
    return verdict
