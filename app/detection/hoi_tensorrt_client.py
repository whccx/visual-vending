"""调用独立 TensorRT 检测服务并校验逐帧数据。"""

import math
import base64
import binascii
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from app import config

LOG = logging.getLogger("sku." + __name__)


class HOIDetectorError(RuntimeError):
    """检测服务不可用或返回结果不符合契约。"""


class HOIDetectorTimeout(HOIDetectorError):
    """检测服务请求超时。"""


def _valid_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in box):
        return False
    return box[2] > box[0] and box[3] > box[1]


def validate_detection(result, expected_frames):
    """拒绝帧错位、无效坐标和关联索引越界的数据。"""
    if not isinstance(result, dict):
        raise HOIDetectorError("检测服务返回值不是对象")
    count = result.get("frame_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise HOIDetectorError("检测服务返回的帧数无效")
    if expected_frames > 0 and count != expected_frames:
        raise HOIDetectorError("检测服务返回帧数不匹配")
    for name in ("areas", "hands", "frames"):
        if not isinstance(result.get(name), list) or len(result[name]) != count:
            raise HOIDetectorError("检测服务返回的 {} 帧数不匹配".format(name))
    for area, hands, objects in zip(result["areas"], result["hands"], result["frames"]):
        if area is not None and not _valid_box(area):
            raise HOIDetectorError("检测服务返回无效货柜框")
        if not isinstance(hands, list) or not isinstance(objects, list):
            raise HOIDetectorError("检测服务返回无效的帧数据")
        if any(not isinstance(hand, dict) or not _valid_box(hand.get("box")) for hand in hands):
            raise HOIDetectorError("检测服务返回无效手框")
        for obj in objects:
            if not isinstance(obj, dict) or not _valid_box(obj.get("box")):
                raise HOIDetectorError("检测服务返回无效商品框")
            score, links = obj.get("score"), obj.get("hand_indices")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise HOIDetectorError("检测服务返回无效置信度")
            if not isinstance(links, list) or any(
                isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(hands)
                for index in links
            ):
                raise HOIDetectorError("检测服务返回无效手商品关联")
    return result


def detect_video(video_path, expected_frames):
    """发送共享目录内的视频路径，不传输视频帧或模型张量。"""
    path = Path(video_path)
    if not path.is_absolute():
        raise HOIDetectorError("HOI 视频路径必须为绝对路径")
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                config.HOI_TENSORRT_URL + "/detect",
                json={"video_path": str(path)},
                timeout=(config.HOI_TENSORRT_CONNECT_TIMEOUT, config.HOI_TENSORRT_REQUEST_TIMEOUT),
            )
            response.raise_for_status()
            result = response.json()
    except requests.Timeout as exc:
        raise HOIDetectorTimeout("HOI 检测服务请求超时") from exc
    except (requests.RequestException, ValueError) as exc:
        raise HOIDetectorError("HOI 检测服务请求失败") from exc
    return validate_detection(result, expected_frames)


def segment_video(video_path, points, threshold):
    """批量请求整帧 BG0，验证并解码每个商品框内的 PNG 掩码。"""
    if not points:
        return {}
    expected = {int(point["frame"]): list(point["box"]) for point in points}
    if len(expected) != len(points):
        raise HOIDetectorError("BG0 批次帧号重复")
    started = time.monotonic()
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                config.HOI_TENSORRT_URL + "/segment",
                json={"video_path": str(video_path), "points": points, "threshold": threshold},
                timeout=(config.HOI_TENSORRT_CONNECT_TIMEOUT, config.HOI_TENSORRT_REQUEST_TIMEOUT),
            )
            response.raise_for_status()
            result = response.json()
    except requests.Timeout as exc:
        raise HOIDetectorTimeout("BG0 检测服务请求超时") from exc
    except (requests.RequestException, ValueError) as exc:
        raise HOIDetectorError("BG0 检测服务请求失败") from exc
    if not isinstance(result, dict) or not isinstance(result.get("masks"), list):
        raise HOIDetectorError("BG0 服务返回无效掩码列表")
    masks = {}
    for item in result["masks"]:
        if not isinstance(item, dict):
            raise HOIDetectorError("BG0 服务返回无效掩码")
        frame, box = item.get("frame"), item.get("box")
        if (isinstance(frame, bool) or not isinstance(frame, int) or frame not in expected
                or frame in masks or box != expected[frame]):
            raise HOIDetectorError("BG0 服务返回帧号或商品框不一致")
        try:
            encoded = base64.b64decode(item.get("mask_png", ""), validate=True)
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise HOIDetectorError("BG0 服务返回无法解码的掩码") from exc
        height, width = box[3] - box[1], box[2] - box[0]
        if decoded is None or decoded.shape != (height, width) or not np.isin(decoded, [0, 255]).all():
            raise HOIDetectorError("BG0 服务返回掩码尺寸或数值无效")
        masks[frame] = decoded.astype(bool)
    if len(masks) != len(expected):
        raise HOIDetectorError("BG0 服务返回帧数不匹配")
    LOG.info("BG0 批量服务：帧数=%s 请求及解码耗时=%.3f 秒，服务耗时=%s",
             len(masks), time.monotonic() - started, result.get("timings"))
    return masks
