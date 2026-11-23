"""复用 SKU 项目的图片建模、运动轨迹和 HOI-DETR 匹配。"""

import hashlib
import json
import logging
import math
import subprocess
from collections import Counter
from functools import cached_property
from io import BytesIO
from pathlib import Path

from vending.media import MediaError
from vending.store import outcome

LOG = logging.getLogger("vending.engine")


class ImageRejected(ValueError):
    """图片内容未达到建模条件。"""


def image_hashes(images: list[bytes]) -> list[str]:
    return [hashlib.sha256(image).hexdigest() for image in images]


def pack_features(features, colors, full_colors) -> bytes:
    import numpy as np

    stream = BytesIO()
    np.savez_compressed(stream, features=features, colors=colors, full_colors=full_colors)
    return stream.getvalue()


def unpack_features(data: bytes) -> dict:
    import numpy as np

    with np.load(BytesIO(data), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in ("features", "colors", "full_colors")}


def probe_video(path: Path) -> tuple[int, int, int, float]:
    import cv2

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name:format=format_name",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        info = json.loads(result.stdout)
        containers = set(info["format"]["format_name"].split(","))
        if (
            not containers.intersection({"mp4", "avi", "webm"})
            or info["streams"][0]["codec_name"] != "h264"
        ):
            raise MediaError("视频必须为 MP4、AVI 或 WebM 容器及 H.264 编码")
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        if isinstance(exc, MediaError):
            raise
        raise MediaError("视频无法解析") from exc

    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            not capture.isOpened()
            or min(count, width, height) <= 0
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise MediaError("视频没有可读帧")
        dark = []
        for frame in sorted({0, count // 4, count // 2, 3 * count // 4, count - 1}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if ok:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dark.append(float((gray < 18).mean()) >= 0.98)
        if not dark or all(dark):
            raise MediaError("视频无法读取或采样画面持续黑屏")
    finally:
        capture.release()
    return width, height, count, fps


class Pipeline:
    """以单进程 GPU worker 复用模型状态。"""

    @property
    def dino(self):
        if not hasattr(self, "_dino"):
            from transformers import AutoModel

            from app import config

            self._dino = (
                AutoModel.from_pretrained(str(config.MODEL), local_files_only=True).eval().cuda()
            )
        return self._dino

    @cached_property
    def _held_product_models(self):
        from app.detection.held_product_detect import load_detector

        return load_detector()

    @property
    def detector(self):
        return self._held_product_models[0]

    @property
    def hands(self):
        return self._held_product_models[1]

    def build(self, images: list[bytes], *, check_size: bool) -> bytes:
        import numpy as np

        from app.catalog.product_catalog import decode, features_for_images
        from app.recognition.full_color import from_bgra

        full_colors = []
        for image in images:
            try:
                pixels = decode(image)[0]
            except ValueError as exc:
                raise ImageRejected("参考图片无法解码") from exc
            height, width = pixels.shape[:2]
            landscape = (max(width, height), min(width, height))
            if check_size and (
                landscape[0] < 1920 or landscape[1] < 1080 or landscape[0] / landscape[1] > 16 / 9
            ):
                raise ImageRejected("图片尺寸或长宽比不符合规范")
            full_colors.append(from_bgra(pixels))
        features, colors = features_for_images(images, model=self.dino)
        return pack_features(features, colors, np.asarray(full_colors, dtype=np.float32))

    def recognize(self, videos: list[Path], cabinet_id: str, assets: dict[str, bytes]) -> dict:
        catalog = {
            index: unpack_features(data) for index, data in enumerate(assets.values(), start=1)
        }
        sku_ids = dict(enumerate(assets, start=1))
        views = [self._recognize_view(path, cabinet_id, catalog, sku_ids) for path in videos]
        primary = views[0]
        if len(views) == 1:
            return primary
        if primary["status"]["code"] not in (2, 3):
            return primary
        if any(
            view["status"]["code"] != primary["status"]["code"]
            or view.get("result") != primary.get("result")
            for view in views[1:]
        ):
            return outcome(4, "主辅摄像头结果不一致，需复核")
        return primary

    def _recognize_view(self, path: Path, cabinet_id: str, catalog: dict, sku_ids: dict) -> dict:
        from app import config
        from app.api.recognize_hoi_detr import _classify_without_match
        from app.detection import camera_block
        from app.detection.hand_area_gate import filter_hand_linked_candidates
        from app.detection.hoi_tensorrt_client import detect_video
        from app.recognition.hoi_mask_matching import rank_masked_track
        from app.recognition.product_matching import accept_ranking, prepare_catalog
        from app.tracking.hoi_detr_motion import inspect_video_motion
        from app.tracking.trajectory_hoi_detr import build_visual_motion_tracks

        width, height, count, _ = probe_video(path)
        detection = detect_video(path, count)
        if camera_block.judge_camera_blocked(path, detection["areas"]):
            return outcome(6, "检测到与柜门状态不一致的持续遮挡，需复核")
        frames, gate = filter_hand_linked_candidates(
            detection["hands"],
            detection["frames"],
            detection["areas"],
            max_linked_hand_area_ratio=config.hoi_max_product_hand_area_ratio(),
            require_departure=True,
            hand_track_base_distance=config.hoi_hand_track_base_distance(),
            hand_track_iou_threshold=config.hoi_hand_track_iou_threshold(),
        )
        moving, _ = inspect_video_motion(path, frames)
        all_tracks = build_visual_motion_tracks(moving)
        tracks = {}
        for track in all_tracks:
            points = track["points"]
            track_id = track["last_candidate"]["area_track_id"]
            if len(points) < config.SCORE_WINDOW_FRAMES:
                continue
            score = sum(point["motion_px"] for point in points)
            if track_id not in tracks or score > tracks[track_id][0]:
                tracks[track_id] = (score, track)

        if not tracks:
            status, _ = _classify_without_match(detection, path, self)
            if status == "door_opened_no_taken" and gate["taken_tracks"] == 0:
                return outcome(3)
            return outcome(4, "未获得足够的商品离架或身份匹配证据")
        if len(tracks) != gate["taken_tracks"]:
            return outcome(4, "存在离架轨迹未能确认商品，需复核")

        matching = prepare_catalog(catalog)
        quantities = Counter()
        for _, track in tracks.values():
            mx, my = width * 0.01, height * 0.01
            track = dict(
                track,
                points=[
                    point
                    for point in track["points"]
                    if point["box"][0] >= mx
                    and point["box"][1] >= my
                    and point["box"][2] <= width - mx
                    and point["box"][3] <= height - my
                ],
            )
            ranking, valid, _ = rank_masked_track(
                path,
                track,
                self.dino,
                cabinet_id,
                self,
                saved_features=catalog,
                matching_catalog=matching,
                use_trt_bg0=True,
            )
            best = accept_ranking(ranking, valid)
            if not best:
                return outcome(4, "存在商品轨迹无法匹配本单范围，需复核")
            quantities[sku_ids[best["sku"]]] += 1
        items = [
            {"ks_sku_id": sku_id, "sku_num": count} for sku_id, count in sorted(quantities.items())
        ]
        return outcome(2, items=items)
