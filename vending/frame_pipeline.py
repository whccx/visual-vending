"""跨视频共享帧任务的识别执行器。"""

import base64
import io
import json
import logging
import math
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import requests
from PIL import Image

from app import config
from vending.engine import Pipeline, unmatched_video_result, unpack_features
from vending.media import MediaError
from vending.store import outcome

LOG = logging.getLogger("vending.frame_pipeline")
CODEC_LOCK = threading.Lock()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_png(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f"图片保存失败：{path}")


def read_frame(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"视频帧无法读取：{path}")
    return image


def validate_jpeg(data: bytes, width: int, height: int) -> None:
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("JPEG 标记不完整")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != (width, height):
            raise ValueError("JPEG 图片尺寸不正确")


def split_video(path: Path, directory: Path, publish) -> int:
    """NVDEC 解码并按帧发布已校验的 nvJPEG 输出。"""
    import torch

    sys.path.insert(0, str(config.GPU_RUNTIME / "nvcodec"))
    import PyNvVideoCodec as nvc
    from torchvision.io import encode_jpeg

    decoder = nvc.SimpleDecoder(
        str(path), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
    )
    stream = torch.cuda.Stream()
    meta = decoder.get_stream_metadata()
    count = len(decoder)
    if (meta.width, meta.height) != (1920, 1080) or count <= 0:
        raise MediaError("视频尺寸或帧数不符合检测要求")
    publish("metadata", 0, count)
    for first in range(0, count, 8):
        length = min(8, count - first)
        with CODEC_LOCK, torch.cuda.stream(stream):
            decoded = decoder.get_batch_frames(length)
            if len(decoded) != length:
                raise MediaError("视频解码提前结束")
            images = [torch.from_dlpack(frame).clone() for frame in decoded]
            torch.cuda.synchronize()
            encoded = encode_jpeg(images, quality=100)
            torch.cuda.synchronize()
            payloads = [item.cpu().numpy().tobytes() for item in encoded]
        for index, (image, data) in enumerate(zip(images, payloads), first + 1):
            try:
                validate_jpeg(data, meta.width, meta.height)
            except (OSError, ValueError):
                LOG.warning("视频帧 JPEG 异常，使用 CPU 重新编码：帧=%s", index)
                with CODEC_LOCK, torch.cuda.stream(stream):
                    rgb = image.permute(1, 2, 0).contiguous().cpu().numpy()
                ok, encoded = cv2.imencode(
                    ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                if not ok:
                    raise MediaError("视频帧 JPEG 编码失败")
                data = encoded.tobytes()
                validate_jpeg(data, meta.width, meta.height)
            target = directory / "frames" / f"{index:06d}.jpg"
            target.with_suffix(".tmp").write_bytes(data)
            target.with_suffix(".tmp").replace(target)
            save_json(target.with_suffix(".json"), {"frame": index})
            publish("frame", index, None)
    return count


class FrameRuntime:
    """共享 CPU、GPU 和检测服务的有限并发池。"""

    def __init__(self, video_concurrency=5):
        cv2.setNumThreads(1)
        self.video_pool = ThreadPoolExecutor(
            max_workers=video_concurrency, thread_name_prefix="视频"
        )
        self.area_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="柜区")
        self.gate_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="筛帧")
        self.hoi_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="交互检测")
        self.bg0_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="商品抠图")
        self.score_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="商品比对")
        self.local = threading.local()
        self.model_lock = threading.Lock()
        self.fallback_lock = threading.Lock()
        self.fallback_pipeline = Pipeline()
        self.model = None

    def close(self):
        for pool in (
            self.video_pool,
            self.area_pool,
            self.gate_pool,
            self.hoi_pool,
            self.bg0_pool,
            self.score_pool,
        ):
            pool.shutdown(wait=True, cancel_futures=True)

    def area(self, image_path, output):
        from app.detection.area_detection import detect_frame_mask_with_confidence, load_model

        if not hasattr(self.local, "area"):
            self.local.area = load_model()
        mask, box, confidence = detect_frame_mask_with_confidence(
            read_frame(image_path), self.local.area
        )
        if mask is not None:
            save_png(output.with_name(output.stem + "-mask.png"), mask.astype(np.uint8) * 255)
        save_json(output, {"frame": int(output.stem), "box": box, "confidence": confidence})
        return box

    def gate(self, image_path, previous_path, output):
        import mediapipe as mp

        from app.detection.held_product_detect import load_gpu_hands

        if not hasattr(self.local, "hands"):
            self.local.hands = load_gpu_hands()
        image = read_frame(image_path)
        height, width = image.shape[:2]
        image = cv2.resize(image, (960, round(height * 960 / width)))
        hands = []
        for rotation in (
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ):
            rotated = image if rotation is None else cv2.rotate(image, rotation)
            rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
            result = self.local.hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            hands.append(len(result.hand_landmarks))
        current = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if current is None:
            raise ValueError("筛帧图片无法读取")
        current = cv2.resize(current, (480, 270), interpolation=cv2.INTER_AREA)
        motion = None
        if previous_path is not None:
            previous = cv2.imread(str(previous_path), cv2.IMREAD_GRAYSCALE)
            if previous is None:
                raise ValueError("筛帧的上一采样帧无法读取")
            previous = cv2.resize(previous, (480, 270), interpolation=cv2.INTER_AREA)
            diff = current.astype(np.int16) - previous.astype(np.int16)
            diff -= int(np.median(diff))
            motion = float((np.abs(diff) >= 15).mean())
        keep = any(hands) or motion is None or motion >= 0.005
        save_json(
            output,
            {
                "frame": int(output.stem),
                "rotation_counts": hands,
                "motion_fraction": motion,
                "keep": keep,
            },
        )
        return keep

    @staticmethod
    def request(path, body):
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                config.HOI_TENSORRT_URL + path,
                json=body,
                timeout=(config.HOI_TENSORRT_CONNECT_TIMEOUT, config.HOI_TENSORRT_REQUEST_TIMEOUT),
            )
            response.raise_for_status()
            return response.json()

    def detect(self, image_path, area, inferred, output):
        from app.detection.hoi_tensorrt_client import validate_detection

        hands, products = [], []
        if inferred:
            data = self.request("/frame/detect", {"image_path": str(image_path)})
            hands, products = data["hands"], data["products"]
        validate_detection(
            {"frame_count": 1, "areas": [area], "hands": [hands], "frames": [products]}, 1
        )
        save_json(
            output,
            {
                "frame": int(output.stem),
                "area": area,
                "inferred": inferred,
                "hands": hands,
                "products": products,
            },
        )

    def segment(self, image_path, box, output):
        data = self.request(
            "/frame/segment",
            {"image_path": str(image_path), "box": box, "threshold": config.BG0_MASK_THRESHOLD},
        )
        raw = base64.b64decode(data["mask_png"], validate=True)
        mask = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if (
            mask is None
            or mask.shape != (box[3] - box[1], box[2] - box[0])
            or not np.isin(mask, [0, 255]).all()
        ):
            raise ValueError("BG0 服务返回无效掩码")
        save_png(output, mask)
        metadata = output.with_suffix(".json")
        save_json(metadata, {"frame": int(image_path.stem), "box": box, "mask": output.name})
        return metadata

    def dino(self):
        if self.model is None:
            with self.model_lock:
                if self.model is None:
                    from transformers import AutoModel

                    self.model = (
                        AutoModel.from_pretrained(str(config.MODEL), local_files_only=True)
                        .eval()
                        .cuda()
                    )
        return self.model

    def score(self, image_path, metadata_path, output, catalog):
        import torch

        from app.recognition.hoi_mask_matching import (
            _enhance_crop,
            _reject_small_mask,
            _remove_hand_from_mask,
        )
        from app.recognition.product_matching import score_crop

        metadata = json.loads(metadata_path.read_text())
        box = metadata["box"]
        raw_mask = cv2.imread(str(metadata_path.parent / metadata["mask"]), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError("商品掩码无法读取")
        mask = raw_mask > 0
        image = read_frame(image_path)
        x1, y1, x2, y2 = box
        save_png(output.with_suffix(".png"), image[y1:y2, x1:x2])
        item = {"frame": int(image_path.stem), "mask": mask, "bg0_fraction": float(mask.mean())}
        if config.HOI_REMOVE_HAND and item["bg0_fraction"] >= config.HOI_MASK_MATCH_MIN_FRACTION:
            if not hasattr(self.local, "score_state"):
                self.local.score_state = SimpleNamespace()
            item["mask"], _, _ = _remove_hand_from_mask(image, box, mask, self.local.score_state)
        _reject_small_mask(item)
        if item["mask"] is None:
            save_json(output, {"frame": int(image_path.stem), "box": box, "scores": None})
            return output
        rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        enhanced = _enhance_crop(rgb)
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
        with torch.inference_mode():
            _, scores = score_crop(
                torch.from_numpy(enhanced).cuda().permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(item["mask"])).cuda(),
                self.dino(),
                catalog,
                mean,
                std,
                color_crop=torch.from_numpy(rgb).cuda().permute(2, 0, 1),
            )
        if any(not math.isfinite(value) for pair in scores.values() for value in pair):
            LOG.warning("商品比对存在非有限分数，跳过帧：%s", image_path.name)
            scores = None
        save_json(output, {"frame": int(image_path.stem), "box": box, "scores": scores})
        return output

    def recognize(self, videos, cabinet_id, assets, *, comparison_assets=None):
        catalog = {
            index: unpack_features(data)
            for index, data in enumerate(
                [*assets.values(), *(comparison_assets or {}).values()], start=1
            )
        }
        sku_ids = dict(enumerate(assets, start=1))
        futures = [
            self.video_pool.submit(self.recognize_view, path, cabinet_id, catalog, sku_ids)
            for path in videos
        ]
        try:
            views = [future.result() for future in futures]
        finally:
            wait(futures)
        primary = views[0]
        if len(views) == 1 or primary["status"]["code"] not in (2, 3, 7):
            return primary
        if any(
            view["status"]["code"] != primary["status"]["code"]
            or view.get("result") != primary.get("result")
            for view in views[1:]
        ):
            return outcome(4, "主辅摄像头结果不一致，需复核")
        return primary

    def recognize_view(self, path, cabinet_id, catalog, sku_ids):
        from vending.engine import probe_video

        probe_video(path)
        directory = path.parent / (path.stem + "-frames")
        directory.mkdir()
        for name in ("frames", "area", "gate", "hoi", "bg0", "scores"):
            (directory / name).mkdir()
        area_futures, gate_futures, hoi_futures = {}, {}, {}
        condition = threading.Condition()
        state = {"count": None, "error": None}

        def publish(kind, frame, value):
            with condition:
                if kind == "metadata":
                    state["count"] = value
                else:
                    image = directory / "frames" / f"{frame:06d}.jpg"
                    area_futures[frame] = self.area_pool.submit(
                        self.area, image, directory / "area" / f"{frame:06d}.json"
                    )
                    if (frame - 1) % 5 == 0:
                        previous = (
                            directory / "frames" / f"{frame - 5:06d}.jpg" if frame > 5 else None
                        )
                        gate_futures[frame] = self.gate_pool.submit(
                            self.gate, image, previous, directory / "gate" / f"{frame:06d}.json"
                        )
                condition.notify_all()

        def wait_future(collection, number):
            with condition:
                while number not in collection:
                    if state["error"] is not None:
                        raise state["error"]
                    condition.wait(0.1)
                return collection[number]

        def schedule():
            with condition:
                while state["count"] is None:
                    if state["error"] is not None:
                        raise state["error"]
                    condition.wait(0.1)
                count = state["count"]
            last_area = None
            gate_cache = {}
            for frame in range(1, count + 1):
                area = wait_future(area_futures, frame).result()
                if area is not None:
                    last_area = frame
                start, end = max(1, frame - 10), min(count, frame + 10)
                first = 1 + max(0, (start - 1 + 4) // 5) * 5
                last = 1 + ((end - 1) // 5) * 5
                for sample in range(first, last + 1, 5):
                    if sample not in gate_cache:
                        gate_cache[sample] = wait_future(gate_futures, sample).result()
                keep = any(gate_cache[sample] for sample in range(first, last + 1, 5))
                inferred = (
                    last_area is not None
                    and frame - last_area <= config.hoi_area_missing_max_frames()
                    and keep
                )
                image = directory / "frames" / f"{frame:06d}.jpg"
                hoi_futures[frame] = self.hoi_pool.submit(
                    self.detect, image, area, inferred, directory / "hoi" / f"{frame:06d}.json"
                )
            for future in hoi_futures.values():
                future.result()

        # Future 保存调度线程异常，确保解码和模型错误都能交给当前订单处理。
        try:
            with ThreadPoolExecutor(max_workers=1) as coordinator:
                scheduling = coordinator.submit(schedule)
                try:
                    split_video(path, directory, publish)
                except Exception as error:
                    with condition:
                        state["error"] = error
                        condition.notify_all()
                    try:
                        scheduling.result()
                    except Exception:
                        LOG.debug("视频解码失败后逐帧调度已退出", exc_info=True)
                    raise
                scheduling.result()
        finally:
            wait([*area_futures.values(), *gate_futures.values(), *hoi_futures.values()])
        return self.match_video(directory, catalog, sku_ids)

    def match_video(self, directory, catalog, sku_ids):
        from app.api.recognize_hoi_detr import _classify_without_match
        from app.detection.camera_block import judge_camera_blocked
        from app.detection.hand_area_gate import (
            cabinet_masks_for_candidates,
            filter_hand_linked_candidates,
        )
        from app.recognition.product_matching import (
            prepare_catalog,
            ranking_rejection_reason,
            select_scoring_window,
        )
        from app.tracking.hoi_detr_motion import inspect_video_motion
        from app.tracking.trajectory_hoi_detr import build_visual_motion_tracks

        files = sorted((directory / "hoi").glob("*.json"))
        detection = [json.loads(path.read_text()) for path in files]
        areas = [row["area"] for row in detection]
        hands = [row["hands"] for row in detection]
        products = [row["products"] for row in detection]
        sequence = directory / "frames" / "%06d.jpg"
        if judge_camera_blocked(sequence, areas):
            return outcome(6, "检测到与柜门状态不一致的持续遮挡，需复核")
        packed = []
        for row in detection:
            mask = None
            if row["area"] is not None and row["hands"] and row["products"]:
                mask = cv2.imread(
                    str(directory / "area" / f"{row['frame']:06d}-mask.png"), cv2.IMREAD_GRAYSCALE
                )
                if mask is None:
                    raise ValueError("柜区掩码缺失")
            packed.append(np.packbits(mask.reshape(-1)) if mask is not None else None)
        masks = cabinet_masks_for_candidates(packed, (1080, 1920), hands, products)
        candidates, gate = filter_hand_linked_candidates(
            hands,
            products,
            areas,
            max_linked_hand_area_ratio=config.hoi_max_product_hand_area_ratio(),
            require_departure=True,
            hand_track_base_distance=config.hoi_hand_track_base_distance(),
            hand_track_iou_threshold=config.hoi_hand_track_iou_threshold(),
            area_masks_by_frame=masks,
        )
        LOG.info("货柜掩码轨迹筛选：%s", gate)
        moving, _ = inspect_video_motion(sequence, candidates)
        tracks = {}
        for track in build_visual_motion_tracks(moving):
            points = track["points"]
            if len(points) < config.SCORE_WINDOW_FRAMES:
                continue
            track_id = track["last_candidate"]["area_track_id"]
            motion = sum(point["motion_px"] for point in points)
            if track_id not in tracks or motion > tracks[track_id][0]:
                tracks[track_id] = (motion, points)
        if not tracks:
            with self.fallback_lock:
                status, area_frames, departure = _classify_without_match(
                    {"areas": areas, "hands": hands, "frames": products},
                    sequence,
                    self.fallback_pipeline,
                    masks,
                )
            return unmatched_video_result(
                status,
                departure,
                gate["taken_tracks"],
                door_opened=area_frames >= config.hoi_door_min_area_frames(),
            )
        if len(tracks) != gate["taken_tracks"]:
            return outcome(4, "存在未形成有效识别轨迹的拿取，需复核")
        matching = prepare_catalog(catalog)
        counts = Counter()
        unmatched = outside_scope = False
        for _, points in tracks.values():
            futures = []
            for point in points:
                x1, y1, x2, y2 = point["box"]
                if x1 < 19.2 or y1 < 10.8 or x2 > 1900.8 or y2 > 1069.2:
                    continue
                box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
                frame = point["frame"]
                image = directory / "frames" / f"{frame:06d}.jpg"
                mask_path = directory / "bg0" / f"{frame:06d}-{'-'.join(map(str, box))}.png"
                futures.append(
                    (frame, image, box, self.bg0_pool.submit(self.segment, image, box, mask_path))
                )
            scoring = []
            try:
                for frame, image, box, future in futures:
                    metadata = future.result()
                    output = directory / "scores" / f"{frame:06d}-{'-'.join(map(str, box))}.json"
                    scoring.append(
                        (
                            frame,
                            self.score_pool.submit(self.score, image, metadata, output, matching),
                        )
                    )
                scores = []
                for frame, future in scoring:
                    record = json.loads(future.result().read_text())
                    if record["scores"] is not None:
                        scores.append(
                            (frame, {int(key): value for key, value in record["scores"].items()})
                        )
            finally:
                wait([future for _, _, _, future in futures])
                wait([future for _, future in scoring])
            frames, ranking = select_scoring_window(
                scores,
                matching[0],
                sku_scope=sku_ids.keys() if len(catalog) > len(sku_ids) else None,
            )
            rejection = ranking_rejection_reason(ranking, len(frames))
            if rejection == "ambiguous_margin":
                return outcome(
                    4, f"前两名商品综合分差小于 {config.SCORE_MIN_MARGIN:g}，无法确认商品"
                )
            if rejection == "insufficient_evidence":
                return outcome(4, "商品轨迹的有效评分帧或匹配证据不足")
            if rejection == "unmatched_scores":
                unmatched = True
            elif ranking[0]["sku"] in sku_ids:
                counts[sku_ids[ranking[0]["sku"]]] += 1
            else:
                outside_scope = True
        if unmatched:
            return outcome(7, "已确认拿取，存在未能匹配本单范围内 SKU 的商品")
        if outside_scope:
            return outcome(7, "已确认拿取，商品可靠匹配到本单范围外的 SKU")
        return outcome(
            2, items=[{"ks_sku_id": sku, "sku_num": count} for sku, count in sorted(counts.items())]
        )
