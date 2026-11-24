"""在独立容器中逐步验证视频拆帧、柜区分割和 HOI 检测。

split、area 在项目的 worker 镜像中运行，split 启用 NVIDIA 的 video 能力。
hoi 在 detector 镜像中运行，将脚本和 app 挂入容器并保留 /runtime:/service。
两个容器将同一个可写目录挂载到同一路径，各阶段依次读取上一阶段的产物。
"""

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import cv2

from app import config

PNG_OPTIONS = [cv2.IMWRITE_PNG_COMPRESSION, 1]
JOB_COUNTS = (1, 3, 5)


def write_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_png(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image, PNG_OPTIONS):
        raise OSError(f"图片保存失败 {path}")


def read_json(path: Path):
    return json.loads(path.read_text())


def read_frame(directory: Path, frame: int, suffix: str):
    path = directory / f"{frame:06d}{suffix}"
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"拆帧图片读取失败 {path}")
    return image


@contextmanager
def measure(timings: dict, key: str):
    started = time.monotonic()
    try:
        yield
    finally:
        timings[key] += time.monotonic() - started


class Overlap:
    """统计进入模型调用且尚未返回的峰值数量。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.calls = 0
        self.updated = time.monotonic()
        self.seconds_by_active = defaultdict(float)

    def _change(self, count: int) -> None:
        now = time.monotonic()
        if self.active:
            self.seconds_by_active[self.active] += now - self.updated
        self.updated = now
        self.active += count

    @contextmanager
    def call(self):
        with self.lock:
            self._change(1)
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            with self.lock:
                self._change(-1)

    def report(self) -> dict:
        """返回每个并发数维持的墙钟时长，模型调用包括该阶段的后处理。"""
        return {
            "peak_model_calls": self.peak,
            "model_calls": self.calls,
            "model_wall_seconds_by_active": dict(self.seconds_by_active),
        }


def split_cpu_png(source: Path, frames: Path, timings: dict) -> tuple[int, float]:
    """按原视频顺序保存无损帧图。"""
    options = [cv2.CAP_PROP_N_THREADS, 2] if hasattr(cv2, "CAP_PROP_N_THREADS") else []
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, options)
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开视频 {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            with measure(timings, "decode_seconds"):
                ok, image = capture.read()
            if not ok:
                break
            if image.shape != (1080, 1920, 3):
                raise ValueError(f"HOI 引擎需要 1920×1080 视频 {source}")
            count += 1
            with measure(timings, "png_write_seconds"):
                write_png(frames / f"{count:06d}.png", image)
    finally:
        capture.release()
    if count == 0 or (expected > 0 and expected != count):
        raise ValueError(f"帧数异常 {source}，实际 {count}，元数据 {expected}")
    return count, fps


def split_nvdec(source: Path, frames: Path, timings: dict, suffix: str) -> tuple[int, float]:
    """用 NVDEC 解码到显存，JPEG 模式在 GPU 上编码后再取回字节。"""
    import PyNvVideoCodec as nvc
    import torch
    from torchvision.io import encode_jpeg

    decoder = nvc.SimpleDecoder(
        str(source), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
    )
    metadata = decoder.get_stream_metadata()
    if (metadata.height, metadata.width) != (1080, 1920):
        raise ValueError(f"HOI 引擎需要 1920×1080 视频 {source}")
    for first in range(0, len(decoder), 8):
        size = min(8, len(decoder) - first)
        with measure(timings, "nvdec_seconds"):
            images = [torch.from_dlpack(frame) for frame in decoder.get_batch_frames(size)]
        if len(images) != size or any(image.shape != (3, 1080, 1920) for image in images):
            raise ValueError(f"NVDEC 返回缺失或尺寸异常的帧 {source}，起始帧 {first + 1}")
        if suffix == ".jpg":
            with measure(timings, "nvjpeg_seconds"):
                encoded = encode_jpeg(images, quality=100)
            with measure(timings, "readback_seconds"):
                files = [image.cpu().numpy().tobytes() for image in encoded]
            with measure(timings, "write_seconds"):
                for index, data in enumerate(files, start=first + 1):
                    (frames / f"{index:06d}.jpg").write_bytes(data)
        else:
            for index, image in enumerate(images, start=first + 1):
                with measure(timings, "readback_seconds"):
                    image = image.permute(1, 2, 0).contiguous().cpu().numpy()
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                with measure(timings, "png_write_seconds"):
                    write_png(frames / f"{index:06d}.png", bgr)
    return len(decoder), metadata.average_fps


def split_video(source: Path, directory: Path, backend: str) -> dict:
    """把指定拆帧后端的输出保存到独立帧目录。"""
    started = time.monotonic()
    frames = directory / "frames"
    frames.mkdir(parents=True)
    timings = defaultdict(float)
    suffix = ".jpg" if backend == "nvdec-jpeg" else ".png"
    if backend == "cpu-png":
        count, fps = split_cpu_png(source, frames, timings)
    else:
        count, fps = split_nvdec(source, frames, timings, suffix)
    return {
        "source": source.name,
        "directory": directory.name,
        "frame_count": count,
        "fps": fps,
        "suffix": suffix,
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
        "timings": dict(timings),
    }


def detect_area(
    model, overlap: Overlap, frames: Path, output: Path, count: int, suffix: str
) -> dict:
    """逐帧保存柜区框、置信度及命中帧的掩码。"""
    from app.detection.area_detection import detect_frame_mask_with_confidence

    timings = defaultdict(float)
    areas = []
    output.mkdir()
    for frame in range(1, count + 1):
        with measure(timings, "image_read_seconds"):
            image = read_frame(frames, frame, suffix)
        with measure(timings, "model_seconds"), overlap.call():
            mask, box, confidence = detect_frame_mask_with_confidence(image, model)
        areas.append(box)
        with measure(timings, "write_seconds"):
            mask_name = None
            if mask is not None:
                mask_name = f"{frame:06d}-mask.png"
                write_png(output / mask_name, mask.astype("uint8") * 255)
            write_json(
                output / f"{frame:06d}.json",
                {
                    "frame": frame,
                    "box": box,
                    "confidence": confidence,
                    "mask": mask_name,
                },
            )
    write_json(output / "areas.json", areas)
    return {"frames_with_area": sum(box is not None for box in areas), "timings": dict(timings)}


def detect_hoi(
    detector, overlap: Overlap, frames: Path, output: Path, areas: list, suffix: str
) -> dict:
    """复用 TensorRT 单帧推理，并按连续柜区漏检规则跳过部分帧。"""
    from detector import hoi_input

    timings = defaultdict(float)
    output.mkdir()
    last_area = None
    missing_limit = config.hoi_area_missing_max_frames()
    inferred = hand_boxes = product_boxes = 0
    for frame, area in enumerate(areas, start=1):
        if area is not None:
            last_area = frame
        should_infer = last_area is not None and frame - last_area <= missing_limit
        hands, products = [], []
        if should_infer:
            with measure(timings, "image_read_and_prepare_seconds"):
                tensor = hoi_input(read_frame(frames, frame, suffix))
            with measure(timings, "model_seconds"), overlap.call():
                hands, products = detector.decode(detector.hoi.infer(tensor))
            inferred += 1
            hand_boxes += len(hands)
            product_boxes += len(products)
        with measure(timings, "write_seconds"):
            write_json(
                output / f"{frame:06d}.json",
                {
                    "frame": frame,
                    "area": area,
                    "inferred": should_infer,
                    "hands": hands,
                    "products": products,
                },
            )
    return {
        "frames_inferred": inferred,
        "frames_skipped": len(areas) - inferred,
        "hand_boxes": hand_boxes,
        "product_boxes": product_boxes,
        "timings": dict(timings),
    }


def run_detection(stage: str, entries: list, root: Path, jobs: int, area_jobs: int) -> dict:
    """每路独占模型实例，整段视频的帧序和产物分别保存。"""
    overlap = Overlap()

    def run_lane(videos: list) -> list:
        if stage == "area":
            from app.detection.area_detection import load_model

            model = load_model()
        else:
            from detector import Detector

            model = Detector(Path("/models"))
        results = []
        try:
            if stage == "hoi":
                model.warmup()
            for entry in videos:
                directory = root / entry["directory"]
                frames = directory / "frames"
                output = directory / f"{stage}-{jobs}"
                suffix = entry.get("suffix", ".png")
                started = time.monotonic()
                if stage == "area":
                    result = detect_area(
                        model, overlap, frames, output, entry["frame_count"], suffix
                    )
                else:
                    areas = read_json(directory / f"area-{area_jobs}" / "areas.json")
                    if len(areas) != entry["frame_count"]:
                        raise ValueError(f"柜区结果帧数不匹配 {directory}")
                    result = detect_hoi(model, overlap, frames, output, areas, suffix)
                    result["cuda_stream"] = int(model.hoi.stream)
                finished = time.monotonic()
                result.update(
                    source=entry["source"],
                    wall_seconds=finished - started,
                    started_monotonic=started,
                    finished_monotonic=finished,
                )
                results.append(result)
                print(
                    f"{stage} 完成 {entry['source']}，耗时 {result['wall_seconds']:.2f} 秒",
                    flush=True,
                )
        finally:
            if stage == "hoi":
                model.close()
        return results

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        reports = pool.map(run_lane, (entries[index::jobs] for index in range(jobs)))
        return {
            "videos": [result for report in reports for result in report],
            **overlap.report(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("split", "area", "hoi"))
    parser.add_argument("output", type=Path, help="两种容器共同挂载的实验目录")
    parser.add_argument("videos", nargs="*", type=Path, help="拆帧时指定 3 或 5 个视频")
    parser.add_argument("--jobs", type=int, choices=JOB_COUNTS, default=3)
    parser.add_argument(
        "--frame-backend",
        choices=("nvdec-jpeg", "nvdec-png", "cpu-png"),
        default="nvdec-jpeg",
        help="拆帧后端，默认用 NVDEC 解码及 GPU JPEG 编码；需要无损时使用 nvdec-png",
    )
    parser.add_argument(
        "--area-jobs",
        type=int,
        choices=JOB_COUNTS,
        default=3,
        help="HOI 读取哪一档柜区结果，默认复用 3 路预处理结果",
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    started = time.monotonic()
    if args.stage == "split":
        sources = [path.resolve(strict=True) for path in args.videos]
        if len(sources) not in (3, 5) or len(set(sources)) != len(sources):
            parser.error("拆帧需要 3 或 5 个不同的视频")
        if args.jobs > len(sources):
            parser.error("并发数不能大于视频数量")
        if args.frame_backend != "cpu-png":
            sys.path.insert(0, str(config.GPU_RUNTIME / "nvcodec"))
        args.output.mkdir()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(
                    split_video, source, args.output / f"video-{index:02d}", args.frame_backend
                )
                for index, source in enumerate(sources, start=1)
            ]
            result = {
                "backend": args.frame_backend,
                "videos": [future.result() for future in futures],
            }
        report = args.output / "split.json"
    else:
        if args.videos:
            parser.error("检测阶段从拆帧清单读取视频")
        entries = read_json(args.output / "split.json")["videos"]
        if args.jobs > len(entries):
            parser.error("并发数不能大于视频数量")
        report = args.output / f"{args.stage}-{args.jobs}.json"
        result = run_detection(args.stage, entries, args.output, args.jobs, args.area_jobs)
    result.update(jobs=args.jobs, wall_seconds=time.monotonic() - started)
    write_json(report, result)
    print(f"{args.stage} 完成，整批耗时 {result['wall_seconds']:.2f} 秒，报告 {report}")


if __name__ == "__main__":
    main()
