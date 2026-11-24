"""在真实模型 GPU 并发负载下复现并验证 NVDEC/nvJPEG 同步。"""

import argparse
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import cv2
from PIL import Image

from app import config


def main():
    import mediapipe as mp
    import PyNvVideoCodec as nvc
    import torch
    from torchvision.io import encode_jpeg

    from app.detection.area_detection import load_model
    from app.detection.held_product_detect import load_gpu_hands

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "original",
            "post_sync",
            "single_stream",
            "default_pre",
            "default_post",
            "verified",
        ),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    codec_lock = threading.Lock()
    stop = threading.Event()
    shared = torch.cuda.Stream()
    image = cv2.imread(str(args.reference))
    if image is None:
        raise ValueError("缺少 reference.jpg")

    def area_load():
        model = load_model()
        count = 0
        while not stop.is_set():
            model.predict(image, imgsz=640, device=0, verbose=False)
            count += 1
        return count

    def hand_load():
        model = load_gpu_hands()
        count = 0
        try:
            small = cv2.resize(image, (960, 540))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            while not stop.is_set():
                model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                count += 1
        finally:
            model.close()
        return count

    def encode_video(source):
        if args.mode == "verified":
            from scripts.verified_frame_pipeline import split_one

            class Broker:
                def add(self, *args):
                    pass

            bad = []
            count = 0
            recoveries = []
            for iteration in range(args.repeat):
                directory = args.output / f"{source.stem}-{iteration + 1}"
                (directory / "frames").mkdir(parents=True)
                split_one(source, directory, Broker(), False)
                record = json.loads((directory / "encoding-validation.json").read_text())
                recoveries.extend(record["recoveries"])
                for path in sorted((directory / "frames").glob("*.jpg")):
                    count += 1
                    try:
                        with Image.open(path) as image:
                            image.load()
                            if image.size != (1920, 1080):
                                raise ValueError("图像尺寸不符")
                    except (OSError, ValueError) as exc:
                        bad.append({"frame": path.name, "error": str(exc)})
            return {"video": source.name, "frames": count, "bad": bad, "recoveries": recoveries}
        decoder = nvc.SimpleDecoder(
            str(source), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
        )
        total = len(decoder)
        bad = []
        count = 0
        stream = shared if args.mode == "single_stream" else torch.cuda.Stream()
        for iteration in range(args.repeat):
            if iteration:
                decoder = nvc.SimpleDecoder(
                    str(source), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
                )
            for first in range(0, total, 8):
                length = min(8, total - first)
                stream_context = (
                    nullcontext() if args.mode.startswith("default_") else torch.cuda.stream(stream)
                )
                with codec_lock, stream_context:
                    decoded = decoder.get_batch_frames(length)
                    frames = [torch.from_dlpack(item).clone() for item in decoded]
                    torch.cuda.synchronize()
                    encoded = encode_jpeg(frames, quality=100)
                    if args.mode in ("post_sync", "default_post"):
                        torch.cuda.synchronize()
                    payloads = [item.cpu().numpy().tobytes() for item in encoded]
                for offset, data in enumerate(payloads, 1):
                    count += 1
                    try:
                        if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                            raise ValueError("JPEG 标记不完整")
                        with Image.open(io.BytesIO(data)) as result:
                            result.load()
                            if result.size != (1920, 1080):
                                raise ValueError("图像尺寸不符")
                    except (OSError, ValueError) as exc:
                        row = {"repeat": iteration + 1, "frame": first + offset, "error": str(exc)}
                        bad.append(row)
                        if len(bad) <= 10:
                            (
                                args.output / f"{source.stem}-{iteration}-{first + offset}.jpg"
                            ).write_bytes(data)
                        print(args.mode, source.name, row, flush=True)
        return {"video": source.name, "frames": count, "bad": bad, "stream": stream.cuda_stream}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(args.videos) + 4) as pool:
        loads = [pool.submit(area_load) for _ in range(3)]
        loads.append(pool.submit(hand_load))
        try:
            results = list(pool.map(encode_video, args.videos))
        finally:
            stop.set()
        counts = [item.result() for item in loads]
    report = {
        "mode": args.mode,
        "seconds": time.monotonic() - started,
        "load_counts": counts,
        "videos": results,
        "bad_total": sum(len(item["bad"]) for item in results),
        "recovery_total": sum(len(item.get("recoveries", [])) for item in results),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        args.mode,
        "帧数",
        sum(item["frames"] for item in results),
        "损坏",
        report["bad_total"],
        "模型处理次数",
        counts,
        flush=True,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(config.GPU_RUNTIME / "nvcodec"))
    main()
