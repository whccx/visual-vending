"""比较 NVDEC 解码后的多种 JPEG 编码方式，逐帧验证完整性。"""

import argparse
import hashlib
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image

from app import config

GPU_LOCK = threading.Lock()


def validate(data):
    """完整解码编码数据，检查尺寸和 JPEG 结束标记。"""
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("JPEG 标记不完整")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != (1920, 1080):
            raise ValueError(f"JPEG 尺寸异常 {image.size}")


def encode_video(source, root, repeat):
    import PyNvVideoCodec as nvc
    import torch
    from torchvision.io import encode_jpeg

    video_dir = root / source.stem
    video_dir.mkdir()
    modes = ("gpu_batch", "gpu_single", "cpu_opencv")
    for mode in modes:
        (video_dir / mode).mkdir()
    decoder = nvc.SimpleDecoder(
        str(source), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
    )
    metadata = decoder.get_stream_metadata()
    if (metadata.width, metadata.height) != (1920, 1080):
        raise ValueError(f"尺寸异常 {source}")
    count = len(decoder)
    report = {
        "video": source.name,
        "repeat": repeat,
        "frames": count,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "modes": {mode: {"validated": 0, "failures": []} for mode in modes},
    }
    started = time.monotonic()
    for first in range(0, count, 8):
        number = min(8, count - first)
        with GPU_LOCK:
            decoded = decoder.get_batch_frames(number)
            images = [torch.from_dlpack(frame) for frame in decoded]
            if len(images) != number:
                raise ValueError(f"解码不足 {source.name} 第 {first + 1} 帧")
            torch.cuda.synchronize()
            batch = [item.cpu().numpy().tobytes() for item in encode_jpeg(images, quality=100)]
            single = [encode_jpeg(item, quality=100).cpu().numpy().tobytes() for item in images]
            raw = [item.permute(1, 2, 0).contiguous().cpu().numpy() for item in images]
        for offset, (batch_data, single_data, rgb) in enumerate(zip(batch, single, raw), start=1):
            frame = first + offset
            ok, cpu_image = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            if not ok:
                raise RuntimeError(f"CPU 编码失败 {source.name} 第 {frame} 帧")
            for mode, data in zip(modes, (batch_data, single_data, cpu_image.tobytes())):
                path = video_dir / mode / f"{frame:06d}.jpg"
                path.write_bytes(data)
                try:
                    validate(path.read_bytes())
                    report["modes"][mode]["validated"] += 1
                except (OSError, ValueError) as exc:
                    failure = {
                        "frame": frame,
                        "error": str(exc),
                        "file": str(path),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                    report["modes"][mode]["failures"].append(failure)
                    print(f"{source.name} {mode} 第 {frame} 帧异常 {exc}", flush=True)
    report["seconds"] = time.monotonic() - started
    print(f"{source.name} 已校验 {count} 帧", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 1 or args.workers < 1:
        parser.error("重复次数和并发数必须大于零")
    sys.path.insert(0, str(config.GPU_RUNTIME / "nvcodec"))
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for iteration in range(1, args.repeat + 1):
        repeat_dir = args.output / f"run-{iteration:02d}"
        repeat_dir.mkdir()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(encode_video, video, repeat_dir, iteration) for video in args.videos
            ]
            for future in futures:
                reports.append(future.result())
        (args.output / "report.json").write_text(
            json.dumps({"reports": reports}, ensure_ascii=False, indent=2) + "\n"
        )
    print("编码验证完成", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
