"""为离线逐帧流水线提供可校验、可回退的 NVDEC JPEG 帧写入。"""

import hashlib
import io
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import cv2
from PIL import Image

from app import config
from scripts import frame_pipeline
from scripts.parallel_video_detection import write_json

CODEC_LOCK = threading.Lock()


def check_jpeg(data: bytes, size: tuple[int, int]) -> None:
    """解码整张 JPEG，确保标记、尺寸和压缩数据可用。"""
    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("JPEG 标记不完整")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != size:
            raise ValueError(f"JPEG 尺寸错误 {image.size}")


def verify_or_reencode(image, data: bytes, gpu_encode, stream=None):
    """GPU 结果损坏时重试一次，再从独立的原始帧做 CPU 编码。"""
    import torch

    height, width = image.shape[1:]
    try:
        check_jpeg(data, (width, height))
        return data, None
    except (OSError, ValueError) as error:
        original_error = str(error)
    stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with CODEC_LOCK, stream_context:
        encoded = gpu_encode(image, quality=100)
        torch.cuda.synchronize()
        retry = encoded.cpu().numpy().tobytes()
    try:
        check_jpeg(retry, (width, height))
        return retry, {"path": "gpu_retry", "first_error": original_error}
    except (OSError, ValueError) as error:
        retry_error = str(error)
    # 独立复制的 RGB 帧仍然有效，CPU JPEG 是 GPU 编码的最后回退。
    with CODEC_LOCK:
        rgb = image.permute(1, 2, 0).contiguous().cpu().numpy()
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    if not ok:
        raise RuntimeError("CPU JPEG 编码失败")
    fallback = encoded.tobytes()
    check_jpeg(fallback, (width, height))
    return fallback, {
        "path": "cpu_fallback",
        "first_error": original_error,
        "retry_error": retry_error,
    }


def split_one(source: Path, video: Path, broker, gate_enabled: bool) -> None:
    """独立保留 NVDEC 帧，校验 JPEG 后发布原流水线任务。"""
    import PyNvVideoCodec as nvc
    import torch
    from torchvision.io import encode_jpeg

    decoder = nvc.SimpleDecoder(
        str(source), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
    )
    stream = torch.cuda.Stream()
    metadata = decoder.get_stream_metadata()
    count = len(decoder)
    if (metadata.height, metadata.width) != (1080, 1920) or count <= 0:
        raise ValueError(f"视频尺寸或帧数无效 {source}")
    write_json(video / "metadata.json", {"frame_count": count})
    recoveries = []
    for first in range(0, count, 8):
        length = min(8, count - first)
        with CODEC_LOCK, torch.cuda.stream(stream):
            decoded = decoder.get_batch_frames(length)
            if len(decoded) != length:
                raise ValueError(f"视频解码提前结束 {source}")
            # 复制到独立显存，直到该批帧校验和必要的重编码结束。
            images = [torch.from_dlpack(frame).clone() for frame in decoded]
            torch.cuda.synchronize()
            encoded = encode_jpeg(images, quality=100)
            # nvJPEG 在其他 GPU 任务并行时可能尚未完成 bitstream 写入。
            torch.cuda.synchronize()
            jpeg_bytes = [item.cpu().numpy().tobytes() for item in encoded]
        for index, (image, data) in enumerate(zip(images, jpeg_bytes), start=first + 1):
            verified, recovery = verify_or_reencode(image, data, encode_jpeg, stream)
            if recovery:
                recovery.update(frame=index, invalid_sha256=hashlib.sha256(data).hexdigest())
                recoveries.append(recovery)
                print(f"{source.name} 第 {index} 帧编码异常，采用 {recovery['path']}", flush=True)
            path = video / "frames" / f"{index:06d}.jpg"
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(verified)
            temporary.replace(path)
            write_json(
                video / "frames" / f"{index:06d}.json", {"video": video.name, "frame": index}
            )
            broker.add("area", video.name, index)
            if gate_enabled and (index - 1) % 5 == 0:
                broker.add("gate", video.name, index)
    write_json(video / "encoding-validation.json", {"frames": count, "recoveries": recoveries})
    write_json(
        video / "split.json",
        {
            "video": video.name,
            "source": source.name,
            "frame_count": count,
            "fps": metadata.average_fps,
            "suffix": ".jpg",
        },
    )


def main() -> None:
    sys.path.insert(0, str(config.GPU_RUNTIME / "nvcodec"))
    frame_pipeline.split_one = split_one
    frame_pipeline.main()


if __name__ == "__main__":
    main()
