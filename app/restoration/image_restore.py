"""通过独立的 ComfyUI 后端在 GPU 上修复商品图片。"""

import io
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

# 在 Ultralytics 启动时替换 Image.open 之前保留 Pillow 原始解码器。
_PIL_OPEN = Image.open

from app import config

BACKEND = config.restore_backend()
DATA_DIR = config.restore_data_dir()
LOG = logging.getLogger("sku." + __name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
TIMEOUT = 180
_lock = threading.Lock()
PROMPT = (
    "Remove only the hand and background from the photo of the product. "
    "Keep the already visible parts of the product exactly as they appear in the input, "
    "including their original shape, angle, colors, texture, printed text, logos and details. "
    "Do not enhance, sharpen, redraw, restyle, invent or replace any visible product details. "
    "Only fill in the areas hidden by the hand, matching the adjacent visible product. "
    "Output the complete product alone on a plain white background."
)


def _workflow(filename, prefix):
    """构造商品图修复工作流，使用指定输入文件名和输出前缀。

    返回：
        可提交给后端的节点字典；采样种子在每次调用时随机生成。
    """
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "qwen-image-2.1-Q4_K_M.gguf"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
        },
        "4": {"class_type": "LoadImage", "inputs": {"image": filename}},
        "5": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": PROMPT,
                "negative_prompt": "",
                "resolution": 384,
                "images.image_1": ["4", 0],
            },
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "seed": int.from_bytes(os.urandom(4), "big"),
                "steps": 16,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["5", 2],
                "denoise": 1.0,
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "9": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": "birefnet.safetensors"},
        },
        "10": {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["9", 0], "image": ["7", 0]},
        },
        "12": {"class_type": "InvertMask", "inputs": {"mask": ["10", 0]}},
        "11": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["7", 0], "alpha": ["12", 0]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["11", 0], "filename_prefix": prefix},
        },
    }


def restore_image(data):
    """校验并缩放输入图片，调用 GPU 后端修复并返回 PNG 字节。

    参数：
        data: 上传图片的原始字节，大小不得超过 MAX_UPLOAD_BYTES。
    返回：
        修复后的 PNG 图片字节，最长边不超过预处理图片的尺寸。
    异常：
        图片无效、过大、后端失败或任务超时时抛出相应异常。
    调用：
        API 的 restore_product_image 接口调用本函数；内部通过锁串行访问后端。
    """
    request_started = time.monotonic()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("图片为空或超过 10 MB")
    try:
        with _PIL_OPEN(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("图片像素超过 2000 万")
            image = ImageOps.exif_transpose(source).convert("RGB")
            original_size = image.size
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            input_size = image.size
            buf = io.BytesIO()
            image.save(buf, format="PNG")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片，请上传有效图片") from exc

    token = "sku_restore_" + uuid.uuid4().hex
    filename = token + ".png"
    output_file = None
    lock_requested = time.monotonic()
    with _lock:
        lock_acquired = time.monotonic()
        try:
            stats = requests.get(BACKEND + "/system_stats", timeout=5)
            stats.raise_for_status()
            if not any(device.get("type") == "cuda" for device in stats.json().get("devices", [])):
                raise RuntimeError("修复服务未使用 CUDA GPU")
            upload = requests.post(
                BACKEND + "/upload/image",
                files={"image": (filename, buf.getvalue(), "image/png")},
                data={"type": "input", "overwrite": "false"},
                timeout=30,
            )
            upload.raise_for_status()
            if upload.json().get("name") != filename:
                raise RuntimeError("修复服务上传文件名异常")
            response = requests.post(
                BACKEND + "/prompt", json={"prompt": _workflow(filename, token)}, timeout=30
            )
            response.raise_for_status()
            prompt_id = response.json()["prompt_id"]
            deadline = time.monotonic() + TIMEOUT
            while time.monotonic() < deadline:
                response = requests.get(BACKEND + "/history/" + prompt_id, timeout=10)
                response.raise_for_status()
                job = response.json().get(prompt_id)
                if job:
                    if job.get("status", {}).get("status_str") != "success":
                        raise RuntimeError("图像生成失败")
                    result = job["outputs"]["8"]["images"][0]
                    output_file = result["filename"]
                    if Path(output_file).name != output_file or not output_file.startswith(
                        token + "_"
                    ):
                        raise RuntimeError("修复服务返回的文件名异常")
                    response = requests.get(
                        BACKEND + "/view",
                        params={"filename": output_file, "subfolder": "", "type": "output"},
                        timeout=30,
                    )
                    response.raise_for_status()
                    with _PIL_OPEN(io.BytesIO(response.content)) as result_image:
                        result_image.thumbnail(input_size, Image.Resampling.LANCZOS)
                        LOG.info(
                            "图片修复：原始尺寸=%sx%s 处理尺寸=%sx%s 输出尺寸=%sx%s 等待=%.3f 秒 执行=%.3f 秒 总耗时=%.3f 秒",
                            *original_size,
                            *input_size,
                            *result_image.size,
                            lock_acquired - lock_requested,
                            time.monotonic() - lock_acquired,
                            time.monotonic() - request_started,
                        )
                        out = io.BytesIO()
                        result_image.save(out, format="PNG")
                        return out.getvalue()
                time.sleep(0.5)
            raise TimeoutError("图片修复超时")
        finally:
            (DATA_DIR / "input" / filename).unlink(missing_ok=True)
            if output_file:
                (DATA_DIR / "output" / output_file).unlink(missing_ok=True)
