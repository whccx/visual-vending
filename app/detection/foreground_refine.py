"""使用本机 BiRefNet 修补 YOLOE 自动主体的附属区域。"""
import io
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from app.restoration import image_restore


def refine(rgba):
    """调用本机背景分割模型，返回映射回原图的二值前景掩码。"""
    height, width = rgba.shape[:2]
    image = Image.fromarray(rgba[:, :, :3], mode="RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    token = "sku_cutout_" + uuid.uuid4().hex
    filename = token + ".png"
    output_name = None
    backend = image_restore.BACKEND
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": filename}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": token}},
    }
    with image_restore._lock:
        try:
            upload = requests.post(backend + "/upload/image", files={"image": (filename, buffer.getvalue(), "image/png")}, data={"type": "input", "overwrite": "false"}, timeout=30)
            upload.raise_for_status()
            if upload.json().get("name") != filename:
                raise RuntimeError("前景分割上传文件名异常")
            response = requests.post(backend + "/prompt", json={"prompt": workflow}, timeout=30)
            response.raise_for_status()
            prompt_id = response.json()["prompt_id"]
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                history = requests.get(backend + "/history/" + prompt_id, timeout=10)
                history.raise_for_status()
                job = history.json().get(prompt_id)
                if not job:
                    time.sleep(0.25)
                    continue
                if job.get("status", {}).get("status_str") != "success":
                    raise RuntimeError("前景分割失败")
                output = job["outputs"]["6"]["images"][0]
                output_name = output["filename"]
                if Path(output_name).name != output_name or not output_name.startswith(token + "_"):
                    raise RuntimeError("前景分割文件名异常")
                result = requests.get(backend + "/view", params=output, timeout=30)
                result.raise_for_status()
                with Image.open(io.BytesIO(result.content)) as result_image:
                    alpha = result_image.convert("RGBA").getchannel("A")
                    return np.asarray(alpha.resize((width, height), Image.Resampling.BILINEAR)) >= 128
            raise TimeoutError("前景分割超时")
        finally:
            (image_restore.DATA_DIR / "input" / filename).unlink(missing_ok=True)
            if output_name:
                (image_restore.DATA_DIR / "output" / output_name).unlink(missing_ok=True)
