"""下载训练样本、执行独立训练进程并原子发布候选权重。"""

import hashlib
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from vending.config import Settings
from vending.media import IMAGE_LIMIT, MediaError, download

LOG = logging.getLogger("vending.training")
PATCH = "arcface_patch.safetensors"
TRAINING_TIMEOUT = 1200
DATASET_LIMIT = 512 * 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_for(directory: Path) -> dict:
    weights = directory / PATCH
    summary = directory / "training_result.json"
    if not summary.is_file() or not weights.is_file():
        raise ValueError("训练进程没有生成完整的候选权重")
    digest = file_sha256(weights)
    if json.loads(summary.read_text())["patch_sha256"] != digest:
        raise ValueError("训练候选权重的 SHA-256 校验失败")
    return {"state": "succeeded", "artifact": {"sha256": digest, "size": weights.stat().st_size}}


def collect_samples(config: Settings, payload: dict, directory: Path) -> Path:
    samples = directory / "samples"
    samples.mkdir()
    manifest = {"positives": [], "negatives": []}
    count = total = 0
    hashes = set()

    def save(url, sku=None):
        nonlocal count, total
        count += 1
        path = samples / f"{count:04d}.img"
        remaining = DATASET_LIMIT - total
        if remaining <= 0:
            raise MediaError("训练样本总大小超过限制")
        with path.open("wb") as stream:
            download(url, config.asset_hosts, stream, min(IMAGE_LIMIT, remaining))
            total += stream.tell()
        with path.open("rb") as stream:
            header = stream.read(12)
        digest = file_sha256(path)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            suffix = ".png"
        elif header.startswith(b"\xff\xd8\xff"):
            suffix = ".jpg"
        elif header[:4] == b"RIFF" and header[8:] == b"WEBP":
            suffix = ".webp"
        else:
            raise MediaError("训练样本必须为 PNG、JPEG 或 WebP 图片")
        if digest in hashes:
            raise MediaError("多个训练地址返回相同的图片")
        hashes.add(digest)
        path = path.rename(path.with_suffix(suffix))
        sample = {"path": str(path.relative_to(directory))}
        if sku is not None:
            sample["sku_id"] = sku
        manifest["positives" if sku is not None else "negatives"].append(sample)

    for group in payload["positive_samples"]:
        for url in group["image_urls"]:
            save(url, group["sku_id"])
    for url in payload["negative_image_urls"]:
        save(url)
    destination = directory / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return destination


def run_training(config: Settings, identifier: str, payload: dict) -> dict:
    parent = config.data_dir / "training"
    parent.mkdir(mode=0o700, exist_ok=True)
    output = parent / identifier
    if output.exists():
        return result_for(output)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".training-") as temporary:
        staging = Path(temporary)
        manifest = collect_samples(config, payload, staging)
        log_path = parent / f"{identifier}.log"
        with log_path.open("w") as log:
            subprocess.run(
                [sys.executable, "-m", "vending.arcface_training", str(manifest), str(staging)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=TRAINING_TIMEOUT,
            )
        result = result_for(staging)
        staging.rename(output)
        LOG.info("ArcFace 候选权重已生成，任务 %s", identifier)
        return result
