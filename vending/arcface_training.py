"""使用有标注的图片和库外样本，离线生成 ViT-7B ArcFace 候选权重。"""

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from app import config

SIZE = 224
DTYPE = torch.bfloat16
SEED = 119
DEFAULT_STEPS = 110
PATCH_NAME = "arcface_patch.safetensors"
RESULT_NAME = "training_result.json"
FORMATS = {"JPEG", "PNG", "WEBP"}
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


@dataclass
class Dataset:
    references: dict[str, list[np.ndarray]]
    observations: dict[str, list[np.ndarray]]
    negatives: dict[str, list[np.ndarray]]
    manifest_hash: str
    sample_hash: str


def _sample_path(directory: Path, entry: dict) -> Path:
    name = entry.get("path")
    if not isinstance(name, str) or not name:
        raise ValueError("训练样本需要相对图片路径")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in SUFFIXES:
        raise ValueError("训练样本路径必须指向清单目录下的 JPEG、PNG 或 WebP 图片")
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("训练样本必须是清单目录中的文件")
    return path


def _reference_pixels(image: Image.Image) -> list[np.ndarray]:
    from app.catalog.product_catalog import image_variants

    values = np.asarray(image)
    prepared = []
    for _, variant in image_variants(values):
        rgb = variant[:, :, :3]
        if variant.shape[2] == 4:
            alpha = variant[:, :, 3:4].astype(np.float32) / 255
            rgb = (rgb.astype(np.float32) * alpha).astype(np.uint8)
        prepared.append(cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
    return prepared


def _pixels(data: bytes, entry: dict) -> list[np.ndarray]:
    from app.recognition.hoi_mask_matching import _enhance_crop

    flags = [entry.get(key, False) for key in ("reference", "preprocessed")]
    if not all(isinstance(flag, bool) for flag in flags) or all(flags):
        raise ValueError("reference 和 preprocessed 只能择一使用，且必须为布尔值")
    reference, preprocessed = flags
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in FORMATS or getattr(source, "is_animated", False):
            raise ValueError("训练样本必须是单帧 JPEG、PNG 或 WebP 图片")
        if preprocessed:
            if source.mode != "RGB" or source.size != (SIZE, SIZE):
                raise ValueError("预处理训练样本必须为 224×224 RGB 图片")
            return [np.array(source)]
        has_alpha = "A" in source.getbands() or "transparency" in source.info
        image = source.convert("RGBA" if has_alpha else "RGB")
        if reference:
            return _reference_pixels(image)
        rgba = np.asarray(image)
        rgb = _enhance_crop(rgba[:, :, :3])
        if has_alpha:
            mask = rgba[:, :, 3] > 0
            if not mask.any():
                raise ValueError("训练裁剪没有可见像素")
            rgb *= mask[:, :, None]
    pixels = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    resized = F.interpolate(pixels, (SIZE, SIZE), mode="bilinear", align_corners=False)
    return [resized[0].round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()]


def load_dataset(manifest_path: Path) -> Dataset:
    """检查输入清单、图片位置与标签，准备 CPU 上的训练图像。"""
    manifest_path = manifest_path.resolve(strict=True)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if not isinstance(manifest, dict):
        raise TypeError("训练清单必须为 JSON 对象")
    positives, negatives = manifest.get("positives"), manifest.get("negatives")
    if not isinstance(positives, list) or not isinstance(negatives, list) or not negatives:
        raise ValueError("训练清单须提供正样本列表及至少一个负样本")
    if any(not isinstance(entry, dict) for entry in [*positives, *negatives]):
        raise ValueError("正负样本均须以对象提供")
    classes = [entry.get("sku_id") for entry in positives]
    if any(not isinstance(label, str) or not label.strip() for label in classes):
        raise ValueError("正样本须提供有效的 sku_id")
    if len(set(classes)) < 2:
        raise ValueError("至少需要两个不同 SKU 的正样本")

    references, observations, unknown = defaultdict(list), defaultdict(list), defaultdict(list)
    sample_hash = hashlib.sha256()
    identities = {}
    for entry, label in [*zip(positives, classes), *((entry, None) for entry in negatives)]:
        path = _sample_path(manifest_path.parent, entry)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).digest()
        sample_hash.update(digest)
        if digest in identities and identities[digest] != label:
            raise ValueError("同一张样本图片出现在不同 SKU 或正负样本中")
        identities[digest] = label
        prepared = _pixels(raw, entry)
        if label is not None:
            group = references if entry.get("reference", False) else observations
            group[label].extend(prepared)
        else:
            source_group = entry.get("source_group", "request")
            if not isinstance(source_group, str) or not source_group.strip():
                raise ValueError("负样本来源分组必须为非空字符串")
            unknown[source_group].extend(prepared)
    return Dataset(
        dict(references),
        dict(observations),
        dict(unknown),
        hashlib.sha256(raw_manifest).hexdigest(),
        sample_hash.hexdigest(),
    )


def _augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    channel = rng.uniform(0.82, 1.18, size=(1, 1, 3))
    output = np.clip(rgb * channel * rng.uniform(0.78, 1.22), 0, 255).astype(np.uint8)
    if rng.random() < 0.22:
        output = cv2.GaussianBlur(output, (3, 3), rng.uniform(0.35, 1.4))
    return output


def _batches(data: Dataset, labels: list[str], steps: int) -> Iterator[tuple[list, list]]:
    rng = np.random.default_rng(SEED)
    anchors = [data.references.get(label) or data.observations[label] for label in labels]
    observations = [data.observations.get(label, []) for label in labels]
    label_indices = {label: index for index, label in enumerate(labels)}
    observation_classes = [label_indices[label] for label in data.observations]
    unknown = list(data.negatives.values())

    def sample(group):
        return _augment(group[int(rng.integers(len(group)))], rng)

    for _ in range(steps):
        slots = rng.choice(len(labels), size=6, replace=len(labels) < 6).tolist()
        if observation_classes:
            slots.extend(int(rng.choice(observation_classes)) for _ in range(2))
        else:
            slots.extend(rng.integers(len(labels), size=2).tolist())
        first = [sample(anchors[index]) for index in slots]
        second = [
            sample(
                observations[index]
                if observations[index] and rng.random() < 0.7
                else anchors[index]
            )
            for index in slots
        ]
        negatives = [sample(unknown[index]) for index in rng.integers(len(unknown), size=8)]
        yield first + second + negatives, slots * 2


def _normalize(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    pixels = torch.from_numpy(np.stack(images)).to(device).permute(0, 3, 1, 2).float()
    mean = pixels.new_tensor(RGB_MEAN)[None, :, None, None]
    std = pixels.new_tensor(RGB_STD)[None, :, None, None]
    return ((pixels / 255 - mean) / std).to(DTYPE)


def _embedding(model, pixels) -> torch.Tensor:
    return F.normalize(model(pixel_values=pixels).pooler_output.float(), dim=-1)


@torch.no_grad()
def _centers(model, data: Dataset, labels: list[str], device: torch.device) -> torch.Tensor:
    centers = []
    for label in labels:
        images = data.references.get(label) or data.observations[label]
        center = None
        count = 0
        for offset in range(0, len(images), 8):
            for reference in _embedding(model, _normalize(images[offset : offset + 8], device)):
                center = (
                    reference
                    if center is None
                    else F.normalize((center * count + reference) / (count + 1), dim=0)
                )
                count += 1
        centers.append(center)
    return torch.stack(centers)


def _project(layers, norm, hidden, positions: tuple) -> torch.Tensor:
    for layer in layers:
        hidden = layer(hidden, position_embeddings=positions)
    return F.normalize(norm(hidden)[:, 0, :].float(), dim=-1)


@torch.no_grad()
def _cache_prefix(
    model, batches, directory: Path, device: torch.device
) -> tuple[list, tuple, float]:
    """分批计算冻结前缀并将激活暂存到磁盘，避免保留前缀梯度。"""
    position_input = torch.zeros((1, 3, SIZE, SIZE), device=device, dtype=DTYPE)
    positions = model.rope_embeddings(position_input)
    prefix, tail = model.layer[:-2], model.layer[-2:]
    split_error = None
    targets = []
    for step, (images, labels) in enumerate(batches):
        hidden_batches = []
        for offset in range(0, len(images), 8):
            inputs = _normalize(images[offset : offset + 8], device)
            hidden = model.embeddings(inputs)
            for layer in prefix:
                hidden = layer(hidden, position_embeddings=positions)
            if split_error is None:
                whole = _embedding(model, inputs)
                split = _project(tail, model.norm, hidden, positions)
                split_error = float((whole - split).abs().max())
                if not math.isfinite(split_error) or split_error > 1e-5:
                    raise ValueError("训练与识别模型的前向计算不一致")
            hidden_batches.append(hidden.cpu())
        save_file({"hidden": torch.cat(hidden_batches)}, directory / f"{step:03d}.safetensors")
        targets.append(labels)
        if step % 20 == 0:
            print(f"已缓存第 {step + 1} 步的冻结层激活", flush=True)
    return targets, positions, split_error


def _loss(vectors, weights, old_centers, targets) -> dict:
    positives, unknowns = vectors.split((len(targets), vectors.shape[0] - len(targets)))
    centers = F.normalize(weights, dim=-1)
    similarities = (positives @ centers.T).clamp(-0.9999, 0.9999)
    target_index = targets[:, None]
    target = similarities.gather(1, target_index)
    margin = target * math.cos(0.22) - (1 - target.square()).sqrt() * math.sin(0.22)
    logits = similarities.scatter(1, target_index, margin) * 24
    first, second = positives.chunk(2, dim=0)
    return {
        "arc": F.cross_entropy(logits, targets),
        "same": 1 - (first * second).sum(1).mean(),
        "unknown_center": F.relu((unknowns @ centers.T).amax(1) - 0.18).square().mean(),
        "unknown_refs": F.relu((unknowns @ positives.T).amax(1) - 0.24).square().mean(),
        "anchor": 1 - (centers * old_centers).sum(1).mean(),
    }


def _fit(tail, norm, centers, labels, positions, directory, device) -> list[dict]:
    trainable = nn.ModuleList([*tail, norm]).float().requires_grad_(True)
    parameters = list(trainable.parameters())
    weights = nn.Parameter(centers.clone())
    optimizer = torch.optim.AdamW(
        [{"params": parameters, "lr": 1.2e-5}, {"params": [weights], "lr": 1.7e-3}],
        weight_decay=0.01,
        fused=True,
    )
    centers = F.normalize(centers, dim=1)
    history = []
    for step, labels_at_step in enumerate(labels, 1):
        targets = torch.tensor(labels_at_step, device=device)
        hidden_file = directory / f"{step - 1:03d}.safetensors"
        hidden = load_file(hidden_file, device=str(device))["hidden"]
        with torch.autocast("cuda", dtype=DTYPE):
            vectors = _project(tail, norm, hidden, positions)
        values = _loss(vectors, weights, centers, targets)
        loss = (
            values["arc"]
            + 1.25 * values["same"]
            + 8 * values["unknown_center"]
            + 4 * values["unknown_refs"]
            + 0.5 * values["anchor"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*parameters, weights], 2.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == len(labels):
            record = {"step": step, "loss": float(loss.detach())}
            record.update({name: float(value.detach()) for name, value in values.items()})
            history.append(record)
            print(f"已训练 {step}/{len(labels)} 步，损失 {record['loss']:.5f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    trainable.requires_grad_(False).to(dtype=DTYPE, device="cpu")
    return history


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(manifest: Path, output: Path, steps: int = DEFAULT_STEPS) -> dict:
    """训练末两层并原子写入可供离线验证的候选权重。"""
    from transformers import AutoModel
    from transformers.utils.logging import disable_progress_bar

    started = perf_counter()
    if not 1 <= steps <= DEFAULT_STEPS:
        raise ValueError(f"训练步数必须在 1 到 {DEFAULT_STEPS} 之间")
    torch.set_num_threads(3)
    data = load_dataset(manifest)
    labels = list(dict.fromkeys((*data.references, *data.observations)))
    base = config.ARCFACE_MODEL
    base_config_hash = _sha256(base / "config.json")
    base_manifest = base / "arcface-manifest.json"
    identity = {
        "steps": steps,
        "dataset_sha256": data.manifest_hash,
        "dataset_samples_sha256": data.sample_hash,
        "base_model": config.ARCFACE_MODEL_ID,
        "base_config_sha256": base_config_hash,
        "base_manifest_sha256": _sha256(base_manifest) if base_manifest.is_file() else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path, patch_path = output / RESULT_NAME, output / PATCH_NAME
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        if all(existing.get(key) == value for key, value in identity.items()) and (
            patch_path.is_file() and existing.get("patch_sha256") == _sha256(patch_path)
        ):
            return existing
        raise ValueError("输出目录已经包含与本次输入不一致的训练结果")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ArcFace 微调需要支持 BF16 的 CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    disable_progress_bar()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModel.from_pretrained(
            base,
            local_files_only=True,
            torch_dtype=DTYPE,
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    if model.config.hidden_size != config.ARCFACE_DIMS or len(model.layer) != 40:
        raise ValueError("训练骨干必须为当前的 ViT-7B ArcFace 架构")
    centers = _centers(model, data, labels, device)
    with tempfile.TemporaryDirectory(prefix=".arcface-", dir=output) as work:
        directory = Path(work)
        targets, positions, split_error = _cache_prefix(
            model, _batches(data, labels, steps), directory, device
        )
        tail, norm = model.layer[-2:], model.norm
        prefix_count = len(model.layer) - len(tail)
        del model
        torch.cuda.empty_cache()
        history = _fit(tail, norm, centers, targets, positions, directory, device)
        patch = {
            f"layer.{prefix_count + index}.{name}": tensor.contiguous()
            for index, layer in enumerate(tail)
            for name, tensor in layer.state_dict().items()
        }
        patch.update(
            {f"norm.{name}": tensor.contiguous() for name, tensor in norm.state_dict().items()}
        )
        staged_patch = directory / PATCH_NAME
        save_file(
            patch,
            staged_patch,
            metadata={"base_model": identity["base_model"], "dtype": "bfloat16"},
        )
        result = {
            **identity,
            "artifact": PATCH_NAME,
            "patch_sha256": _sha256(staged_patch),
            "sku_count": len(labels),
            "reference_variants": sum(map(len, data.references.values())),
            "positive_crops": sum(map(len, data.observations.values())),
            "negative_crops": sum(map(len, data.negatives.values())),
            "updated_parameters": sum(tensor.numel() for tensor in patch.values()),
            "split_forward_max_error": split_error,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "history": history,
        }
        staged_result = directory / RESULT_NAME
        staged_result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        staged_patch.replace(patch_path)
        staged_result.replace(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    steps = int(os.environ.get("VENDING_TRAINING_STEPS", DEFAULT_STEPS))
    result = train(args.manifest, args.output_dir, steps)
    print(f"候选权重已准备，更新参数 {result['updated_parameters']:,} 个", flush=True)


if __name__ == "__main__":
    main()
