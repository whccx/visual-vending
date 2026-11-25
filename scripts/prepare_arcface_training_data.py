"""将已完成的 ArcFace 离线训练数据整理成可复用的数据集。"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
CABINET = "862337050531238"
SIZE = 224
STEPS = 110
SEED = 119
NEGATIVE_GROUPS = {"no-1", "no-3", "no-7", "no-11", "no-15", "no-16"}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixels_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def copy_image(destination: Path, path: str, source: Path) -> str:
    output = destination / path
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    digest = file_sha256(output)
    assert digest == file_sha256(source), source
    return digest


def save_rgb(destination: Path, path: str, pixels: np.ndarray) -> str:
    output = destination / path
    output.parent.mkdir(parents=True, exist_ok=True)
    assert pixels.shape == (SIZE, SIZE, 3) and pixels.dtype == np.uint8
    Image.fromarray(pixels).save(output)
    with Image.open(output) as image:
        assert image.mode == "RGB" and np.array_equal(np.asarray(image), pixels), output
    return file_sha256(output)


def load_experiment(path: Path):
    spec = importlib.util.spec_from_file_location("arcface_original_dataset", path / "dataset.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PROJECT == PROJECT
    return module


def capture_query_inputs(original, method, *args) -> list[dict]:
    """运行当次数据选择与预处理，跳过训练没有读取的 GPU 色彩统计。"""

    def cpu_query(video, track, frame, split, number, bgr, mask, size):
        assert mask.any(), (video, track, frame)
        return {
            "video": video,
            "track": track,
            "frame": frame,
            "split": split,
            "number": number,
            "bgr": bgr,
            "mask": mask,
            "pixels": original.query_pixels(bgr, mask, size),
        }

    query = original.query
    original.query = cpu_query
    try:
        return method(*args)
    finally:
        original.query = query


def reference_records(original, destination: Path) -> tuple[list, list, dict]:
    refs, legacy = original.product_references(SIZE)
    by_product = defaultdict(list)
    for number, label, rgb in refs:
        by_product[number, label.split(".png", 1)[0] + ".png"].append(
            {
                "suffix": label.split(".png", 1)[1],
                "rgb_sha256": pixels_sha256(rgb),
            }
        )
    samples = []
    for (number, name), variants in by_product.items():
        source = PROJECT / "tests/product" / CABINET / str(number) / name
        path = f"references/{number:02d}/{name}"
        assert len(variants) == 8
        samples.append(
            {
                "kind": "reference",
                "split": "train",
                "path": path,
                "pixels_path": None,
                "sku_id": legacy[number]["sku_id"],
                "product_number": number,
                "group": f"sku-{number}",
                "sha256": copy_image(destination, path, source),
                "pixels_sha256": None,
                "reference_variants": variants,
                "source": str(source.relative_to(PROJECT)),
            }
        )
    assert len(samples) == 175 and len(legacy) == 32 and len(refs) == 1400
    return samples, refs, legacy


def video_records(original, destination: Path, queries: list, legacy: dict) -> list:
    records = []
    for row in queries:
        kind = "positive_crop" if row["number"] else "negative_crop"
        video, track, frame = row["video"], row["track"], row["frame"]
        number = row["number"] or None
        prefix = f"{kind}s/{video}/{track}/{frame:06d}"
        path, ready = f"{prefix}/crop.png", f"{prefix}/rgb224.png"
        if number is None:
            source = original.SAMPLES / video / track / f"{frame:06d}.png"
            crop_hash = copy_image(destination, path, source)
            archive_name = f"samples/{video}/{track}/{frame:06d}.png"
            video_source = f"tests/video/no/{video.removeprefix('no-')}.mp4"
            masks = None
        else:
            cache, _, _ = original.VIDEO_SOURCES[video]
            folder = original.BENCH / cache
            source = folder / "crop-5" / track / f"{frame:06d}.png"
            mask = folder / "bg0-5" / track / f"{frame:06d}-mask.png"
            hand = folder / "scores" / track / f"{frame:06d}-hand.png"
            if not hand.is_file():
                hand = folder / "score-work" / track / f"{frame:06d}" / track / hand.name
            masks = {
                "bg0": {"path": str(mask.relative_to(PROJECT)), "sha256": file_sha256(mask)},
                "hand": {"path": str(hand.relative_to(PROJECT)), "sha256": file_sha256(hand)}
                if hand.is_file()
                else None,
            }
            archive_name = str(source.relative_to(PROJECT))
            video_source = f"tests/video/{CABINET}/{video}.mp4"
            combined = np.dstack((row["bgr"], row["mask"].astype(np.uint8) * 255))
            output = destination / path
            output.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(output), combined), output
            crop_hash = file_sha256(output)
        restored = cv2.imread(str(destination / path), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(restored[:, :, :3], row["bgr"])
        assert np.array_equal(restored[:, :, 3] > 0, row["mask"])
        records.append(
            {
                "kind": kind,
                "split": "train",
                "path": path,
                "pixels_path": ready,
                "sku_id": legacy[number]["sku_id"] if number else None,
                "product_number": number,
                "group": video,
                "track_id": track,
                "frame": frame,
                "sha256": crop_hash,
                "pixels_sha256": save_rgb(destination, ready, row["pixels"]),
                "rgb_sha256": pixels_sha256(row["pixels"]),
                "source": {
                    "video": video_source,
                    "crop": archive_name,
                    "sha256": file_sha256(source),
                    "masks": masks,
                },
            }
        )
    return records


def external_boxes(data_root: Path) -> dict:
    """还原原始 Drink-284 截框和样本标识，后续逐像素比对离线入口。"""
    boxes_by_id = {}
    seen = Counter()
    classes = {4, 5, 6, 7, 9}
    for label in sorted((data_root / "labels/train").glob("*.txt")):
        image_path = data_root / "images/train" / (label.stem + ".jpg")
        boxes = [line.split() for line in label.read_text().splitlines() if line.strip()]
        selected = [box for box in boxes if int(box[0]) in classes and seen[int(box[0])] < 11]
        if not selected or not image_path.exists():
            continue
        bgr = cv2.imread(str(image_path))
        h, w = bgr.shape[:2]
        for index, box in enumerate(selected):
            category = int(box[0])
            if seen[category] >= 11:
                continue
            cx, cy, dw, dh = map(float, box[1:5])
            left, right = max(0, round((cx - dw / 2) * w)), min(w, round((cx + dw / 2) * w))
            top, bottom = max(0, round((cy - dh / 2) * h)), min(h, round((cy + dh / 2) * h))
            if right - left < 16 or bottom - top < 16:
                continue
            key = f"{category}/{label.stem}/{index}"
            boxes_by_id[key] = (
                image_path,
                label,
                (left, top, right, bottom),
                bgr[top:bottom, left:right].copy(),
            )
            seen[category] += 1
    return boxes_by_id


def external_records(original, destination: Path, selected_ids: list[str]) -> tuple[list, list]:
    boxes = external_boxes(original.OTHER)
    ready_by_id = dict(original.other_catalog(SIZE))
    assert set(ready_by_id) == set(boxes)
    assert list(selected_ids) == [k for k in ready_by_id if k in set(selected_ids)]
    selected, records = [], []
    for external_id in selected_ids:
        image, label, bounds, bgr = boxes[external_id]
        pixels = original.query_pixels(bgr, np.ones(bgr.shape[:2], dtype=bool), SIZE)
        assert np.array_equal(ready_by_id[external_id], pixels), external_id
        prefix = f"negative_crops/Drink-284/{external_id}"
        crop_path, ready_path = f"{prefix}/crop.png", f"{prefix}/rgb224.png"
        crop = destination / crop_path
        crop.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(crop), bgr), crop
        assert np.array_equal(cv2.imread(str(crop)), bgr)
        records.append(
            {
                "kind": "negative_crop",
                "split": "train",
                "path": crop_path,
                "pixels_path": ready_path,
                "sku_id": None,
                "product_number": None,
                "group": "Drink-284",
                "external_id": external_id,
                "sha256": file_sha256(crop),
                "pixels_sha256": save_rgb(destination, ready_path, pixels),
                "rgb_sha256": pixels_sha256(pixels),
                "source": {
                    "dataset": "Drink_284_Detection_Labelme",
                    "image": str(image.relative_to(original.OTHER)),
                    "image_sha256": file_sha256(image),
                    "label": str(label.relative_to(original.OTHER)),
                    "label_sha256": file_sha256(label),
                    "bbox_xyxy": list(bounds),
                },
            }
        )
        selected.append((external_id, ready_by_id[external_id]))
    assert len(records) == len(selected) == 52
    return records, selected


def training_input_digest(refs: list, queries: list, external: list) -> str:
    """按原实验的随机序列计算全部 110 个输入批次的 CPU 指纹。"""
    rng = np.random.default_rng(SEED)
    known, known_video, unknown = (defaultdict(list) for _ in range(3))
    for number, _, pixels in refs:
        known[number].append(pixels)
    for row in queries:
        if row["number"]:
            known_video[row["number"]].append(row["pixels"])
        else:
            unknown[row["video"]].append(row["pixels"])
    unknown["Drink-284"] = [pixels for _, pixels in external]
    names = list(unknown)

    def sample(group: list) -> np.ndarray:
        pixels = group[int(rng.integers(len(group)))].astype(np.float32)
        channel = rng.uniform(0.82, 1.18, size=(1, 1, 3))
        pixels = np.clip(pixels * channel * rng.uniform(0.78, 1.22), 0, 255).astype(np.uint8)
        if rng.random() < 0.22:
            pixels = cv2.GaussianBlur(pixels, (3, 3), rng.uniform(0.35, 1.4))
        return pixels

    digest = hashlib.sha256()
    for _ in range(STEPS):
        slots = rng.choice(list(range(1, 33)), size=6, replace=False).tolist()
        slots.extend(int(rng.choice(list(known_video))) for _ in range(2))
        positives = [sample(known[number]) for number in slots]
        positives.extend(
            sample(
                known_video[number]
                if number in known_video and rng.random() < 0.7
                else known[number]
            )
            for number in slots
        )
        selected_unknown = [names[index] for index in rng.integers(len(names), size=8)]
        negatives = [sample(unknown[name]) for name in selected_unknown]
        digest.update(np.stack(positives + negatives).tobytes())
        digest.update(bytes(number - 1 for number in slots * 2))
    return digest.hexdigest()


def restored_samples(original, destination: Path, records: list) -> tuple[list, list, list]:
    refs, queries, external = [], [], []
    for row in records:
        if row["kind"] == "reference":
            reference, _, _ = original.decode((destination / row["path"]).read_bytes())
            name = Path(row["path"]).name
            for (suffix, variant), evidence in zip(
                original.image_variants(reference), row["reference_variants"], strict=True
            ):
                bgr = variant[:, :, :3]
                if variant.shape[2] == 4:
                    alpha = variant[:, :, 3:4].astype(np.float32) / 255
                    bgr = (bgr.astype(np.float32) * alpha).astype(np.uint8)
                rgb = cv2.cvtColor(
                    cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB
                )
                assert suffix == evidence["suffix"] and pixels_sha256(rgb) == evidence["rgb_sha256"]
                refs.append((row["product_number"], name + suffix, rgb))
            continue
        with Image.open(destination / row["pixels_path"]) as image:
            pixels = np.asarray(image)
        assert pixels_sha256(pixels) == row["rgb_sha256"]
        if row["group"] == "Drink-284":
            external.append((row["external_id"], pixels))
        else:
            queries.append(
                {
                    "video": row["group"],
                    "split": "train",
                    "number": row["product_number"] or 0,
                    "pixels": pixels,
                }
            )
    return refs, queries, external


def training_input(records: list) -> dict:
    positives, negatives = [], []
    for row in records:
        kind = row["kind"]
        if kind == "reference":
            positives.append({"path": row["path"], "sku_id": row["sku_id"], "reference": True})
        elif kind == "positive_crop":
            positives.append(
                {"path": row["pixels_path"], "sku_id": row["sku_id"], "preprocessed": True}
            )
        else:
            negatives.append(
                {
                    "path": row["pixels_path"],
                    "preprocessed": True,
                    "source_group": row["group"],
                }
            )
    return {"positives": positives, "negatives": negatives}


def exclusions(original, report: dict) -> dict:
    """保留划分和帧数证据，训练目录内不保存这些帧。"""
    groups = defaultdict(Counter)
    for row in report["tracks"]:
        if row["split"] in {"calibrate", "test"}:
            groups[row["split"]][row["video"]] += row["frame_count"]
    groups["excluded"]["no-4"] = 0
    output = {}
    for split, counts in groups.items():
        output[split] = []
        for video, frames in sorted(counts.items()):
            if video.startswith("no-"):
                source = f"tests/video/no/{video.removeprefix('no-')}.mp4"
            else:
                source = f"tests/video/{CABINET}/{video}.mp4"
                assert video in original.VIDEO_SOURCES
            item = {"video": video, "source": source, "frames_in_offline_evidence": frames}
            if (PROJECT / source).is_file():
                item["sha256"] = file_sha256(PROJECT / source)
            if video == "no-4":
                item["reason"] = "故意排除与测试视频 no-2 疑似同款的文件"
            output[split].append(item)
    return output


def verify_files(destination: Path, records: list, inputs: dict) -> None:
    for row in records:
        assert file_sha256(destination / row["path"]) == row["sha256"]
        if row["pixels_path"]:
            assert file_sha256(destination / row["pixels_path"]) == row["pixels_sha256"]
    assert len(inputs["positives"]) == 227 and len(inputs["negatives"]) == 172
    assert all(
        (destination / item["path"]).is_file() for group in inputs.values() for item in group
    )
    groups = {row["group"] for row in records}
    assert not groups.intersection(
        {"6-1", "6-2", "no-2", "no-4", "2-2", "10-2", "23-2", "no-8", "no-13", "no-18", "no-22"}
    )


def prepare(experiment: Path, destination: Path) -> dict:
    original = load_experiment(experiment)
    report = read_json(experiment / "vit-7b-arcface.json")
    samples, refs, legacy = reference_records(original, destination)
    positive = [
        q
        for q in capture_query_inputs(original, original.cached_positive, SIZE)
        if q["split"] == "train"
    ]
    negative = capture_query_inputs(original, original.collected_negatives, SIZE, NEGATIVE_GROUPS)
    assert Counter(row["video"] for row in positive) == {"2-1": 7, "10-1": 8, "12-1": 29, "23-1": 8}
    assert all(row["split"] == "train" for row in negative)
    assert Counter(row["video"] for row in negative) == {
        video: count for video, count in report["negative_training_sources"] if video != "Drink-284"
    }
    queries = positive + negative
    samples.extend(video_records(original, destination, queries, legacy))
    extra_records, external = external_records(original, destination, report["external_sample_ids"])
    samples.extend(extra_records)
    original_digest = training_input_digest(refs, queries, external)
    restored_digest = training_input_digest(*restored_samples(original, destination, samples))
    assert restored_digest == original_digest
    inputs = training_input(samples)
    verify_files(destination, samples, inputs)
    counts = Counter(row["kind"] for row in samples)
    assert counts == {"reference": 175, "positive_crop": 52, "negative_crop": 172}
    manifest = {
        "schema_version": 1,
        "name": "arcface-v1",
        "cabinet_id": CABINET,
        "source": {
            "training_run": "2026-11-25 ViT-7B+ArcFace",
            "training_patch_sha256": file_sha256(experiment / "vit-7b-arcface.safetensors"),
            "dataset_code_sha256": file_sha256(experiment / "dataset.py"),
            "training_code_sha256": file_sha256(experiment / "train_7b_arcface.py"),
            "benchmark_code_sha256": file_sha256(experiment / "benchmark.py"),
        },
        "counts": {
            **counts,
            "sku": 32,
            "reference_variants": 1400,
            "frame_negative": 120,
            "external_negative": 52,
        },
        "training": {
            "input_size": [SIZE, SIZE],
            "random_seed": SEED,
            "steps": STEPS,
            "batch_size": 24,
            "positive_per_batch": 16,
            "negative_per_batch": 8,
            "augmentation": {
                "reference_scale_divisors": [1, 2, 3, 4],
                "reference_clockwise_angles": [0, 90],
                "channel_gain": [0.82, 1.18],
                "brightness_gain": [0.78, 1.22],
                "blur_probability": 0.22,
                "blur_sigma": [0.35, 1.4],
            },
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "cached_crop_saturation": 1.5,
            "cached_crop_contrast": 1.1,
            "ordered_augmented_rgb_and_targets_sha256": original_digest,
            "cpu_reproduction_match": original_digest == restored_digest,
        },
        "holdouts": exclusions(original, report),
        "samples": samples,
    }
    write_json(destination / "manifest.json", manifest)
    write_json(destination / "training-input.json", inputs)
    write_readme(destination, counts, original_digest)
    return manifest


def write_readme(destination: Path, counts: Counter, digest: str) -> None:
    readme = f"""# ViT-7B + ArcFace 本次训练数据

这份数据对应已完成的 110 步离线训练，来自柜号 `{CABINET}`。数据文件可在工作机中直接读取，父目录为项目的 `models/data`。

| 数据 | 数量 | 来源 |
|---|---:|---|
| 商品参考原图 | {counts["reference"]} | 32 个实际 SKU，保留原始 PNG 和透明通道 |
| 真商品视频裁剪 | {counts["positive_crop"]} | `2-1` 7 帧，`10-1` 8 帧，`12-1` 29 帧，`23-1` 8 帧 |
| 库外视频裁剪 | 120 | `no-1`、`no-3`、`no-7`、`no-11`、`no-15`、`no-16` |
| 外部负样本裁剪 | 52 | Drink-284 中本次训练实际选择的截框 |

## 使用

- `training-input.json` 列出相对路径和训练标签。正样本的 `sku_id` 为实际商品库 ID。负样本使用 `source_group` 保留原视频或 Drink-284 分组，无 SKU 标签。
- 标注 `reference: true` 的正样本是原始参考图。原训练按尺寸除以 1、2、3、4，每个尺寸保留原方向和顺时针 90°，得到 1,400 个参考变体。
- 标注 `preprocessed: true` 的图像是已准备好的 RGB 224×224 PNG。视频裁剪同时存储去手后最终掩码的 RGBA 原尺寸图；外部裁剪另存原尺寸 RGB 图。
- `manifest.json` 记录逐文件 SHA-256、原始裁剪及掩码来源、外部样本标注框、SKU 标签和增广设置。掩码比对复用了原离线数据选择代码。颜色描述符未用于本次训练损失。

## 数据划分与复核

- 训练目录仅含实际训练输入。测试视频 `6-1`、`6-2` 和 `no-2`，标定视频 `2-2`、`10-2`、`23-2`、`no-8`、`no-13`、`no-18`、`no-22`，以及排除的 `no-4` 只在清单中记录。
- 已从归档读回全部图像，按原采样顺序、随机种子 119 和原训练的增广规则复算 110 个批次。增强后 RGB 序列与分类标签合并指纹与本次源数据一致。

`{digest}`
"""
    (destination / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True, help="原离线实验目录")
    parser.add_argument("--output", type=Path, default=PROJECT / "models/data/arcface-v1")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arcface-source-", dir=target.parent) as temporary:
        pending = Path(temporary) / target.name
        pending.mkdir(mode=0o755)
        manifest = prepare(args.experiment.resolve(), pending)
        pending.rename(target)
    print(target)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(manifest["training"]["ordered_augmented_rgb_and_targets_sha256"])


if __name__ == "__main__":
    main()
