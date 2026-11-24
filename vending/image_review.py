"""商品主体与画质审核，使用本项目的本地模型执行检测。"""

import io
import json
import logging
import os
import select
import subprocess
import tempfile
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from vending.config import Settings

LOG = logging.getLogger("vending.image_review")

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "models/review"
ONNX_SCRIPT = Path(__file__).resolve().with_name("onnx_detect.py")
ONNX_MODELS = (
    ONNX_DIR
    / "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",
    ONNX_DIR / "textual.onnx",
    ONNX_DIR / "non_maximum_suppression.onnx",
)
DEFAULT_ONNX_PYTHON = "/opt/comfy-conda/bin/python"
DEFAULT_ONNX_LIBS = "/opt/comfy-conda/lib/python3.11/site-packages/nvidia/cu13/lib"

PRODUCT_CLASSES = [
    "Can",
    "Bottle",
    "packaging box",
    "food",
    "snack",
    "plastic bag",
    "pouch",
    "packet",
    "drinking glass",
]
# 不沿用工作流中泛化的glass/metal/ring：玻璃柜门或金属货架不是商品主体。
CLASSES = PRODUCT_CLASSES + ["hand", "person", "glass door", "shelf", "refrigerator"]
# 初始阈值是可审计规则，须用人工标注验收集校准，不能当作生产精度证明。
POLICY = {
    "detect_score": 0.12,
    "approve_score": 0.35,
    "min_area": 0.08,
    "min_center": 0.25,
    "max_center": 0.75,
    "blur_reject": 15.0,
    "blur_review": 45.0,
    "full_frame_approve_score": 0.55,
    "subject_selection": "centered_confident_first_v1",
}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def assess(image, detections):
    """检测框与图像规则分离，便于用明确的几何案例验证必要条件。"""
    width, height = image.size
    candidates = []
    for raw in detections:
        if raw.get("class") not in PRODUCT_CLASSES:
            continue
        score = float(raw["confidence"])
        box = np.asarray(raw["bbox"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score):
            raise ValueError("检测框不合法")
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1 or score < POLICY["detect_score"]:
            continue
        item = dict(raw, bbox=[x1, y1, x2, y2], area_ratio=(x2 - x1) * (y2 - y1) / (width * height))
        candidates.append(item)
    # 跨类别去重，同一瓶子不能因为同时被识别成Can/food而被当作多个商品。
    unique = []
    for item in sorted(candidates, key=lambda d: d["confidence"], reverse=True):
        if all(iou(item["bbox"], other["bbox"]) < 0.6 for other in unique):
            unique.append(item)
    result = {
        "status": "completed",
        "detections": unique,
        "dimensions": [width, height],
        "metrics": {},
    }

    def verdict(decision, code, reason):
        return dict(result, decision=decision, reason_code=code, reason=reason)

    if not unique:
        return verdict("rejected", "no_product", "未检测到商品主体")

    # 先选居中主体，再优先选检测分数达标的候选，避免弱大框覆盖可靠主体。
    def rank(item):
        x1, y1, x2, y2 = item["bbox"]
        contains = x1 <= width / 2 <= x2 and y1 <= height / 2 <= y2
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        centered = (
            contains
            and POLICY["min_center"] <= cx <= POLICY["max_center"]
            and POLICY["min_center"] <= cy <= POLICY["max_center"]
        )
        return (
            centered,
            item["confidence"] >= POLICY["approve_score"],
            item["area_ratio"],
            item["confidence"],
        )

    best = max(unique, key=rank)
    x1, y1, x2, y2 = best["bbox"]
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    # 固定尺度计算主体内部清晰度，避免分辨率差异及商品外轮廓抬高模糊分数。
    crop.thumbnail((256, 256))
    gray = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    dx = max(1, w // 10)
    dy = max(1, h // 10)
    core = gray[dy:-dy, dx:-dx]
    if core.size < 16:
        core = gray
    sharpness = float(cv2.Laplacian(core, cv2.CV_64F).var())
    contains = x1 <= width / 2 <= x2 and y1 <= height / 2 <= y2
    result["subject"] = best
    result["metrics"] = {
        "center_x": round(cx, 4),
        "center_y": round(cy, 4),
        "area_ratio": round(best["area_ratio"], 4),
        "confidence": round(float(best["confidence"]), 4),
        "sharpness": round(sharpness, 2),
    }
    centered = (
        POLICY["min_center"] <= cx <= POLICY["max_center"]
        and POLICY["min_center"] <= cy <= POLICY["max_center"]
        and contains
    )
    if not centered:
        return verdict("rejected", "off_center", "商品主体未位于图片中央")
    if best["area_ratio"] < 0.03:
        return verdict("rejected", "subject_too_small", "商品主体过小，不能清楚展示商品")
    if best["area_ratio"] < POLICY["min_area"] or min(x2 - x1, y2 - y1) < 48:
        return verdict(
            "needs_review", "small_subject", "中央存在疑似商品，但主体占比或有效像素不足"
        )
    if best["area_ratio"] >= 0.95 and best["confidence"] < POLICY["full_frame_approve_score"]:
        return verdict(
            "needs_review", "full_frame_box", "主体几乎覆盖全图且检测分数较低，需确认商品主体"
        )
    if sharpness < POLICY["blur_reject"]:
        return verdict("rejected", "blurred_subject", "商品主体明显模糊，清晰度不足")
    if sharpness < POLICY["blur_review"]:
        return verdict(
            "needs_review", "borderline_sharpness", "商品主体位于中央，但清晰度处于临界范围"
        )
    others = [
        d
        for d in unique
        if d is not best
        and d["area_ratio"] >= best["area_ratio"] * 0.35
        and iou(d["bbox"], best["bbox"]) < 0.25
    ]
    if others:
        return verdict("needs_review", "multiple_subjects", "存在多个明显商品，需确认中央主体")
    if best["confidence"] < POLICY["approve_score"]:
        return verdict(
            "needs_review", "weak_detection", "中央存在疑似商品，但检测分数不足以自动通过"
        )
    return verdict(
        "approved", "clear_center_product", "中央存在清晰商品主体，位置、占比和清晰度符合要求"
    )


class LocalProductDetector:
    @cached_property
    def model(self):
        from app.detection.yoloe_cutout import load_model

        model = load_model()
        labels = [name.lower() for name in CLASSES]
        embeddings = model.model.get_text_pe(labels, cache_clip_model=True)
        model.set_classes(labels, embeddings=embeddings)
        return model

    def __call__(self, image: Image.Image) -> list[dict]:
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        result = self.model.predict(
            bgr,
            device=0,
            imgsz=640,
            conf=POLICY["detect_score"],
            iou=0.5,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes
        return [
            {"class": CLASSES[int(label)], "confidence": score, "bbox": box}
            for label, score, box in zip(
                boxes.cls.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.xyxy.cpu().tolist(),
                strict=True,
            )
        ]


class OnnxProductDetector:
    """通过本项目镜像内的 ONNX 运行时（comfy 环境）执行 YOLO-World 检测。

    worker 主进程没有 onnxruntime，此处按需拉起镜像内自带 onnxruntime-gpu 的
    Python 子进程，常驻复用已加载的三个 ONNX 会话。
    """

    def __init__(
        self,
        *,
        python: str = DEFAULT_ONNX_PYTHON,
        script: Path = ONNX_SCRIPT,
        classes: list[str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.python = python
        self.script = Path(script)
        self.classes = list(classes or CLASSES)
        self.timeout = timeout
        self.process = None
        self._tokens = None

    @staticmethod
    def ready(python: str) -> bool:
        """运行时、脚本、模型和词表编码依赖齐备时才可用。"""
        if not Path(python).exists() or not ONNX_SCRIPT.exists():
            return False
        if not all(path.exists() for path in ONNX_MODELS):
            return False
        try:
            from clip.simple_tokenizer import SimpleTokenizer  # noqa: F401
        except ImportError:
            return False
        return True

    def tokens(self) -> list[list[int]]:
        """用本项目运行库中的 CLIP 词表把类别名编成 textual 模型输入。"""
        if self._tokens is None:
            from clip.simple_tokenizer import SimpleTokenizer

            self._tokens = SimpleTokenizer()(self.classes, context_length=77).tolist()
        return self._tokens

    def _ensure_process(self) -> subprocess.Popen:
        if self.process is not None and self.process.poll() is None:
            return self.process
        if not self.ready(self.python):
            raise RuntimeError("YOLO-World 缺少本容器的运行时、模型或词表依赖")
        environment = dict(os.environ)
        libs = os.environ.get("VENDING_REVIEW_ONNX_LIBS", DEFAULT_ONNX_LIBS)
        if Path(libs).is_dir():
            environment["LD_LIBRARY_PATH"] = (
                libs + ":" + environment.get("LD_LIBRARY_PATH", "")
            )
        self.process = subprocess.Popen(
            [self.python, str(self.script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=environment,
            cwd=str(ROOT),
        )
        return self.process

    def _read_line(self) -> str:
        descriptor = self.process.stdout.fileno()
        ready, _, _ = select.select([descriptor], [], [], self.timeout)
        if not ready:
            raise TimeoutError("商品检测子进程超时")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("商品检测子进程已退出")
        return line

    def __call__(self, image: Image.Image) -> list[dict]:
        """把图片交给常驻子进程检测，返回类别、分数和原图坐标检测框。"""
        process = self._ensure_process()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as handle:
            path = handle.name
        try:
            image.save(path, "PNG")
            request = json.dumps(
                {
                    "classes": self.classes,
                    "tokens": self.tokens(),
                    "images": [path],
                    "score_threshold": POLICY["detect_score"],
                    "iou_threshold": 0.5,
                },
                ensure_ascii=False,
            )
            process.stdin.write(request + "\n")
            process.stdin.flush()
            reply = json.loads(self._read_line())
            if "error" in reply:
                raise RuntimeError(f"商品检测失败：{reply['error']}")
            return reply["results"][0]
        except (BrokenPipeError, OSError) as exc:
            self._restart()
            raise RuntimeError(f"商品检测子进程不可用：{exc}") from exc
        finally:
            Path(path).unlink(missing_ok=True)

    def _restart(self) -> None:
        if self.process is not None:
            self.process.kill()
            self.process.wait()
            self.process = None


def make_product_detector(config: Settings) -> Callable[[Image.Image], list[dict]]:
    """按配置选择审核检测器：容器内 ONNX YOLO-World 或本地 YOLOE。"""
    choice = config.review_detector
    python = config.review_onnx_python
    if choice == "onnx" or (choice == "auto" and OnnxProductDetector.ready(python)):
        LOG.info("商品审核检测器：容器内 YOLO-World ONNX")
        return OnnxProductDetector(python=python)
    if choice not in ("auto", "yoloe"):
        raise ValueError(f"未知的商品审核检测器配置：{choice}")
    LOG.info("商品审核检测器：本地 YOLOE")
    return LocalProductDetector()


def review_image(content: bytes, detector: Callable[[Image.Image], list[dict]]) -> dict:
    """从下载的原始图片直接得到单图审核结论。"""
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 30_000_000:
                return {
                    "decision": "rejected",
                    "reason_code": "too_many_pixels",
                    "reason": "图片像素超限",
                }
            rgba = ImageOps.exif_transpose(source).convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        return {
            "decision": "rejected",
            "reason_code": "invalid_image",
            "reason": "文件不是可解码图片",
        }

    dimensions = list(image.size)
    if min(dimensions) < 64:
        return {
            "decision": "needs_review",
            "reason_code": "low_resolution",
            "reason": "图片分辨率不足，无法确认清晰商品主体",
            "dimensions": dimensions,
        }
    if np.asarray(image).std() < 2:
        return {
            "decision": "rejected",
            "reason_code": "blank_image",
            "reason": "空白或纯色图片，没有清晰商品主体",
            "dimensions": dimensions,
        }

    try:
        return assess(image, detector(image))
    except Exception:
        LOG.exception("本地商品主体检测失败")
        return {
            "decision": "needs_review",
            "reason_code": "detector_error",
            "reason": "本地商品检测异常，请重试",
            "dimensions": dimensions,
        }


class ImageReviewRequired(ValueError):
    """参考图需要人工做出审核决定。"""

    def __init__(self, findings: list[dict]) -> None:
        super().__init__("商品图片需人工复核")
        self.findings = findings


def check_reference_images(
    images: list[bytes],
    detector: Callable[[Image.Image], list[dict]],
) -> None:
    """硬拒绝优先。没有硬拒绝时保留所有需要人工的图片。"""
    from vending.engine import ImageRejected

    findings = []
    for index, image in enumerate(images, 1):
        result = review_image(image, detector)
        decision = result["decision"]
        if decision == "rejected":
            raise ImageRejected(f"第 {index} 张参考图：{result['reason']}")
        if decision == "needs_review":
            findings.append({"image_number": index, **result})
            if result["reason_code"] == "detector_error":
                break
    if findings:
        raise ImageReviewRequired(findings)
