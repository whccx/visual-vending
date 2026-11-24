"""YOLO-World ONNX 商品主体检测，供审核子进程使用。

本模块由镜像内自带 onnxruntime-gpu 的 Python 运行（worker 主进程没有
onnxruntime），只依赖 numpy、cv2 和 onnxruntime，不导入本项目的其他模块。
通过标准输入输出按 JSON 行协议接收请求：每行包含类别名、CLIP 词表编码和
待检测图片路径，返回每张图片的类别、分数和检测框。
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

IMAGE_SIZE = 640
MAX_DETECTIONS = 300
MODEL_DIR = Path(
    os.environ.get(
        "VENDING_REVIEW_ONNX_DIR", Path(__file__).resolve().parent.parent / "models/review"
    )
)
WORLD_FILE = (
    MODEL_DIR
    / "yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx"
)
TEXTUAL_FILE = MODEL_DIR / "textual.onnx"
NMS_FILE = MODEL_DIR / "non_maximum_suppression.onnx"


def providers():
    """优先使用 CUDA，缺库时回退 CPU。"""
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def transform_image(image, image_size):
    """缩放到最长边 image_size 后左上角零填充，返回 NCHW 浮点输入。"""
    height, width = image.shape[:2]
    scale = min(image_size / height, image_size / width)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)))
    padded = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded.transpose(2, 0, 1).astype(np.float32) / 255.0, (height, width)


def untransform_bboxes(bboxes, image_size, original_hw):
    """把 640 输入空间的检测框还原到原图坐标。"""
    scale = image_size / max(original_hw)
    bboxes = bboxes / scale
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, original_hw[1])
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, original_hw[0])
    return bboxes


class Detector:
    def __init__(self) -> None:
        self.world = onnxruntime.InferenceSession(str(WORLD_FILE), providers=providers())
        self.textual = onnxruntime.InferenceSession(str(TEXTUAL_FILE), providers=providers())
        self.nms = onnxruntime.InferenceSession(
            str(NMS_FILE), providers=["CPUExecutionProvider"]
        )
        self._token_key = None
        self._features = None

    def text_features(self, tokens):
        """按词表编码缓存文本向量，避免每张图片重跑文本编码。"""
        key = tuple(map(tuple, tokens))
        if self._token_key != key:
            (features,) = self.textual.run(
                None, {"input": np.asarray(tokens, dtype=np.int64)}
            )
            features = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)
            self._token_key = key
            self._features = features
        return self._features

    def suppress(self, boxes, scores, iou_threshold, score_threshold):
        selected = self.nms.run(
            ["selected_indices"],
            {
                "boxes": boxes[None, :, :].astype(np.float32),
                "scores": scores[None, :, :].transpose(0, 2, 1).astype(np.float32),
                "max_output_boxes_per_class": np.array([MAX_DETECTIONS], dtype=np.int64),
                "iou_threshold": np.array([iou_threshold], dtype=np.float32),
                "score_threshold": np.array([score_threshold], dtype=np.float32),
            },
        )[0]
        labels = selected[:, 1]
        boxes = boxes[selected[:, 2]]
        scores = scores[selected[:, 2], labels]
        if len(boxes) > MAX_DETECTIONS:
            keep = np.argsort(scores)[-MAX_DETECTIONS:]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        return boxes, scores, labels

    def detect(self, image, classes, tokens, score_threshold, iou_threshold):
        """返回每张图的检测结果列表，坐标已还原到原图。"""
        features = self.text_features(tokens)
        input_image, original_hw = transform_image(image, IMAGE_SIZE)
        scores, boxes = self.world.run(
            ["scores", "boxes"],
            {"images": input_image[None], "text_features": features[None]},
        )
        boxes, scores, labels = self.suppress(
            boxes[0], scores[0], iou_threshold, score_threshold
        )
        boxes = untransform_bboxes(boxes, IMAGE_SIZE, original_hw)
        return [
            {
                "class": classes[int(label)],
                "confidence": round(float(score), 4),
                "bbox": [round(float(value), 2) for value in box],
            }
            for box, score, label in zip(boxes, scores, labels, strict=True)
        ]


def serve() -> None:
    detector = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if detector is None:
                detector = Detector()
            results = []
            for path in request["images"]:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"无法读取图片：{path}")
                results.append(
                    detector.detect(
                        image,
                        request["classes"],
                        request["tokens"],
                        float(request.get("score_threshold", 0.12)),
                        float(request.get("iou_threshold", 0.5)),
                    )
                )
            reply = {"results": results}
        except Exception as exc:
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    serve()
