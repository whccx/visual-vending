"""货柜和 HOI 检测的纯 TensorRT 实现，不依赖 PyTorch。"""

import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda


def check(result):
    if result[0] != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError("CUDA 错误：{}".format(result[0]))
    return result[1:] if len(result) > 1 else None


class Engine:
    """管理一个 TensorRT 引擎的执行上下文与 CUDA 缓冲区。"""

    def __init__(self, path, logger, input_shape=None):
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("无法加载 TensorRT 引擎：{}".format(path))
        self.context = self.engine.create_execution_context()
        self.stream = check(cuda.cudaStreamCreate())[0]
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        if input_shape is not None and not self.context.set_input_shape(self.input_name, input_shape):
            raise ValueError("TensorRT 输入形状不支持：{}".format(input_shape))
        self.buffers = {}
        for name in names:
            shape = tuple(self.context.get_tensor_shape(name))
            array = np.empty(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            pointer = check(cuda.cudaMalloc(array.nbytes))[0]
            self.buffers[name] = (array, pointer)
            if not self.context.set_tensor_address(name, int(pointer)):
                raise RuntimeError("TensorRT 张量绑定失败：{}".format(name))

    def infer(self, data):
        array, pointer = self.buffers[self.input_name]
        np.copyto(array, data)
        check(cuda.cudaMemcpyAsync(pointer, array.ctypes.data, array.nbytes,
                                   cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT 推理失败")
        for name, (array, pointer) in self.buffers.items():
            if name != self.input_name:
                check(cuda.cudaMemcpyAsync(array.ctypes.data, pointer, array.nbytes,
                                           cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        check(cuda.cudaStreamSynchronize(self.stream))
        return {name: value[0] for name, value in self.buffers.items() if name != self.input_name}

    def close(self):
        for _, pointer in self.buffers.values():
            check(cuda.cudaFree(pointer))
        check(cuda.cudaStreamDestroy(self.stream))


def soft_nms(boxes, scores, labels):
    """按类别执行与既有 HOI 测试相同的线性 Soft-NMS。"""
    remaining = np.arange(len(scores))
    work = scores.copy()
    keep = []
    while len(remaining):
        best = remaining[np.argmax(work[remaining])]
        if work[best] < 0.3:
            break
        keep.append(best)
        remaining = remaining[remaining != best]
        if not len(remaining):
            break
        left = np.maximum(boxes[remaining, :2], boxes[best, :2])
        right = np.minimum(boxes[remaining, 2:], boxes[best, 2:])
        intersection = np.prod(np.maximum(right - left, 0), axis=1)
        area = np.prod(np.maximum(boxes[remaining, 2:] - boxes[remaining, :2], 0), axis=1)
        best_area = np.prod(np.maximum(boxes[best, 2:] - boxes[best, :2], 0))
        iou = intersection / np.maximum(area + best_area - intersection, 1e-9)
        work[remaining] *= np.where((labels[remaining] == labels[best]) & (iou > 0.5), 1 - iou, 1)
        remaining = remaining[work[remaining] >= 0.3]
    return np.asarray(keep, dtype=np.int64)


def hoi_input(image):
    resized = cv2.resize(image, (800, 450)).astype(np.float32)[:, :, ::-1]
    normalized = (resized - np.array([123.675, 116.28, 103.53], np.float32)) / np.array(
        [58.395, 57.12, 57.375], np.float32
    )
    padded = cv2.copyMakeBorder(normalized, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded.transpose(2, 0, 1)[None]


def shelf_input(image):
    resized = cv2.resize(image, (640, 360))
    padded = cv2.copyMakeBorder(resized, 12, 12, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def shelf_box(output):
    scores = output[0, 4]
    index = int(np.argmax(scores))
    if scores[index] < 0.5:
        return None
    cx, cy, width, height = (float(value) for value in output[0, :4, index])
    coords = [(cx - width / 2) * 3, (cy - height / 2 - 12) * 3,
              (cx + width / 2) * 3, (cy + height / 2 - 12) * 3]
    box = [max(0, min(1920 if i % 2 == 0 else 1080, round(value))) for i, value in enumerate(coords)]
    return box if box[2] > box[0] and box[3] > box[1] else None


class Detector:
    """单实例串行处理视频，并保持三个 TensorRT 引擎常驻。"""

    def __init__(self, model_dir: Path):
        cv2.setNumThreads(1)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.model_dir = model_dir
        self.hoi = Engine(model_dir / "hoi_full_safe_fp16.engine", self.logger)
        self.relations = {}

    def warmup(self):
        self.hoi.infer(np.zeros((1, 3, 480, 800), np.float32))
        self.relations[1] = Engine(self.model_dir / "hoi_relation.engine", self.logger, (1, 512))
        self.relations[1].infer(np.zeros((1, 512), np.float32))

    def decode(self, outputs):
        logits, boxes, embeddings = (outputs[n][0] for n in ("class_logits", "boxes", "query_embeddings"))
        scores = 1 / (1 + np.exp(-np.clip(logits.reshape(-1), -80, 80)))
        indices = np.argpartition(scores, -300)[-300:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        labels, queries = indices % 3, indices // 3
        scores = scores[indices]
        selected = boxes[queries]
        xyxy = np.concatenate((selected[:, :2] - selected[:, 2:] / 2,
                               selected[:, :2] + selected[:, 2:] / 2), axis=1)
        xyxy *= np.array([1920, 1080, 1920, 1080], dtype=np.float32)
        keep = soft_nms(xyxy, scores, labels)
        hands = [int(k) for k in keep if labels[k] == 0]
        objects = [int(k) for k in keep if labels[k] == 1]
        pairs = [np.concatenate((embeddings[queries[h]], embeddings[queries[o]]))
                 for o in objects for h in hands]
        if pairs:
            count = len(pairs)
            if count not in self.relations:
                self.relations[count] = Engine(self.model_dir / "hoi_relation.engine", self.logger, (count, 512))
            links = self.relations[count].infer(np.asarray(pairs, np.float32))["logits"].argmax(axis=1)
            links = links.reshape(len(objects), len(hands))
        else:
            links = np.empty((len(objects), len(hands)))
        return ([{"box": xyxy[k].tolist()} for k in hands],
                [{"box": xyxy[k].tolist(), "score": float(scores[k]),
                  "hand_indices": np.flatnonzero(links[i]).tolist()} for i, k in enumerate(objects)])

    def detect(self, path, area_missing_max_frames=375, area_masks_boxes=None):
        """返回与输入逐帧对齐的检测结果和阶段耗时。"""
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("无法打开视频")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames, hands, areas = [], [], []
        elapsed_shelf = elapsed_hoi = elapsed_post = 0.0
        missing_frames = 0
        seen_area = False
        started = time.monotonic()
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape != (1080, 1920, 3):
                    raise ValueError("当前引擎仅支持 1920×1080 视频")
                step = time.monotonic()
                if area_masks_boxes is None:
                    raise ValueError("缺少货柜分割结果")
                if len(areas) >= len(area_masks_boxes):
                    raise ValueError("货柜掩码帧数少于视频")
                area = area_masks_boxes[len(areas)]
                elapsed_shelf += time.monotonic() - step
                areas.append(area)
                if area is None:
                    missing_frames += 1
                    if not seen_area or missing_frames > area_missing_max_frames:
                        frames.append([])
                        hands.append([])
                        continue
                else:
                    seen_area = True
                    missing_frames = 0
                data = hoi_input(image)
                step = time.monotonic()
                output = self.hoi.infer(data)
                elapsed_hoi += time.monotonic() - step
                step = time.monotonic()
                frame_hands, candidates = self.decode(output)
                elapsed_post += time.monotonic() - step
                hands.append(frame_hands)
                frames.append(candidates)
        finally:
            capture.release()
        if not frames:
            raise ValueError("视频没有可解码的帧")
        if area_masks_boxes is not None and len(area_masks_boxes) != len(frames):
            raise ValueError("货柜掩码帧数与视频不一致")
        return {"frame_count": len(frames), "fps": fps, "width": 1920, "height": 1080,
                "areas": areas, "hands": hands, "frames": frames,
                "timings": {"total_seconds": time.monotonic() - started,
                            "shelf_seconds": elapsed_shelf, "hoi_seconds": elapsed_hoi,
                            "postprocess_seconds": elapsed_post}}

    def close(self):
        for engine in self.relations.values():
            engine.close()
        self.hoi.close()
