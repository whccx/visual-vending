"""常驻轻量人物检测进程，通过标准输入接收逐行 JSON 视频请求。"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# 此进程通过绝对脚本路径启动；将项目根加入模块搜索路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app import config
from app.logging import setup_logging

setup_logging("person-gate")
LOG = logging.getLogger("sku.person_gate")

# 保留原标准输出作为协议通道，将模型库的普通输出重定向到标准错误。
stream = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
import cv2
import torch
from ultralytics import YOLO

torch.set_num_threads(4)
cv2.setNumThreads(2)
model = YOLO(str(config.PERSON_MODEL))
LOG.info("人物检测模型加载完成")
stream.write(json.dumps({"ready": True}) + "\n")
for line in sys.stdin:
    try:
        request = json.loads(line)
        cap = cv2.VideoCapture(request["video"])
        if not cap.isOpened():
            raise ValueError("无法读取视频")
        flags = []
        started = time.monotonic()
        try:
            while True:
                images = []
                for _ in range(16):
                    ok, image = cap.read()
                    if not ok:
                        break
                    images.append(image)
                if not images:
                    break
                # 最多 16 帧合批检测，只保留人物类别是否出现。
                results = model.predict(
                    images,
                    device=0,
                    imgsz=640,
                    rect=False,
                    conf=0.01,
                    classes=[0],
                    iou=0.5,
                    max_det=100,
                    verbose=False,
                )
                if len(results) != len(images):
                    raise RuntimeError("人体检测结果帧数不一致")
                flags.extend(bool(len(result.boxes)) for result in results)
        finally:
            cap.release()
        if not flags:
            raise ValueError("视频没有可解码帧")
        stream.write(json.dumps({"flags": flags, "seconds": time.monotonic() - started}) + "\n")
        LOG.info("人物扫描完成：帧数=%d 耗时=%.3f 秒", len(flags), time.monotonic() - started)
    except Exception as exc:
        LOG.exception("人物扫描失败")
        stream.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
