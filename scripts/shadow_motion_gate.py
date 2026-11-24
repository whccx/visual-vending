"""离线对照手部和全画面运动筛帧，不改变完整 HOI 基线。"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.shadow_hand_gate import evaluate, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("hand_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    hand_report = read_json(args.hand_report)
    stride = hand_report["stride"]
    videos = sorted(args.baseline.glob("video-*"))
    if len(videos) != len(hand_report["videos"]):
        raise ValueError("视频数量与手部影子报告不一致")
    results = []
    started = time.monotonic()
    for video, hand in zip(videos, hand_report["videos"]):
        info = read_json(video / "split.json")
        if info["source"] != hand["source"]:
            raise ValueError("视频与手部影子报告不对应")
        observations = []
        previous = None
        for sample in hand["samples"]:
            frame = sample["frame"]
            image = cv2.imread(str(video / "frames" / f"{frame:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"无法读取 {info['source']} 第 {frame} 帧")
            image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            changed = None
            if previous is not None:
                # 减去整体亮度漂移，再衡量局部有多少像素产生变化。
                diff = image.astype(np.int16) - previous.astype(np.int16)
                diff -= int(np.median(diff))
                changed = float((np.abs(diff) >= 15).mean())
            previous = image
            observations.append({**sample, "motion_fraction": changed})
        policies = []
        for threshold in (0.005, 0.01, 0.02, 0.05):
            for radius in (10, 15, 25, 40):
                combined = [
                    dict(
                        sample,
                        hands=(
                            sample["hands"]
                            or sample["motion_fraction"] is None
                            or sample["motion_fraction"] >= threshold
                        ),
                    )
                    for sample in observations
                ]
                result = evaluate(video, info["frame_count"], combined, stride, radius)
                result["motion_fraction_threshold"] = threshold
                policies.append(result)
        results.append(
            {
                "source": info["source"],
                "frame_count": info["frame_count"],
                "samples": observations,
                "policies": policies,
            }
        )
        print(f"{info['source']} 运动扫描完成", flush=True)
    report = {
        "mode": "shadow",
        "stride": stride,
        "motion_method": "480x270 灰度差，像素差15，减去整体亮度变化",
        "motion_seconds": time.monotonic() - started,
        "videos": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(args.output)
    print(f"影子测试完成 {args.output}", flush=True)


if __name__ == "__main__":
    main()
