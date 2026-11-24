"""离线影子评估 GPU 手部筛帧，保留逐采样帧证据和原有 HOI 对照。"""

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp

from app.detection.held_product_detect import load_gpu_hands


def read_json(path):
    return json.loads(path.read_text())


def ranges(values):
    """将相邻帧号压缩成闭区间，便于人工检查。"""
    result = []
    for value in sorted(values):
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def evaluate(video, count, samples, stride, radius):
    """计算候选筛帧与基线检测、有效轨迹及评分窗口的重合。"""
    positive = [row["frame"] for row in samples if row["hands"]]
    keep = set()
    for frame in positive:
        keep.update(range(max(1, frame - radius), min(count, frame + radius) + 1))
    # 未扫描尾部的帧依样本间距纳入最后一个采样的保护区域。
    if samples and count - samples[-1]["frame"] >= stride:
        raise ValueError("采样未覆盖视频结尾")
    hoi = [read_json(video / "hoi-5" / f"{frame:06d}.json") for frame in range(1, count + 1)]
    inferred = {row["frame"] for row in hoi if row["inferred"]}
    linked = {row["frame"] for row in hoi if any(item["hand_indices"] for item in row["products"])}
    track = read_json(video / "track-5.json")
    track_frames = {point["frame"] for item in track["tracks"] for point in item["frames"]}
    result = read_json(video / "result.json")
    scored_frames = {frame for item in result["tracks"] for frame in item["frames"]}
    return {
        "radius_frames": radius,
        "sample_hits": len(positive),
        "keep_frames": len(keep),
        "inferred_frames": len(inferred),
        "would_run_hoi": len(inferred & keep),
        "would_skip_hoi": len(inferred - keep),
        "skip_fraction": len(inferred - keep) / len(inferred) if inferred else 0,
        "missing_track_frames": sorted(track_frames - keep),
        "missing_scoring_frames": sorted(scored_frames - keep),
        "missing_linked_frames": len(linked - keep),
        "linked_frames": len(linked),
        "kept_ranges": ranges(keep),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="完整逐帧检测和识别实验目录")
    parser.add_argument("output", type=Path, help="本次影子测试输出 JSON")
    parser.add_argument("--stride", type=int, default=5, help="采样间隔")
    parser.add_argument("--width", type=int, default=960, help="手部扫描宽度")
    args = parser.parse_args()
    if args.stride < 1 or args.width < 1:
        parser.error("采样间隔及宽度必须大于零")
    cv2.setNumThreads(1)
    started = time.monotonic()
    detector = load_gpu_hands()
    init_seconds = time.monotonic() - started
    results = []
    scan_started = time.monotonic()
    try:
        for video in sorted(args.baseline.glob("video-*")):
            info = read_json(video / "split.json")
            count = info["frame_count"]
            samples = []
            started = time.monotonic()
            for frame in range(1, count + 1, args.stride):
                path = video / "frames" / f"{frame:06d}.jpg"
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"无法读取帧图 {path}")
                height, width = image.shape[:2]
                image = cv2.resize(image, (args.width, round(height * args.width / width)))
                rotations = [
                    None,
                    cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE,
                ]
                hands = []
                for rotation in rotations:
                    rotated = image if rotation is None else cv2.rotate(image, rotation)
                    rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
                    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    hands.append(len(result.hand_landmarks))
                samples.append({"frame": frame, "hands": any(hands), "rotation_counts": hands})
            elapsed = time.monotonic() - started
            results.append(
                {
                    "source": info["source"],
                    "frame_count": count,
                    "sample_count": len(samples),
                    "scan_seconds": elapsed,
                    "samples": samples,
                    "policies": [
                        evaluate(video, count, samples, args.stride, radius)
                        for radius in (5, 10, 15, 25, 40)
                    ],
                }
            )
            print(f"{info['source']} 扫描 {len(samples)} 帧，耗时 {elapsed:.2f} 秒", flush=True)
    finally:
        detector.close()
    report = {
        "mode": "shadow",
        "detector": "MediaPipe GPU 四方向",
        "stride": args.stride,
        "width": args.width,
        "init_seconds": init_seconds,
        "scan_seconds": time.monotonic() - scan_started,
        "videos": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"影子测试完成，结果 {args.output}", flush=True)


if __name__ == "__main__":
    main()
