"""跨视频共享帧任务的离线识别流水线。

启动 detector 和 controller 两个角色，共享实验目录和 SQLite 任务队列。
每个任务完成后原子保存 JSON，控制器依据完成状态派发后续任务。
"""

import argparse
import io
import json
import os
import sqlite3
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app import config
from scripts.parallel_video_detection import read_frame, read_json, write_json, write_png
from scripts.video_recognition_stages import load_sku_features, score_track, track_video

JPEG_LOCK = threading.Lock()


class Broker:
    """SQLite 事务保证多个消费者领取不同的任务。"""

    def __init__(self, path):
        self.path = path

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def initialize(self):
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE task (
                    id INTEGER PRIMARY KEY, stage TEXT NOT NULL, video TEXT NOT NULL,
                    frame INTEGER NOT NULL DEFAULT 0, track TEXT NOT NULL DEFAULT '',
                    payload TEXT NOT NULL DEFAULT '{}', state TEXT NOT NULL DEFAULT 'pending',
                    started REAL, finished REAL, error TEXT,
                    UNIQUE(stage, video, frame, track)
                );
                CREATE INDEX task_claim ON task(stage, state, id);
                CREATE TABLE status (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    def add(self, stage, video, frame=0, track="", payload=None):
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO task(stage,video,frame,track,payload) VALUES(?,?,?,?,?)",
                (stage, video, frame, str(track), json.dumps(payload or {})),
            )

    def claim(self, stages):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            marks = ",".join("?" for _ in stages)
            row = db.execute(
                f"SELECT * FROM task WHERE state=? AND stage IN ({marks}) ORDER BY id LIMIT 1",
                ("pending", *stages),
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE task SET state='running',started=? WHERE id=?",
                    (time.monotonic(), row["id"]),
                )
            db.execute("COMMIT")
            return dict(row) if row else None

    def finish(self, task, error=None):
        with self.connect() as db:
            db.execute(
                "UPDATE task SET state=?, finished=?, error=? WHERE id=?",
                ("failed" if error else "done", time.monotonic(), error, task["id"]),
            )

    def status(self, key, value=None):
        with self.connect() as db:
            if value is not None:
                db.execute("INSERT OR REPLACE INTO status VALUES (?,?)", (key, value))
            row = db.execute("SELECT value FROM status WHERE key=?", (key,)).fetchone()
            return row[0] if row else None


def frame_path(video, frame):
    return video / "frames" / f"{frame:06d}.jpg"


def split_one(source, video, broker, gate_enabled):
    """NVDEC 和 nvJPEG 每批输出八帧，单帧写入后马上发布柜区任务。"""
    import PyNvVideoCodec as nvc
    import torch
    from PIL import Image
    from torchvision.io import encode_jpeg

    decoder = nvc.SimpleDecoder(
        str(source), use_device_memory=True, output_color_type=nvc.OutputColorType.RGBP
    )
    metadata = decoder.get_stream_metadata()
    if (metadata.height, metadata.width) != (1080, 1920) or len(decoder) <= 0:
        raise ValueError(f"视频尺寸或帧数无效 {source}")
    count = len(decoder)
    write_json(video / "metadata.json", {"frame_count": count})
    for first in range(0, count, 8):
        length = min(8, count - first)
        # 同步同批帧的解码、编码和回读，避免跨线程复用 GPU 缓冲区。
        with JPEG_LOCK:
            images = [torch.from_dlpack(frame) for frame in decoder.get_batch_frames(length)]
            if len(images) != length:
                raise ValueError(f"视频解码提前结束 {source}")
            torch.cuda.synchronize()
            encoded = encode_jpeg(images, quality=100)
            jpeg_bytes = [image.cpu().numpy().tobytes() for image in encoded]
        for index, data in enumerate(jpeg_bytes):
            for attempt in range(4):
                try:
                    with Image.open(io.BytesIO(data)) as decoded:
                        decoded.load()
                    break
                except OSError:
                    if attempt == 3:
                        raise ValueError(f"GPU JPEG 编码失败 {source}，第 {first + index + 1} 帧")
                    with JPEG_LOCK:
                        data = encode_jpeg(images[index], quality=100).cpu().numpy().tobytes()
            jpeg_bytes[index] = data
        for index, image in enumerate(jpeg_bytes, start=first + 1):
            path = frame_path(video, index)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(image)
            tmp.replace(path)
            write_json(
                video / "frames" / f"{index:06d}.json", {"video": video.name, "frame": index}
            )
            broker.add("area", video.name, index)
            if gate_enabled and (index - 1) % 5 == 0:
                broker.add("gate", video.name, index)
    write_json(
        video / "split.json",
        {
            "video": video.name,
            "source": source.name,
            "frame_count": count,
            "fps": metadata.average_fps,
            "suffix": ".jpg",
        },
    )


def area_one(task, root, model):
    from app.detection.area_detection import detect_frame_mask_with_confidence

    video, frame = root / task["video"], task["frame"]
    image = read_frame(video / "frames", frame, ".jpg")
    mask, box, confidence = detect_frame_mask_with_confidence(image, model)
    mask_name = None
    if mask is not None:
        mask_name = f"{frame:06d}-mask.png"
        write_png(video / "area-5" / mask_name, mask.astype("uint8") * 255)
    write_json(
        video / "area-5" / f"{frame:06d}.json",
        {
            "video": task["video"],
            "frame": frame,
            "box": box,
            "confidence": confidence,
            "mask": mask_name,
        },
    )


def gate_one(task, root, model):
    """每隔五帧扫描手部和全画面运动，保留两种线索。"""
    import mediapipe as mp

    video, frame = root / task["video"], task["frame"]
    image = read_frame(video / "frames", frame, ".jpg")
    height, width = image.shape[:2]
    image = cv2.resize(image, (960, round(height * 960 / width)))
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    hands = []
    for rotation in rotations:
        rotated = image if rotation is None else cv2.rotate(image, rotation)
        rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        hands.append(len(result.hand_landmarks))

    current = cv2.imread(str(frame_path(video, frame)), cv2.IMREAD_GRAYSCALE)
    if current is None:
        raise ValueError(f"无法读取运动扫描帧 {video.name}/{frame}")
    current = cv2.resize(current, (480, 270), interpolation=cv2.INTER_AREA)
    motion = None
    if frame > 5:
        previous = cv2.imread(str(frame_path(video, frame - 5)), cv2.IMREAD_GRAYSCALE)
        if previous is None:
            raise ValueError(f"无法读取上一采样帧 {video.name}/{frame - 5}")
        previous = cv2.resize(previous, (480, 270), interpolation=cv2.INTER_AREA)
        diff = current.astype(np.int16) - previous.astype(np.int16)
        diff -= int(np.median(diff))
        motion = float((np.abs(diff) >= 15).mean())
    write_json(
        video / "gate" / f"{frame:06d}.json",
        {
            "video": task["video"],
            "frame": frame,
            "rotation_counts": hands,
            "motion_fraction": motion,
            "keep": any(hands) or motion is None or motion >= 0.005,
        },
    )


def hoi_one(task, root, model):
    from detector import hoi_input

    video, frame = root / task["video"], task["frame"]
    area = read_json(video / "area-5" / f"{frame:06d}.json")["box"]
    payload = json.loads(task["payload"])
    hands, products = [], []
    if payload["inferred"]:
        tensor = hoi_input(read_frame(video / "frames", frame, ".jpg"))
        hands, products = model.decode(model.hoi.infer(tensor))
    write_json(
        video / "hoi-5" / f"{frame:06d}.json",
        {
            "video": task["video"],
            "frame": frame,
            "area": area,
            "inferred": payload["inferred"],
            "hands": hands,
            "products": products,
        },
    )


def crop_one(task, root):
    import torch
    from torchvision.io import ImageReadMode, decode_jpeg

    video, frame, track = root / task["video"], task["frame"], task["track"]
    box = json.loads(task["payload"])["box"]
    x1, y1, x2, y2 = box
    encoded = torch.from_numpy(
        np.frombuffer(frame_path(video, frame).read_bytes(), dtype=np.uint8).copy()
    )
    rgb = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cuda")
    image = rgb[:, y1:y2, x1:x2].permute(1, 2, 0).contiguous().cpu().numpy()
    directory = video / "crop-5" / track
    directory.mkdir(exist_ok=True)
    write_png(directory / f"{frame:06d}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    write_json(
        directory / f"{frame:06d}.json", {"video": task["video"], "frame": frame, "box": box}
    )


def bg0_one(task, root, model):
    video, frame, track = root / task["video"], task["frame"], task["track"]
    box = json.loads(task["payload"])["box"]
    image = read_frame(video / "frames", frame, ".jpg")
    mask = model.segment(image, box, config.BG0_MASK_THRESHOLD)
    directory = video / "bg0-5" / track
    directory.mkdir(exist_ok=True)
    x1, y1, x2, y2 = box
    rgba = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    write_png(directory / f"{frame:06d}-mask.png", rgba[:, :, 3])
    write_png(directory / f"{frame:06d}.png", rgba)
    write_json(
        directory / f"{frame:06d}.json", {"video": task["video"], "frame": frame, "box": box}
    )


def finalize_video(video):
    """按轨迹读取帧评分 JSON，复用现有窗口和拒识规则。"""
    from app.recognition.product_matching import ranking_rejection_reason, select_scoring_window
    from vending.store import outcome

    report = read_json(video / "track-5.json")
    tracks = []
    for track in report["tracks"]:
        scores = []
        for point in track["frames"]:
            path = video / "scores" / str(track["track_id"]) / f"{point['frame']:06d}.json"
            row = read_json(path)
            if "scores" in row:
                scores.append((point["frame"], row["scores"]))
        keys = list(scores[0][1]) if scores else []
        frames, ranking = select_scoring_window(scores, keys)
        tracks.append(
            {
                "track_id": track["track_id"],
                "frames": frames,
                "ranking": ranking,
                "rejection": ranking_rejection_reason(ranking, len(frames)),
            }
        )
    result = report.get("result")
    counts, unmatched = Counter(), False
    for track in tracks:
        reason = track["rejection"]
        if reason == "ambiguous_margin":
            result = outcome(4, f"前两名商品综合分差小于 {config.SCORE_MIN_MARGIN:g}，无法确认商品")
            break
        if reason == "insufficient_evidence":
            result = outcome(4, "商品轨迹的有效评分帧或匹配证据不足")
            break
        if reason == "unmatched_scores":
            unmatched = True
        else:
            counts[track["ranking"][0]["sku"]] += 1
    if result is None:
        result = (
            outcome(7, "已确认拿取，存在未能匹配本单范围内 SKU 的商品")
            if unmatched
            else outcome(
                2,
                items=[
                    {"ks_sku_id": sku, "sku_num": count} for sku, count in sorted(counts.items())
                ],
            )
        )
    write_json(video / "result.json", {"result": result, "tracks": tracks})


def worker_loop(broker, root, stages, initialize, handle):
    try:
        resource = initialize()
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()
        print(error, flush=True)
        broker.status("error", error)
        broker.status("stop", "1")
        return
    try:
        while broker.status("stop") != "1":
            task = broker.claim(stages)
            if task is None:
                time.sleep(0.02)
                continue
            try:
                handle(task, resource)
                broker.finish(task)
            except Exception:  # noqa: BLE001
                error = traceback.format_exc()
                broker.finish(task, error)
                print(error, flush=True)
                broker.status("stop", "1")
                return
    finally:
        if hasattr(resource, "close"):
            resource.close()


def detector_role(args, broker):
    import tensorrt as trt
    from bg0 import Bg0Segmenter
    from detector import Detector

    def hoi_init():
        model = Detector(Path("/models"))
        model.warmup()
        return model

    def bg_init():
        return Bg0Segmenter(Path("/models"), trt.Logger(trt.Logger.WARNING))

    def hoi_handle(task, model):
        hoi_one(task, args.output, model)

    def bg_handle(task, model):
        bg0_one(task, args.output, model)

    with ThreadPoolExecutor(max_workers=args.hoi_workers + args.bg0_workers) as pool:
        futures = [
            pool.submit(worker_loop, broker, args.output, ["hoi"], hoi_init, hoi_handle)
            for _ in range(args.hoi_workers)
        ]
        futures += [
            pool.submit(worker_loop, broker, args.output, ["bg0"], bg_init, bg_handle)
            for _ in range(args.bg0_workers)
        ]
        broker.status("detector_started", "1")
        for future in futures:
            future.result()


def controller_role(args, broker):
    started = time.monotonic()
    import torch

    from app.detection.area_detection import load_model
    from app.detection.held_product_detect import load_gpu_hands
    from app.recognition.product_matching import prepare_catalog
    from scripts.parallel_video_detection import Overlap
    from vending.engine import Pipeline

    os.environ["OPENCV_VIDEOIO_PRIORITY_CV_IMAGES"] = "1100"
    cv2.setNumThreads(1)
    sys.path.insert(0, str(config.GPU_RUNTIME / "nvcodec"))
    catalog = prepare_catalog(load_sku_features(args.database, args.sku_scope))
    scoring = (
        Pipeline().dino,
        catalog,
        torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None],
        torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None],
    )
    overlap = Overlap()
    entries = {}
    for index, source in enumerate(args.videos, start=1):
        video = args.output / f"video-{index:02d}"
        for name in (
            "frames",
            "area-5",
            "hoi-5",
            "crop-5",
            "bg0-5",
            "scores",
            "score-work",
            "gate",
        ):
            (video / name).mkdir(parents=True, exist_ok=True)
        entries[video.name] = {"source": source.name, "directory": video.name, "suffix": ".jpg"}

    def area_handle(task, model):
        area_one(task, args.output, model)

    def gate_handle(task, model):
        gate_one(task, args.output, model)

    def cpu_handle(task, state):
        video = args.output / task["video"]
        if task["stage"] == "track":
            track_video(video, entries[task["video"]], 5, 5, 5)
        elif task["stage"] == "crop":
            crop_one(task, args.output)
        else:
            point = {"frame": task["frame"], "box": json.loads(task["payload"])["box"]}
            work = video / "score-work" / task["track"] / f"{task['frame']:06d}"
            work.mkdir(parents=True)
            score_track(
                video,
                entries[task["video"]],
                {"track_id": task["track"], "frames": [point]},
                work,
                5,
                5,
                scoring,
                state,
                overlap,
            )
            out = video / "scores" / task["track"]
            out.mkdir(exist_ok=True)
            detail = read_json(work / task["track"] / f"{task['frame']:06d}.json")
            write_json(out / f"{task['frame']:06d}.json", detail)

    with ThreadPoolExecutor(
        max_workers=args.area_workers + args.cpu_workers + len(entries) + int(args.gate)
    ) as pool:
        workers = [
            pool.submit(worker_loop, broker, args.output, ["area"], load_model, area_handle)
            for _ in range(args.area_workers)
        ]
        workers += [
            pool.submit(
                worker_loop,
                broker,
                args.output,
                ["track", "crop", "score"],
                SimpleNamespace,
                cpu_handle,
            )
            for _ in range(args.cpu_workers)
        ]
        if args.gate:
            workers.append(
                pool.submit(worker_loop, broker, args.output, ["gate"], load_gpu_hands, gate_handle)
            )
        splitting = [
            pool.submit(split_one, source, args.output / name, broker, args.gate)
            for source, name in zip(args.videos, entries)
        ]
        next_frame = {name: 1 for name in entries}
        last_area = {name: None for name in entries}
        gate_next = {name: 1 for name in entries}
        gate_positive = {name: set() for name in entries}
        completed = set()
        try:
            while len(completed) < len(entries):
                if broker.status("stop") == "1":
                    raise RuntimeError("流水线任务失败，检查任务表和容器日志")
                for future in splitting:
                    if future.done():
                        future.result()
                for future in workers:
                    if future.done():
                        future.result()
                        raise RuntimeError("流水线消费者提前退出")
                with broker.connect() as db:
                    rows = db.execute("SELECT stage,video,frame,track,state FROM task").fetchall()
                states = {
                    (r["stage"], r["video"], r["frame"], r["track"]): r["state"] for r in rows
                }
                for name, entry in entries.items():
                    if name in completed:
                        continue
                    video = args.output / name
                    metadata = video / "metadata.json"
                    if args.gate:
                        if not metadata.exists():
                            continue
                        count = read_json(metadata)["frame_count"]
                        while states.get(("gate", name, gate_next[name], "")) == "done":
                            sample = gate_next[name]
                            if read_json(video / "gate" / f"{sample:06d}.json")["keep"]:
                                gate_positive[name].add(sample)
                            gate_next[name] += 5
                    while states.get(("area", name, next_frame[name], "")) == "done":
                        frame = next_frame[name]
                        if args.gate:
                            last_sample = 1 + 5 * ((min(count, frame + 10) - 1) // 5)
                            if gate_next[name] <= last_sample:
                                break
                        area = read_json(video / "area-5" / f"{frame:06d}.json")["box"]
                        if area is not None:
                            last_area[name] = frame
                        active = (
                            last_area[name] is not None
                            and frame - last_area[name] <= config.hoi_area_missing_max_frames()
                        )
                        retained = not args.gate or any(
                            abs(sample - frame) <= 10 for sample in gate_positive[name]
                        )
                        broker.add(
                            "hoi",
                            name,
                            frame,
                            payload={
                                "inferred": active and retained,
                                "baseline_inferred": active,
                                "gate_keep": retained,
                            },
                        )
                        next_frame[name] += 1
                    split = video / "split.json"
                    if not split.exists():
                        continue
                    entry.update(read_json(split))
                    count = entry["frame_count"]
                    if states.get(("track", name, 0, "")) is None and all(
                        states.get(("hoi", name, i, "")) == "done" for i in range(1, count + 1)
                    ):
                        broker.add("track", name)
                    if states.get(("track", name, 0, "")) != "done":
                        continue
                    track_report = read_json(video / "track-5.json")
                    for track in track_report["tracks"]:
                        tid = str(track["track_id"])
                        for point in track["frames"]:
                            frame = point["frame"]
                            if states.get(("crop", name, frame, tid)) is None:
                                broker.add("crop", name, frame, tid, {"box": point["box"]})
                            if states.get(("bg0", name, frame, tid)) is None:
                                broker.add("bg0", name, frame, tid, {"box": point["box"]})
                            if (
                                states.get(("crop", name, frame, tid)) == "done"
                                and states.get(("bg0", name, frame, tid)) == "done"
                                and states.get(("score", name, frame, tid)) is None
                            ):
                                broker.add("score", name, frame, tid, {"box": point["box"]})
                    if all(
                        states.get(("score", name, point["frame"], str(track["track_id"])))
                        == "done"
                        for track in track_report["tracks"]
                        for point in track["frames"]
                    ):
                        finalize_video(video)
                        completed.add(name)
                        print(
                            f"{entry['source']} 结果完成，耗时 {time.monotonic() - started:.2f} 秒",
                            flush=True,
                        )
                time.sleep(0.03)
            write_json(
                args.output / "summary.json",
                {
                    "wall_seconds": time.monotonic() - started,
                    "started_monotonic": started,
                    "finished_monotonic": time.monotonic(),
                    "gate": "hand_motion_radius10" if args.gate else "off",
                    "videos": {
                        name: read_json(args.output / name / "result.json")["result"]
                        for name in entries
                    },
                },
            )
        finally:
            broker.status("stop", "1")
        for future in workers:
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("init", "detector", "controller"))
    parser.add_argument("output", type=Path)
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--sku-scope", type=Path)
    parser.add_argument("--database", type=Path, default=Path("/workspace/data/platform.sqlite3"))
    parser.add_argument("--area-workers", type=int, default=3)
    parser.add_argument("--hoi-workers", type=int, default=2)
    parser.add_argument("--bg0-workers", type=int, default=2)
    parser.add_argument("--cpu-workers", type=int, default=5)
    parser.add_argument("--gate", action="store_true", help="启用手部和运动联合筛帧")
    args = parser.parse_args()
    broker = Broker(args.output / "queue.sqlite3")
    if args.role == "init":
        args.output.mkdir(parents=True, exist_ok=False)
        broker.initialize()
    elif args.role == "detector":
        detector_role(args, broker)
    else:
        if not args.videos or not args.sku_scope:
            parser.error("controller 需要视频路径和 --sku-scope")
        controller_role(args, broker)


if __name__ == "__main__":
    main()
