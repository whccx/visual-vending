"""验证多线程 CUDA stream 下 nvJPEG 输出的同步正确性。"""

import argparse
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    import torch
    from torchvision.io import encode_jpeg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("original", "device_sync", "shared_stream"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    with Image.open(args.source) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    base = torch.from_numpy(pixels).permute(2, 0, 1).contiguous().cuda()
    shared = torch.cuda.Stream()
    lock = threading.Lock()
    stop = threading.Event()
    args.output.mkdir(parents=True, exist_ok=False)

    def background():
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            matrix = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
            while not stop.is_set():
                result = matrix @ matrix
                del result
                stream.synchronize()

    def worker(index):
        stream = shared if args.mode == "shared_stream" else torch.cuda.Stream()
        invalid = []
        started = time.monotonic()
        for batch in range(args.batches):
            with lock, torch.cuda.stream(stream):
                torch.cuda.synchronize()
                encoded = encode_jpeg([base] * 8, quality=100)
                if args.mode == "device_sync":
                    torch.cuda.synchronize()
                payloads = [item.cpu().numpy().tobytes() for item in encoded]
            for position, data in enumerate(payloads):
                try:
                    if not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                        raise ValueError("JPEG 标记不完整")
                    with Image.open(io.BytesIO(data)) as decoded:
                        decoded.load()
                except (OSError, ValueError) as error:
                    if len(invalid) < 3:
                        (args.output / f"bad-{index}-{batch}-{position}.jpg").write_bytes(data)
                    invalid.append({"batch": batch, "position": position, "error": str(error)})
        return {
            "worker": index,
            "stream": stream.cuda_stream,
            "encoded": args.batches * 8,
            "invalid": invalid,
            "seconds": time.monotonic() - started,
        }

    background_thread = threading.Thread(target=background)
    background_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            reports = list(pool.map(worker, range(args.workers)))
    finally:
        stop.set()
        background_thread.join()
    result = {
        "mode": args.mode,
        "workers": reports,
        "invalid": sum(len(x["invalid"]) for x in reports),
    }
    (args.output / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        args.mode,
        "帧数",
        sum(x["encoded"] for x in reports),
        "损坏",
        result["invalid"],
        "stream",
        [x["stream"] for x in reports],
        flush=True,
    )


if __name__ == "__main__":
    main()
