#!/usr/bin/env python3
"""testlab 离线识别驱动：复制代码变体，不依赖线上 worker 服务。

在 visual-vending-worker 容器内运行：
    python -m scripts.testlab.run_videos tests/video/862337050531238/10-1.mp4 ...

- 商品特征从 data/platform.sqlite3 的启用 SKU 加载，等价于线上识别提交全库。
- 判定顺序与阈值通过环境变量调整：
    TESTLAB_FLOOR_FIRST=1        先查第一名地板分再查前两名分差
    SKU_MIN_VECTOR_SCORE=0.20    向量分地板
    SKU_MIN_COLOR_SCORE=0.60     颜色分地板
    SCORE_MIN_MARGIN=0.04        前两名综合分差门槛（运行时覆盖 app.config 常量）
"""

import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ROOT = Path("/workspace")
DB_PATH = ROOT / "data" / "platform.sqlite3"
SPOOL = ROOT / "data" / "spool"
CONTAINER_ID = "862337050531238"


def load_assets() -> dict[str, bytes]:
    db = sqlite3.connect(DB_PATH)
    try:
        rows = db.execute(
            "SELECT s.id, sv.features FROM sku s JOIN sku_version sv ON sv.sku=s.id "
            "AND sv.id=s.active_version WHERE s.deleted=0 ORDER BY s.id"
        ).fetchall()
    finally:
        db.close()
    return {sku: features for sku, features in rows}


def run_one(runtime, video: Path, assets: dict[str, bytes], container_id: str) -> dict:
    started = time.time()
    result = runtime.recognize([video], container_id, assets)
    status = result["status"]
    print(
        f"RESULT video={video.name} code={status['code']} desc={status['desc']} "
        f"seconds={time.time() - started:.0f}s"
    )
    print(f"DETAIL {json.dumps(result, ensure_ascii=False)}")
    return result


def main() -> None:
    videos = [Path(arg) if Path(arg).is_absolute() else ROOT / arg for arg in sys.argv[1:]]
    missing = [path for path in videos if not path.is_file()]
    if missing or not videos:
        print("用法：python -m scripts.testlab.run_videos <视频路径...>")
        for path in missing:
            print(f"  缺少视频：{path}")
        sys.exit(1)
    from app import config

    if os.environ.get("SCORE_MIN_MARGIN"):
        config.SCORE_MIN_MARGIN = float(os.environ["SCORE_MIN_MARGIN"])
    assets = load_assets()
    print(f"商品库特征 {len(assets)} 个；环境：TESTLAB_FLOOR_FIRST="
          f"{os.environ.get('TESTLAB_FLOOR_FIRST', '0')} "
          f"SKU_MIN_VECTOR_SCORE={os.environ.get('SKU_MIN_VECTOR_SCORE', '0.20')} "
          f"SKU_MIN_COLOR_SCORE={os.environ.get('SKU_MIN_COLOR_SCORE', '0.60')} "
          f"SCORE_MIN_MARGIN={getattr(config, 'SCORE_MIN_MARGIN', '0.04')}")
    SPOOL.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="testlab-", dir=SPOOL) as tmp:
        Path(tmp).chmod(0o750)  # detector 以 nobody+group 1000 读取检测帧
        local = []
        for path in videos:
            target = Path(tmp) / path.name
            shutil.copy(path, target)
            target.chmod(0o640)
            local.append(target)
        from scripts.testlab.frame_pipeline import FrameRuntime

        runtime = FrameRuntime(1)
        try:
            outcomes = []
            for path in local:
                outcomes.append((path.name, run_one(runtime, path, assets, CONTAINER_ID)))
        finally:
            runtime.close()
    print("\n===== testlab 汇总 =====")
    for name, result in outcomes:
        status = result["status"]
        items = result.get("result") if status["code"] == 2 else None
        print(f"{name}: code={status['code']} {status['desc']}"
              + (f" items={json.dumps(items, ensure_ascii=False)}" if items else ""))


if __name__ == "__main__":
    main()
