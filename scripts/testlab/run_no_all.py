#!/usr/bin/env python3
"""testlab 批量验收 tests/video/no：统计非柜视频在调整后规则下的识别码分布。

在 visual-vending-worker 容器内运行（每个视频独立临时目录，处理完即清理）：
    TESTLAB_AMBIGUOUS_UNMATCHED=1 python -m scripts.testlab.run_no_all 1 2 3 ...

输出每个视频的 RESULT 行与批次汇总，脚本结束后写 scripts/testlab/no_results.json 累计。
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import scripts.testlab.run_videos as driver

NO_DIR = driver.ROOT / "tests" / "video" / "no"
RESULTS_FILE = driver.ROOT / "data" / "testlab-no-results.json"


def load_results() -> dict:
    if RESULTS_FILE.is_file():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict) -> None:
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1, sort_keys=True))


def main() -> None:
    numbers = [int(arg) for arg in sys.argv[1:]]
    if not numbers:
        numbers = sorted(
            int(path.stem) for path in NO_DIR.glob("*.mp4") if path.stem.isdigit()
        )
    missing = [n for n in numbers if not (NO_DIR / f"{n}.mp4").is_file()]
    for n in missing:
        print(f"缺少视频：{NO_DIR / f'{n}.mp4'}")
    numbers = [n for n in numbers if n not in missing]
    results = load_results()
    if os.environ.get("TESTLAB_FORCE", "0") != "1":
        done = [n for n in numbers if str(n) in results]
        for n in done:
            print(f"SKIP video={n}.mp4 已有结果 code={results[str(n)]['code']}，跳过（TESTLAB_FORCE=1 可强制重跑）")
        numbers = [n for n in numbers if str(n) not in results]
    assets = driver.load_assets()
    print(f"商品库特征 {len(assets)} 个，待测视频 {len(numbers)} 个")
    results = load_results()
    driver.SPOOL.mkdir(exist_ok=True)
    from scripts.testlab.frame_pipeline import FrameRuntime

    runtime = FrameRuntime(1)
    try:
        import torch

        for n in numbers:
            video = NO_DIR / f"{n}.mp4"
            for attempt in (1, 2):
                try:
                    with tempfile.TemporaryDirectory(prefix=f"testlab-no-", dir=driver.SPOOL) as tmp:
                        Path(tmp).chmod(0o750)  # detector 以 nobody+group 1000 读取检测帧
                        target = Path(tmp) / video.name
                        shutil.copy(video, target)
                        target.chmod(0o640)
                        started = time.time()
                        result = runtime.recognize([target], driver.CONTAINER_ID, assets)
                        code = result["status"]["code"]
                        items = result.get("result") if code == 2 else None
                        print(
                            f"RESULT video={n}.mp4 code={code} desc={result['status']['desc']} "
                            f"seconds={time.time() - started:.0f}s "
                            f"detail={json.dumps(items, ensure_ascii=False) if items else ''}"
                        )
                    break
                except Exception as error:
                    torch.cuda.empty_cache()
                    if attempt == 2:
                        code = -1
                        result = {"status": {"code": code, "desc": f"执行异常：{type(error).__name__}"}}
                        print(f"RESULT video={n}.mp4 code={code} desc={result['status']['desc']}")
                    else:
                        print(f"RETRY video={n}.mp4 第 1 次异常：{type(error).__name__}，清缓存重试")
            results[str(n)] = {"code": code, "desc": result["status"]["desc"]}
            save_results(results)
            torch.cuda.empty_cache()
    finally:
        runtime.close()
    print("\n===== 本批次汇总 =====")
    for n in numbers:
        row = results[str(n)]
        print(f"{n}.mp4: code={row['code']} {row['desc']}")
    counter: dict[int, int] = {}
    for row in results.values():
        counter[row["code"]] = counter.get(row["code"], 0) + 1
    print(f"\n累计 {len(results)} 个视频：{dict(sorted(counter.items()))}")


if __name__ == "__main__":
    main()
