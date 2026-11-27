#!/usr/bin/env python3
"""testlab 验收脚本：柜内视频应识别成功（code 2），非本柜视频应告警柜中无此SKU（code 7）。

在 visual-vending-worker 容器内运行：
    TESTLAB_FLOOR_FIRST=1 SKU_MIN_VECTOR_SCORE=... python -m scripts.testlab.verify_codes

通过环境变量调整判定顺序与阈值，与 run_videos.py 相同。全部满足退出码 0。
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import scripts.testlab.run_videos as driver

EXPECT = {
    "10-1.mp4": (2, "柜内视频应识别成功"),
    "10-2.mp4": (2, "柜内视频应识别成功"),
    "4.mp4": (7, "非本柜视频应告警柜中无此SKU"),
    "15.mp4": (7, "非本柜视频应告警柜中无此SKU"),
}
ORDER = [
    "tests/video/862337050531238/10-1.mp4",
    "tests/video/862337050531238/10-2.mp4",
    "tests/video/no/4.mp4",
    "tests/video/no/15.mp4",
]


def main() -> None:
    videos = [driver.ROOT / arg for arg in ORDER]
    missing = [path for path in videos if not path.is_file()]
    if missing:
        for path in missing:
            print(f"缺少视频：{path}")
        sys.exit(1)
    assets = driver.load_assets()
    print(f"商品库特征 {len(assets)} 个")
    results = []
    driver.SPOOL.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="testlab-verify-", dir=driver.SPOOL) as tmp:
        Path(tmp).chmod(0o750)  # detector 以 nobody+group 1000 读取检测帧
        from scripts.testlab.frame_pipeline import FrameRuntime

        runtime = FrameRuntime(1)
        try:
            for path in videos:
                target = Path(tmp) / path.name
                shutil.copy(path, target)
                target.chmod(0o640)
                started = time.time()
                result = runtime.recognize([target], driver.CONTAINER_ID, assets)
                code = result["status"]["code"]
                print(
                    f"RESULT video={path.name} code={code} desc={result['status']['desc']} "
                    f"seconds={time.time() - started:.0f}s"
                )
                print(f"DETAIL {json.dumps(result, ensure_ascii=False)}")
                results.append((path.name, code))
        finally:
            runtime.close()
    print("\n===== 验收结果 =====")
    passed = True
    for name, code in results:
        expected, reason = EXPECT[name]
        ok = code == expected
        passed &= ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: code={code} 预期={expected}（{reason}）")
    print("全部通过" if passed else "存在未达预期项")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
