"""按柜子隔离保存视频，跨进程安全地去重和分配文件名。"""

import fcntl
import hashlib
import os
import shutil
from pathlib import Path

from app.catalog.product_catalog import validate_cabinet_id


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def store_video(upload, original, job, cabinet_id, video_root):
    """仅在指定柜子内查找相同内容，避免并发请求覆盖视频。"""
    validate_cabinet_id(cabinet_id)
    video_root = Path(video_root)
    cabinet_dir = video_root / cabinet_id
    if cabinet_dir.is_symlink():
        raise ValueError("柜子目录不能是符号链接")
    cabinet_dir.mkdir(parents=True, exist_ok=True)
    name = Path(original)
    temporary = job / ("upload" + name.suffix)
    with temporary.open("xb") as target:
        shutil.copyfileobj(upload.file, target, length=1024 * 1024)
    size = temporary.stat().st_size
    if not size:
        raise ValueError("上传的视频为空")
    checksum = _digest(temporary)
    # 使用同一把文件锁覆盖查重、命名、移动，支持多个服务进程。
    with (video_root / ".video_storage.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for path in sorted(cabinet_dir.iterdir()):
                if (path.is_file() and not path.is_symlink()
                        and path.stat().st_size == size and _digest(path) == checksum):
                    return path
            index = 0
            while True:
                stem = name.stem if index == 0 else f"{name.stem}_{index}"
                path = cabinet_dir / (stem + name.suffix)
                if not path.exists() and not path.is_symlink() and not (cabinet_dir / stem).exists():
                    os.rename(temporary, path)
                    return path
                index += 1
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
