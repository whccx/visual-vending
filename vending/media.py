"""下载业务图片和视频，并接收预先登记的资源 ID。"""

import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests

from vending.config import Settings
from vending.protocol import ApiError, allowed_url

IMAGE_LIMIT = 24 * 1024 * 1024
VIDEO_LIMIT = 512 * 1024 * 1024


class MediaError(ValueError):
    """媒体内容无法获取或不符合接收条件。"""


def http_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def download(url: str, hosts: frozenset[str], output, limit: int) -> None:
    try:
        allowed_url(url, hosts, "url")
        with (
            http_session() as session,
            session.get(
                url,
                stream=True,
                allow_redirects=False,
                timeout=(5, 30),
                headers={"Accept-Encoding": "identity"},
            ) as response,
        ):
            if response.status_code != 200:
                raise MediaError(f"资源服务返回 HTTP {response.status_code}")
            size = 0
            for chunk in response.iter_content(128 * 1024):
                size += len(chunk)
                if size > limit:
                    raise MediaError("媒体文件过大")
                output.write(chunk)
            if size == 0:
                raise MediaError("媒体文件为空")
    except (requests.RequestException, ApiError) as exc:
        raise MediaError("媒体资源不可访问") from exc


def images_from_urls(urls: list[str], config: Settings) -> list[bytes]:
    from io import BytesIO

    images = []
    total = 0
    for url in urls:
        buffer = BytesIO()
        download(url, config.asset_hosts, buffer, IMAGE_LIMIT)
        total += buffer.tell()
        if total > 160 * 1024 * 1024:
            raise MediaError("参考图总大小超过限制")
        images.append(buffer.getvalue())
    return images


def video_files(body: dict, directory: Path, config: Settings) -> list[Path]:
    resources = body.get("resource_infos") or [{"url": url} for url in body["resource_urls"]]
    videos = []
    for index, item in enumerate(resources):
        if body["resource_type"] == 2:
            if config.resource_dir is None:
                raise MediaError("资源 ID 目录未配置")
            source = (config.resource_dir / item["resource_id"]).resolve()
            if not source.is_relative_to(config.resource_dir) or not source.is_file():
                raise MediaError("视频资源 ID 不存在")
            if not 0 < source.stat().st_size <= VIDEO_LIMIT:
                raise MediaError("视频资源为空或过大")
            destination = directory / f"video-{index}{source.suffix}"
            shutil.copyfile(source, destination)
        else:
            url = item["url"]
            suffix = Path(urlsplit(url).path).suffix.lower()
            destination = (
                directory
                / f"video-{index}{suffix if suffix in {'.mp4', '.avi', '.webm'} else '.mp4'}"
            )
            with destination.open("xb") as stream:
                download(url, config.asset_hosts, stream, VIDEO_LIMIT)
        videos.append(destination)
    return videos


def save_cover(image: bytes, digest: str, config: Settings) -> None:
    if not config.public_url:
        return
    from io import BytesIO

    from PIL import Image, ImageOps

    directory = config.data_dir / "covers"
    directory.mkdir(exist_ok=True)
    with Image.open(BytesIO(image)) as original:
        cover = ImageOps.pad(original.convert("RGB"), (300, 300), color="white")
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        name = Path(stream.name)
        try:
            cover.save(stream, format="JPEG", quality=90)
            stream.flush()
            name.replace(directory / f"{digest}.jpg")
        finally:
            name.unlink(missing_ok=True)
