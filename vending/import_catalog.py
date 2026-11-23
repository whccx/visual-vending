"""从已核对的清单迁入旧 ID、参考图和特征。"""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from pydantic import BaseModel, Field

from vending.config import Settings
from vending.engine import Pipeline, image_hashes, pack_features
from vending.media import images_from_urls, save_cover
from vending.schemas import Id, Name, Url
from vending.store import Store


class SkuImport(BaseModel):
    owner: str = ""
    sku_id: Id
    ks_sku_id: Id
    sku_name: Name
    barcode: str | None = None
    img_urls: list[Url] = Field(default_factory=list)
    image_files: list[Path] = Field(default_factory=list)
    aliases: list[Id] = Field(default_factory=list)
    specs: dict = Field(default_factory=dict)
    standard: bool = False
    sub_sku: list[dict] = Field(default_factory=list)
    state: int = Field(default=2, ge=2, le=3)
    comments: str = Field(default="", max_length=256)
    image_rejected: bool = False
    legacy_cabinet: str | None = None
    legacy_product: int | None = None


def old_features(path: Path, cabinet: str, product: int) -> bytes:
    from app.catalog.product_catalog import unpack

    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute(
            "SELECT features,colors,full_colors FROM products WHERE cabinet_id=? AND id=?",
            (cabinet, product),
        ).fetchone()
    if row is None:
        raise ValueError("旧库找不到明确指定的商品")
    return pack_features(*(unpack(blob) for blob in row))


def import_entry(
    store: Store,
    entry: SkuImport,
    config: Settings,
    pipeline: Pipeline,
    manifest_dir: Path,
    legacy_database: Path | None,
) -> None:
    if not entry.standard and entry.owner not in config.accounts:
        raise ValueError("私有 SKU 的 owner 必须是一个已配置的 AccessKey")
    info = entry.model_dump(
        include={"sku_name", "barcode", "img_urls", "specs", "sub_sku", "image_rejected"},
        exclude_none=True,
    )
    if not entry.sub_sku:
        info.pop("sub_sku")
    provided = [path if path.is_absolute() else manifest_dir / path for path in entry.image_files]
    features, images = None, []
    if entry.legacy_cabinet is not None or entry.legacy_product is not None:
        if not legacy_database or entry.legacy_cabinet is None or entry.legacy_product is None:
            raise ValueError("复用旧向量需要提供旧库路径、柜号及商品数字 ID")
        features = old_features(legacy_database, entry.legacy_cabinet, entry.legacy_product)
    if provided:
        images = [path.read_bytes() for path in provided]
    elif entry.state == 2 and features is None and not entry.sub_sku:
        images = images_from_urls(entry.img_urls, config)
    if entry.state == 2 and features is None and not entry.sub_sku:
        if not images:
            raise ValueError("已通过审核的商品需要参考图或明确映射的旧特征")
        features = pipeline.build(images, check_size=config.check_image_size)
    hashes = image_hashes(images) if images else None
    if hashes:
        save_cover(images[0], hashes[0], config)
    store.import_sku(
        owner="" if entry.standard else entry.owner,
        caller_id=entry.sku_id,
        sku_id=entry.ks_sku_id,
        info=info,
        features=features,
        image_hashes=hashes,
        state=entry.state,
        comments=entry.comments,
        standard=entry.standard,
        aliases=entry.aliases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="明确写明对应关系的 JSON 数组")
    parser.add_argument("--legacy-db", type=Path, help="可选旧 SQLite 文件，仅以只读模式打开")
    args = parser.parse_args()
    config = Settings.from_env()
    store = Store(config.data_dir)
    store.initialize()
    records = json.loads(args.manifest.read_text())
    entries = [SkuImport.model_validate(record) for record in records]
    pipeline = Pipeline()
    for entry in entries:
        import_entry(store, entry, config, pipeline, args.manifest.resolve().parent, args.legacy_db)
        print(f"已导入 {entry.ks_sku_id}")


if __name__ == "__main__":
    main()
