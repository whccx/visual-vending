"""分阶段用 ArcFace 向量替换现有 SKU 的旧向量。"""

import argparse
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from app.config import ARCFACE_MODEL_ID
from vending.config import Settings
from vending.engine import image_hashes, pack_features, unpack_features
from vending.media import images_from_urls


def read_only(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def required_versions(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return db.execute("""
        WITH queued AS (
            SELECT r.candidates, rc.candidates AS comparisons
            FROM job j JOIN recognition r ON j.ref = r.id
            LEFT JOIN recognition_comparison rc ON rc.id = r.id
            WHERE j.kind = 'recognition' AND j.state IN ('pending', 'running')
        ), snapshots AS (
            SELECT CAST(s.key AS INTEGER) AS version FROM queued, json_each(candidates) s
            UNION
            SELECT CAST(s.key AS INTEGER) FROM queued, json_each(comparisons) s
        )
        SELECT v.id, v.body, v.features, v.image_hashes
        FROM sku_version v JOIN sku s ON v.sku = s.id
        WHERE (s.deleted = 0 AND s.active_version = v.id)
           OR v.id IN (SELECT version FROM snapshots)
        ORDER BY v.id
    """).fetchall()


def local_originals(row: sqlite3.Row, legacy_db, cabinet: str, source: Path) -> list[bytes] | None:
    import numpy as np

    from app.catalog.product_catalog import unpack

    product = re.fullmatch(r"商品(\d+)号", json.loads(row["body"]).get("sku_name", ""))
    if product is None or legacy_db is None:
        return None
    product_id = int(product[1])
    stored = legacy_db.execute(
        "SELECT features, colors, full_colors FROM products WHERE cabinet_id=? AND id=?",
        (cabinet, product_id),
    ).fetchone()
    legacy = unpack_features(row["features"])
    if stored is None or any(not np.array_equal(unpack(stored[key]), legacy[key]) for key in legacy):
        return None
    folder = source / cabinet / str(product_id)
    paths = sorted(folder.glob("image_*.png"))
    originals = [p for p in paths if re.fullmatch(r"image_\d+\.png", p.name)]
    expected_count = len(json.loads(row["image_hashes"] or "[]") or [])
    if len(originals) != expected_count or 8 * len(originals) != len(legacy["features"]):
        return None
    return [p.read_bytes() for p in originals]


def approved_images(row, settings, legacy_db, cabinet, source) -> list[bytes]:
    images = local_originals(row, legacy_db, cabinet, source)
    if images is not None:
        return images
    expected = json.loads(row["image_hashes"] or "null")
    urls = json.loads(row["body"]).get("img_urls") or []
    if not expected or not urls:
        raise ValueError(f"版本 {row['id']} 缺少可核对的已审核参考图")
    images = images_from_urls(urls, settings)
    if image_hashes(images) != expected:
        raise ValueError(f"版本 {row['id']} 的在线参考图与已审核哈希不一致")
    return images


def prepare(settings: Settings, args) -> None:
    from contextlib import nullcontext

    from vending.arcface import ArcFaceEncoder

    database = settings.data_dir / "platform.sqlite3"
    with closing(read_only(database)) as live:
        versions = required_versions(live)
    if not versions:
        raise ValueError("没有需要构建的商品向量")
    encoder = ArcFaceEncoder()
    legacy_file = args.legacy_catalog / ".sku_catalog_v2.sqlite3" if args.legacy_catalog else None
    legacy_access = closing(read_only(legacy_file)) if legacy_file and legacy_file.is_file() else nullcontext()
    with legacy_access as legacy_db, closing(sqlite3.connect(args.stage)) as stage, stage:
        stage.execute("""
            CREATE TABLE prepared (
                version INTEGER PRIMARY KEY, old_sha256 TEXT NOT NULL, features BLOB NOT NULL
            )
        """)
        for row in versions:
            original = row["features"]
            reference_images = approved_images(row, settings, legacy_db, args.legacy_cabinet, args.legacy_catalog)
            features = encoder.references(reference_images)
            legacy = unpack_features(original)
            entry = pack_features(
                features, legacy["colors"], legacy["full_colors"],
                embedding_model=ARCFACE_MODEL_ID,
            )
            unpack_features(entry, arcface=True)
            stage.execute(
                "INSERT INTO prepared VALUES (?, ?, ?)",
                (row["id"], hashlib.sha256(original).hexdigest(), entry),
            )
            print(f"已构建商品版本 {row['id']}，参考向量 {len(features)} 个", flush=True)
    print(f"已准备 {len(versions)} 个商品版本，线上商品数据未写入", flush=True)


def apply(settings: Settings, path: Path) -> None:
    with closing(read_only(path)) as staged:
        prepared = {row["version"]: row for row in staged.execute("SELECT * FROM prepared")}
    database = settings.data_dir / "platform.sqlite3"
    with closing(sqlite3.connect(database, timeout=30)) as live, live:
        live.row_factory = sqlite3.Row
        live.execute("PRAGMA secure_delete=ON")
        live.execute("BEGIN IMMEDIATE")
        if live.execute("SELECT EXISTS(SELECT 1 FROM job WHERE state='running')").fetchone()[0]:
            raise ValueError("仍有任务在运行，暂时不能切换商品特征")
        required = required_versions(live)
        if {row["id"] for row in required} != prepared.keys():
            raise ValueError("商品版本集合已经变化，请重新构建暂存向量")
        for row in required:
            entry = prepared[row["id"]]
            if row["features"] == entry["features"]:
                continue
            if hashlib.sha256(row["features"]).hexdigest() != entry["old_sha256"]:
                raise ValueError(f"商品版本 {row['id']} 已变化，请重新构建暂存向量")
            unpack_features(entry["features"], arcface=True)
            live.execute("UPDATE sku_version SET features=? WHERE id=?", (entry["features"], row["id"]))
        obsolete = [
            (row["id"],) for row in live.execute("SELECT id FROM sku_version WHERE features IS NOT NULL")
            if row["id"] not in prepared
        ]
        live.executemany("UPDATE sku_version SET features=NULL WHERE id=?", obsolete)
    print(f"已原子替换 {len(required)} 个版本的特征，并删除 {len(obsolete)} 个闲置版本的旧向量")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply"))
    parser.add_argument("stage", type=Path)
    parser.add_argument("--legacy-catalog", type=Path)
    parser.add_argument("--legacy-cabinet", default="")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.action == "prepare":
        prepare(settings, args)
    else:
        apply(settings, args.stage)


if __name__ == "__main__":
    main()
