"""持久化商品目录及 DINOv3 参考特征。"""

import io
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
import re
from pathlib import Path

import cv2
import numpy as np
from app import config

ROOT = config.ROOT
PRODUCT_DIR = config.product_dir()
DB_PATH = PRODUCT_DIR / ".sku_catalog_v2.sqlite3"
MODEL = config.MODEL


class DuplicateProduct(ValueError):
    """商品编号或对应目录已存在。"""


class CabinetNotFound(ValueError):
    """柜子编码尚未登记。"""


class ProductNotFound(ValueError):
    """请求的商品编号不存在。"""


def hist(img, alpha=None):
    """计算有效像素的平滑 HSV 色相直方图。

    参数：
        img: BGR 图像；alpha: 可选透明度通道，低于 128 的像素会排除。
    返回：
        36 维色相直方图，有有效像素时归一化为总和 1。
    """
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    mask = (s >= 60) & (v >= 45)
    if alpha is not None:
        mask &= alpha >= 128
    q = np.bincount((h[mask] // 5).astype(np.int32), minlength=36).astype(float)
    q = (np.roll(q, 2) + 4 * np.roll(q, 1) + 6 * q + 4 * np.roll(q, -1) + np.roll(q, -2)) / 16
    if q.sum():
        q /= q.sum()
    return q


def pack(array):
    """将数组保存为不使用 pickle 的 NumPy 格式字节。"""
    f = io.BytesIO()
    np.save(f, np.asarray(array), allow_pickle=False)
    return f.getvalue()


def unpack(blob):
    """从 NumPy 格式字节恢复数组，不允许加载 pickle 对象。"""
    return np.load(io.BytesIO(blob), allow_pickle=False)


def connect():
    """建立商品库 SQLite 连接，启用外键并以具名行返回查询结果。"""
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def validate_cabinet_id(cabinet_id):
    """校验柜号，避免路径穿越和不同柜号意外混用。"""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", cabinet_id or ""):
        raise ValueError("柜子编码只能包含字母、数字、下划线和连字符")
    return cabinet_id


def initialize():
    """创建按柜号和商品编号隔离的商品库。"""
    with connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS products (cabinet_id TEXT NOT NULL, id INTEGER NOT NULL, name TEXT NOT NULL, image_count INTEGER NOT NULL, features BLOB NOT NULL, colors BLOB NOT NULL, full_colors BLOB NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(cabinet_id, id))")
        db.execute("CREATE TABLE IF NOT EXISTS cabinets (cabinet_id TEXT PRIMARY KEY)")
        db.execute("INSERT OR IGNORE INTO cabinets SELECT DISTINCT cabinet_id FROM products")


def list_cabinets():
    """列出已登记柜子，包括目前没有商品的柜子。"""
    initialize()
    with connect() as db:
        return [r[0] for r in db.execute("SELECT cabinet_id FROM cabinets ORDER BY cabinet_id")]


def register_cabinet(cabinet_id):
    """登记柜子，不影响已存在的商品。"""
    validate_cabinet_id(cabinet_id)
    initialize()
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO cabinets (cabinet_id) VALUES (?)", (cabinet_id,))
    return cabinet_id


def require_cabinet(cabinet_id):
    """柜子未登记时拒绝检测，且不创建视频目录。"""
    validate_cabinet_id(cabinet_id)
    initialize()
    with connect() as db:
        if not db.execute("SELECT 1 FROM cabinets WHERE cabinet_id=?", (cabinet_id,)).fetchone():
            raise CabinetNotFound("没有该柜子，请先登记柜子或入库商品")
    return cabinet_id


def reference_images(cabinet_id, product_id):
    """返回数据库中商品对应的原始参考图。"""
    require_cabinet(cabinet_id)
    with connect() as db:
        if not db.execute("SELECT 1 FROM products WHERE cabinet_id=? AND id=?", (cabinet_id, product_id)).fetchone():
            raise ProductNotFound("商品不存在")
    cabinet = PRODUCT_DIR / cabinet_id
    directory = cabinet / str(product_id)
    if cabinet.is_symlink() or directory.is_symlink() or not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.is_symlink()
                  and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                  and "_scale_" not in p.stem and not p.stem.endswith("_cw90"))


def list_products(cabinet_id):
    validate_cabinet_id(cabinet_id)
    initialize()
    with connect() as db:
        return [{"cabinet_id": cabinet_id, "id": r["id"], "product": f"product_{r['id']}", "name": r["name"], "image_count": r["image_count"]} for r in db.execute("SELECT id,name,image_count FROM products WHERE cabinet_id=? ORDER BY id", (cabinet_id,))]


def load_features(cabinet_id):
    validate_cabinet_id(cabinet_id)
    initialize()
    with connect() as db:
        rows = db.execute("SELECT id,features,colors,full_colors FROM products WHERE cabinet_id=? ORDER BY id", (cabinet_id,)).fetchall()
    return {int(r["id"]): {"features": unpack(r["features"]), "colors": unpack(r["colors"]), "full_colors": unpack(r["full_colors"])} for r in rows}


def decode(data):
    """解码上传图片并生成黑色背景合成图及可见像素色相直方图。

    参数：
        data: 编码后的图片字节。
    返回：
        原图、BGR 合成图和颜色直方图组成的元组。
    异常：
        图片无法读取或通道格式不受支持时抛出 ValueError。
    """
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法读取图片，请上传有效的 JPEG、PNG 或 WebP 图片")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("不支持该图片格式")

    alpha = image[:, :, 3] if image.shape[2] == 4 else None
    orig = image[:, :, :3]
    composite = (
        (orig.astype(np.float32) * (alpha[:, :, None].astype(np.float32) / 255)).astype(np.uint8)
        if alpha is not None
        else orig
    )
    return image, composite, hist(orig, alpha)


def image_variants(image):
    """依次生成原图及四种尺度下的顺时针旋转变体，并附带文件名后缀。"""
    h, w = image.shape[:2]
    for divisor in (1, 2, 3, 4):
        scaled = (
            image
            if divisor == 1
            else cv2.resize(
                image,
                (max(1, round(w / divisor)), max(1, round(h / divisor))),
                interpolation=cv2.INTER_AREA,
            )
        )
        suffix = "" if divisor == 1 else f"_scale_1_{divisor}"
        yield suffix, scaled
        yield suffix + "_cw90", cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)


def features_for_images(payloads, model=None):
    """为上传图片及其缩放、旋转变体计算 DINO 向量和原图颜色特征。

    参数：
        payloads: 图片字节序列。
        model: 可选的已加载模型；未指定时从本地加载并在结束后释放。
    返回：
        float32 特征矩阵与 float64 颜色直方图数组。
    异常：
        CUDA 不可用或图片无效时抛出异常。
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    colors = []
    for data in payloads:
        _, _, color = decode(data)
        colors.append(color)

    owned_model = model is None
    if owned_model:
        model = AutoModel.from_pretrained(str(MODEL), local_files_only=True).eval().cuda()
    torch.set_num_threads(4)
    features = []

    with torch.inference_mode():
        for data in payloads:
            image, _, _ = decode(data)
            aug = []
            for _, variant in image_variants(image):
                orig = variant[:, :, :3]
                alpha = variant[:, :, 3] if variant.shape[2] == 4 else None
                composite = (
                    (orig.astype(np.float32) * (alpha[:, :, None].astype(np.float32) / 255)).astype(
                        np.uint8
                    )
                    if alpha is not None
                    else orig
                )
                z = cv2.resize(composite, (224, 224), interpolation=cv2.INTER_LINEAR)
                z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                z = (z - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                    [0.229, 0.224, 0.225], np.float32
                )
                aug.append(torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1))))

            result = (
                F.normalize(model(pixel_values=torch.stack(aug).cuda()).pooler_output, dim=1)
                .cpu()
                .numpy()
            )
            features.extend(result)

    if owned_model:
        del model
        torch.cuda.empty_cache()
    return np.asarray(features, dtype=np.float32), np.asarray(colors, dtype=np.float64)


def create_product(cabinet_id, name, payloads, product_id=None, model=None, existing=False):
    """创建商品，existing=True 时使用现有目录中的原图且不改写文件。"""
    from app.recognition.full_color import from_bgra

    validate_cabinet_id(cabinet_id)
    initialize()
    if not name.strip() or not payloads:
        raise ValueError("商品名称和参考图片不能为空")
    cabinet = PRODUCT_DIR / cabinet_id
    cabinet.mkdir(parents=True, exist_ok=True)
    if cabinet.is_symlink():
        raise ValueError("柜子目录不能是符号链接")
    with connect() as db:
        if product_id is None:
            used = {int(r[0]) for r in db.execute("SELECT id FROM products WHERE cabinet_id=?", (cabinet_id,))}
            used.update(int(x.name) for x in cabinet.iterdir() if x.is_dir() and x.name.isdigit())
            product_id = max(used, default=0) + 1
        if product_id <= 0:
            raise ValueError("商品编号必须大于 0")
        target = cabinet / str(product_id)
        if db.execute("SELECT 1 FROM products WHERE cabinet_id=? AND id=?", (cabinet_id, product_id)).fetchone() or (target.exists() and not existing):
            raise DuplicateProduct("该柜子中的商品编号已存在")
        if existing and (not target.is_dir() or target.is_symlink()):
            raise ValueError("现有商品目录无效")
    full_colors = [from_bgra(decode(data)[0]) for data in payloads]
    features, colors = features_for_images(payloads, model=model)
    staging = None
    moved = False
    try:
        if not existing:
            staging = Path(tempfile.mkdtemp(prefix=".product_", dir=cabinet))
            for i, data in enumerate(payloads):
                decoded, _, _ = decode(data)
                for suffix, variant in image_variants(decoded):
                    ok, encoded = cv2.imencode(".png", variant)
                    if not ok:
                        raise RuntimeError("保存商品图片失败")
                    (staging / f"image_{i}{suffix}.png").write_bytes(encoded.tobytes())
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM products WHERE cabinet_id=? AND id=?", (cabinet_id, product_id)).fetchone():
                raise DuplicateProduct("该柜子中的商品编号已存在")
            if not existing:
                if target.exists() or target.is_symlink():
                    raise DuplicateProduct("该柜子中的商品编号已存在")
                os.rename(staging, target)
                moved = True
            db.execute("INSERT OR IGNORE INTO cabinets (cabinet_id) VALUES (?)", (cabinet_id,))
            db.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", (cabinet_id, product_id, name.strip(), len(payloads), pack(features), pack(colors), pack(np.asarray(full_colors, dtype=np.float32)), time.strftime("%Y-%m-%dT%H:%M:%S%z")))
            db.commit()
    except BaseException:
        if moved:
            shutil.rmtree(target)
        if staging and staging.exists():
            shutil.rmtree(staging)
        raise
    return {"cabinet_id": cabinet_id, "id": product_id, "product": f"product_{product_id}", "name": name.strip(), "image_count": len(payloads)}


def delete_product(cabinet_id, product_id):
    validate_cabinet_id(cabinet_id)
    initialize()
    target = PRODUCT_DIR / cabinet_id / str(product_id)
    trash = PRODUCT_DIR / ".deleted" / cabinet_id
    moved = None
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if not db.execute("SELECT 1 FROM products WHERE cabinet_id=? AND id=?", (cabinet_id, product_id)).fetchone():
            raise ProductNotFound("商品不存在")
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError("商品目录异常，拒绝删除")
            trash.mkdir(parents=True, exist_ok=True)
            moved = trash / f"{product_id}_{uuid.uuid4().hex}"
            os.rename(target, moved)
        try:
            db.execute("DELETE FROM products WHERE cabinet_id=? AND id=?", (cabinet_id, product_id))
            db.commit()
        except BaseException:
            if moved:
                os.rename(moved, target)
            raise
    return {"cabinet_id": cabinet_id, "id": product_id, "product": f"product_{product_id}", "deleted": True}
