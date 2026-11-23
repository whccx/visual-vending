#!/usr/bin/env python3
"""使用本地 SKU 流水线的视频识别接口。"""

import logging
import tempfile
import threading
import time
import os
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from app.catalog import product_catalog
from app.api.video_storage import store_video as _store_video
from app.api.cabinet_validation import require_known_cabinet
from pydantic import BaseModel, Field
from fastapi.openapi.utils import get_openapi
from fastapi.responses import Response, FileResponse
from urllib.parse import quote
from app.restoration import image_restore
from app.api import recognize_100doh, recognize_hoi_detr
from app.detection import area_detection, product_taken, yoloe_cutout
from app.detection.person_gate_client import scan_person_gate
from app.config import VIDEO_DIR
from app import config
from app.recognition.product_matching import accept_ranking

ROOT = config.ROOT
JOBS = config.JOBS
LOCK = threading.Lock()
LOG = logging.getLogger("sku." + __name__)


def store_video(upload, original, job, cabinet_id):
    """在指定柜子下保存视频，并只复用该柜子的相同内容。"""
    return _store_video(upload, original, job, cabinet_id, VIDEO_DIR)


@asynccontextmanager
async def lifespan(application):
    """管理服务生命周期：启动时加载常驻模型，关闭时释放资源。

    参数：
        application: FastAPI 应用实例，用于保存共享模型和客户端。
    调用：
        FastAPI 启动和关闭服务时通过 lifespan 参数调用。
    """
    from app.logging import setup_logging

    setup_logging("api")
    from app.detection.held_product_detect import load_detector, make_worker_pool
    from app.recognition.vector_compare import MODEL
    from transformers import AutoModel
    import torch

    started = time.monotonic()
    LOG.info("正在加载常驻模型")
    detector, hands = load_detector()
    pool = []
    try:
        workers = config.frame_workers()
        backend = config.hand_backend()
        pool = make_worker_pool(detector, hands, workers, backend)
        dino = AutoModel.from_pretrained(str(MODEL), local_files_only=True).eval().cuda()
    except BaseException:
        for m, h, g, stream in pool:
            if g is not None:
                g.close()
            h.close()
        if not pool:
            hands.close()
        raise
    application.state.detector = detector
    application.state.hands = hands
    application.state.workers = pool
    application.state.dino = dino
    # 其他识别接口共享人物门控客户端，并通过 LOCK 保护。
    try:
        from app.detection.person_gate_client import PersonGateClient

        application.state.person_gate = PersonGateClient()
    except Exception:
        application.state.person_gate = None
        LOG.exception("人物检测不可用，将逐帧处理")
    # GPU 手部模型只加载一次，不执行预热推理；初始化失败不影响原接口。
    try:
        from app.recognition.gpu_video_pipeline import GPUVideoPipeline
        from app.detection.held_product_detect import load_gpu_hands

        application.state.gpu_pipeline = GPUVideoPipeline(
            detector,
            dino,
            load_gpu_hands(),
            batch_size=config.gpu_batch_size(),
        )
    except Exception:
        LOG.exception(
            "实验性 GPU 后端初始化失败，现有接口仍可使用"
        )
    LOG.info(
        "常驻模型加载完成，耗时 %.3f 秒；HOI 检测使用独立服务", time.monotonic() - started
    )
    try:
        yield
    finally:
        person_gate = getattr(application.state, "person_gate", None)
        if person_gate is not None:
            person_gate.close()
        gpu_pipeline = getattr(application.state, "gpu_pipeline", None)
        if gpu_pipeline is not None:
            gpu_pipeline.hands.close()
        for m, h, g, stream in pool:
            if g is not None:
                g.close()
            h.close()
        application.state.detector = None
        application.state.hands = None
        application.state.workers = None
        application.state.dino = None
        del detector, dino
        torch.cuda.empty_cache()


app = FastAPI(
    lifespan=lifespan,
    openapi_tags=[
        {"name": "服务"},
        {"name": "商品识别"},
        {"name": "商品管理"},
        {"name": "图片处理"},
        {"name": "柜内区域"},
        {"name": "拿走检测"},
    ],
    title="云平台-开门柜商品识别",
    version="1.0.0",
    description="上传视频后复用 video 目录中内容相同的视频，不存在时保存新视频。GPU 接口使用 YOLOE-26L、商品轨迹及配置的 DINOv3 向量与颜色权重返回商品候选；轨迹点少于 5 个或前两名综合分差小于 0.03 时返回未识别。综合分不是经过概率校准的置信度。",
)


@app.middleware("http")
async def log_request(request, call_next):
    """记录非健康检查请求的结果及耗时，不记录请求数据。"""
    if request.url.path == "/health":
        return await call_next(request)
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        LOG.exception(
            "请求失败：方法=%s 耗时=%.3f 秒", request.method, time.monotonic() - started
        )
        raise
    LOG.info(
        "请求完成：方法=%s 状态码=%d 耗时=%.3f 秒",
        request.method,
        response.status_code,
        time.monotonic() - started,
    )
    return response


recognize_100doh.RESOURCE_LOCK = LOCK
app.include_router(recognize_100doh.router)
recognize_hoi_detr.RESOURCE_LOCK = LOCK
app.include_router(recognize_hoi_detr.router)


def custom_openapi():
    """生成并缓存 OpenAPI 描述，将上传字段标记为二进制格式。

    返回：
        包含所有应用路由的 OpenAPI 模式字典。
    调用：
        赋给 app.openapi，由 FastAPI 在请求接口文档时调用。
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    fields = (
        ("/products", "post", "images"),
        ("/recognize-yoloe", "post", "video"),
        ("/recognize-100doh", "post", "video"),
        ("/recognize-hoi-detr", "post", "video"),
        ("/detect-cabinet-area", "post", "video"),
        ("/detect-product-taken", "post", "video"),
        ("/images/restore", "post", "image"),
        ("/images/cutout-yoloe", "post", "image"),
    )
    for path, method, field in fields:
        body = schema["paths"][path][method]["requestBody"]["content"]["multipart/form-data"][
            "schema"
        ]
        name = body["$ref"].rsplit("/", 1)[-1]
        prop = schema["components"]["schemas"][name]["properties"][field]
        if prop.get("type") == "array":
            prop["items"]["format"] = "binary"
        else:
            prop["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


class ProductResponse(BaseModel):
    """商品目录接口的响应数据。"""

    cabinet_id: str = Field(description="柜子编码")
    id: int = Field(description="商品数字编号")
    product: str = Field(description="商品标识，例如 product_1")
    name: str = Field(description="商品名称")
    image_count: int = Field(description="商品参考图片数量")
    image_urls: List[str] = Field(default_factory=list, description="原始参考图地址")


class GPURecognitionResponse(BaseModel):
    """GPU 视频识别接口的响应数据。"""

    cabinet_id: str

    status: str
    product: Optional[str] = None
    confidence: Optional[float] = None
    vector_score: Optional[float] = None
    color_score: Optional[float] = None
    track_points: int = 0
    frames: int
    video_path: str = Field(description="项目根目录下实际使用的视频路径")
    duration: float


class AreaFrame(BaseModel):
    """单帧柜内区域检测结果。"""

    frame: int = Field(description="从 1 开始的帧号")
    time_seconds: float = Field(description="该帧时间，秒")
    polygon: Optional[List[List[int]]] = Field(
        default=None, description="按左上、右上、右下、左下顺序的四个像素坐标；未检测到为 null"
    )
    confidence: Optional[float] = None


class AreaResponse(BaseModel):
    """视频柜内区域检测的响应数据。"""

    cabinet_id: str

    status: str = Field(description="至少一帧有检测为 ok，否则为 not_found")
    video_path: str = Field(description="项目根目录下实际使用的视频路径")
    width: int
    height: int
    fps: float
    total_frames: int
    detected_frames: int
    frames: List[AreaFrame]
    duration: float


class TakenResponse(BaseModel):
    """商品是否被拿走的最小响应。"""

    taken: bool = Field(description="同一只手持物并连续离开货架时为 true")


@app.get("/catalog", include_in_schema=False)
def catalog_page():
    """同源提供动态商品库页面。"""
    return FileResponse(config.LOCAL_ROOT / "商品库.html", headers={"Cache-Control": "no-store"})


@app.get("/cabinets", tags=["商品管理"], summary="获取已登记的柜子")
def get_cabinets():
    return [{"cabinet_id": cabinet_id} for cabinet_id in product_catalog.list_cabinets()]


@app.post("/cabinets", tags=["商品管理"], summary="登记柜子", status_code=201)
def add_cabinet(cabinet_id: str = Form(..., description="柜子编码")):
    try:
        return {"cabinet_id": product_catalog.register_cabinet(cabinet_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/products/{product_id}/images/{filename}", tags=["商品管理"], summary="查看商品参考图")
def product_image(product_id: int, filename: str, cabinet_id: str):
    try:
        for image in product_catalog.reference_images(cabinet_id, product_id):
            if image.name == filename:
                return FileResponse(image, headers={"Cache-Control": "private, max-age=3600"})
        raise HTTPException(status_code=404, detail="图片不存在")
    except (product_catalog.CabinetNotFound, product_catalog.ProductNotFound) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get(
    "/products",
    response_model=List[ProductResponse],
    tags=["商品管理"],
    summary="获取当前已有商品列表",
)
def get_products(cabinet_id: str):
    """获取商品库中的商品列表；读取失败时返回 HTTP 500。"""
    try:
        products = product_catalog.list_products(cabinet_id)
        for product in products:
            files = product_catalog.reference_images(cabinet_id, product["id"])
            product["image_urls"] = [
                f"/products/{product['id']}/images/{quote(p.name, safe='')}?cabinet_id={quote(cabinet_id, safe='')}&v={p.stat().st_mtime_ns}"
                for p in files
            ]
        return products
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        LOG.exception("读取商品列表失败")
        raise HTTPException(status_code=500, detail="商品库读取失败，请查看容器日志")


@app.post(
    "/products",
    response_model=ProductResponse,
    status_code=201,
    tags=["商品管理"],
    summary="添加商品",
)
def add_product(
    cabinet_id: str = Form(..., description="柜子编码"),
    name: str = Form(..., description="商品名称"),
    images: List[UploadFile] = File(..., description="一张或多张商品参考图片"),
    product_id: Optional[int] = Form(default=None, description="商品数字编号；不填则自动分配"),
):
    """根据名称和参考图片创建商品，并在结束时关闭所有上传文件。

    参数：
        name: 商品名称。
        images: 一张或多张参考图片。
        product_id: 可选商品编号，未指定时由商品库分配。
    返回：
        商品库创建的商品信息。
    异常：
        编号重复返回 HTTP 409，输入无效返回 HTTP 400，其他错误返回 HTTP 500。
    """
    try:
        payloads = [f.file.read() for f in images]
        with LOCK:
            return product_catalog.create_product(cabinet_id, name, payloads, product_id, model=app.state.dino)
    except product_catalog.DuplicateProduct as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        LOG.exception("添加商品失败")
        raise HTTPException(status_code=500, detail="添加商品失败，请查看容器日志")
    finally:
        for f in images:
            f.file.close()


@app.delete("/products/{product_id}", tags=["商品管理"], summary="删除商品")
def remove_product(product_id: int, cabinet_id: str):
    """按数字编号删除商品；商品不存在时返回 HTTP 404。"""
    try:
        with LOCK:
            return product_catalog.delete_product(cabinet_id, product_id)
    except product_catalog.ProductNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        LOG.exception("删除商品失败")
        raise HTTPException(status_code=500, detail="删除商品失败，请查看容器日志")


@app.post(
    "/images/cutout-yoloe",
    tags=["图片处理"],
    summary="YOLOE 文本提示抠图",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "原图尺寸的透明背景 PNG"},
        400: {"description": "图片或提示词无效"},
        404: {"description": "未检测到匹配目标"},
    },
)
def cutout_yoloe(
    image: UploadFile = File(..., description="待抠图图片，最大 10 MB、2000 万像素"),
    prompt: str = Form("foreground", description="默认 foreground 自动抠出主要主体；也可指定 bottle、person 等英文目标"),
):
    """根据提示词分割图片中的主要主体，返回透明背景 PNG。"""
    try:
        data = image.file.read(yoloe_cutout.MAX_UPLOAD_BYTES + 1)
        # 输入错误应在加载模型前反馈。
        if not prompt.strip() or len(prompt.strip()) > 200:
            raise ValueError("提示词不能为空且不能超过 200 个字符")
        yoloe_cutout.decode_image(data)
        with LOCK:
            if not hasattr(app.state, "cutout_model"):
                app.state.cutout_model = yoloe_cutout.load_model()
            output = yoloe_cutout.cutout(data, prompt, app.state.cutout_model)
        return Response(content=output, media_type="image/png")
    except yoloe_cutout.TargetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        LOG.exception("YOLOE 抠图失败")
        raise HTTPException(status_code=500, detail="YOLOE 抠图失败，请查看容器日志")
    finally:
        image.file.close()


@app.post(
    "/images/restore",
    tags=["图片处理"],
    summary="GPU 商品图片修复",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "修复后的透明背景 PNG，最长边不超过 640 像素",
        }
    },
)
def restore_product_image(
    image: UploadFile = File(..., description="包含手持商品的裁剪图片，最大 10 MB"),
):
    """修复上传的商品裁剪图，返回透明背景 PNG 并关闭上传文件。"""
    try:
        return Response(
            content=image_restore.restore_image(
                image.file.read(image_restore.MAX_UPLOAD_BYTES + 1)
            ),
            media_type="image/png",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        LOG.exception("GPU 商品图片修复失败")
        raise HTTPException(status_code=503, detail="GPU 商品修复服务不可用，请查看容器日志")
    finally:
        image.file.close()


@app.get("/health", tags=["服务"])
def health():
    """返回服务存活状态，不执行模型推理。"""
    return {"status": "ok"}


@app.post(
    "/detect-cabinet-area",
    response_model=AreaResponse,
    tags=["柜内区域"],
    summary="检测视频中的柜内货架区域",
)
def detect_cabinet_area(
    cabinet_id: str = Form(..., description="柜子编码"),
    video: UploadFile = File(..., description="待检测的视频文件"),
    confidence: Optional[float] = Form(default=None, ge=0.01, le=1.0, description="检测置信度阈值，缺省用 SKU_AREA_CONF"),
):
    """逐帧检测上传视频的柜内区域，并返回区域坐标及视频信息。

    参数：
        video: 待检测视频。
        confidence: 区域检测置信度阈值。
    返回：
        包含各帧多边形、帧率和检测数量的 AreaResponse。
    """
    start = time.monotonic()
    JOBS.mkdir(parents=True, exist_ok=True)
    original = (video.filename or "").replace("\\", "/").split("/")[-1]
    if (
        not original
        or original in (".", "..")
        or any(ord(c) < 32 for c in original)
        or not Path(original).suffix
        or not Path(original).stem
    ):
        video.file.close()
        raise HTTPException(status_code=400, detail="请提供有效的视频文件名（包含扩展名）")
    require_known_cabinet(cabinet_id, video)
    with LOCK, tempfile.TemporaryDirectory(prefix="job_area_", dir=JOBS) as tmp:
        try:
            video_path = store_video(video, original, Path(tmp), cabinet_id)
            if not hasattr(app.state, "area_model"):
                try:
                    app.state.area_model = area_detection.load_model()
                except Exception:
                    LOG.exception("柜内区域模型加载失败")
                    raise HTTPException(
                        status_code=503, detail="柜内区域模型加载失败，请查看容器日志"
                    )
            result = area_detection.detect_video(
                video_path, app.state.area_model,
                config.area_confidence() if confidence is None else confidence,
            )
            frames = result["frames"]
            detected = sum(x["polygon"] is not None for x in frames)
            return AreaResponse(
                cabinet_id=cabinet_id,
                status="ok" if detected else "not_found",
                video_path=str(video_path.relative_to(ROOT)),
                width=result["width"],
                height=result["height"],
                fps=result["fps"],
                total_frames=len(frames),
                detected_frames=detected,
                frames=frames,
                duration=round(time.monotonic() - start, 3),
            )
        except HTTPException:
            raise
        except ValueError as e:
            LOG.warning("柜内区域视频输入错误：%s", e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            LOG.exception("柜内区域检测失败")
            raise HTTPException(status_code=500, detail="柜内区域检测失败，请查看容器日志")
        finally:
            video.file.close()


@app.post(
    "/detect-product-taken",
    response_model=TakenResponse,
    tags=["拿走检测"],
    summary="检测手持商品是否离开货架",
)
def detect_product_taken_api(
    cabinet_id: str = Form(..., description="柜子编码"),
    video: UploadFile = File(..., description="待检测的视频文件"),
):
    """仅在同一只手持续离架且手旁持续检测到商品时返回 true。"""
    JOBS.mkdir(parents=True, exist_ok=True)
    original = (video.filename or "").replace("\\", "/").split("/")[-1]
    if (
        not original
        or original in (".", "..")
        or any(ord(char) < 32 for char in original)
        or not Path(original).suffix
        or not Path(original).stem
    ):
        video.file.close()
        raise HTTPException(status_code=400, detail="请提供有效的视频文件名（包含扩展名）")
    require_known_cabinet(cabinet_id, video)
    with LOCK, tempfile.TemporaryDirectory(prefix="job_taken_", dir=JOBS) as tmp:
        try:
            video_path = store_video(video, original, Path(tmp), cabinet_id)
            if not hasattr(app.state, "area_model"):
                try:
                    app.state.area_model = area_detection.load_model()
                except Exception:
                    LOG.exception("柜内区域模型加载失败")
                    raise HTTPException(
                        status_code=503, detail="柜内区域模型加载失败，请查看容器日志"
                    )
            gpu_pipeline = getattr(app.state, "gpu_pipeline", None)
            gpu_hands = gpu_pipeline.hands if gpu_pipeline is not None else None
            taken = product_taken.detect_product_taken(
                video_path,
                app.state.detector,
                app.state.hands,
                app.state.area_model,
                gpu_hands,
            )
            return TakenResponse(taken=taken)
        except HTTPException:
            raise
        except ValueError as exc:
            LOG.warning("拿走检测视频输入错误：%s", exc)
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:
            LOG.exception("拿走检测失败")
            raise HTTPException(status_code=500, detail="拿走检测失败，请查看容器日志")
        finally:
            video.file.close()


@app.post(
    "/recognize-yoloe",
    response_model=GPURecognitionResponse,
    tags=["商品识别"],
    summary="GPU 视频商品识别",
)
def recognize_gpu(cabinet_id: str = Form(..., description="柜子编码"), video: UploadFile = File(..., description="待识别的视频文件")):
    """使用 GPU 管线识别视频中的商品，并返回最佳候选及其分数。

    参数：
        video: 待识别的视频文件。
    返回：
        GPURecognitionResponse；无符合阈值的候选时状态为 not_found。
    """
    start = time.monotonic()
    JOBS.mkdir(parents=True, exist_ok=True)
    original = (video.filename or "").replace("\\", "/").split("/")[-1]
    if (
        not original
        or original in (".", "..")
        or any(ord(c) < 32 for c in original)
        or not Path(original).suffix
        or not Path(original).stem
    ):
        video.file.close()
        raise HTTPException(status_code=400, detail="请提供有效的视频文件名（包含扩展名）")
    require_known_cabinet(cabinet_id, video)
    with LOCK, tempfile.TemporaryDirectory(prefix="job_", dir=JOBS) as tmp:
        try:
            video_path = store_video(video, original, Path(tmp), cabinet_id)
            if not hasattr(app.state, "gpu_pipeline"):
                from app.recognition.gpu_video_pipeline import GPUVideoPipeline
                from app.detection.held_product_detect import load_gpu_hands

                app.state.gpu_pipeline = GPUVideoPipeline(
                    app.state.detector,
                    app.state.dino,
                    load_gpu_hands(),
                    batch_size=config.gpu_batch_size(),
                )
            person_flags = scan_person_gate(app.state, video_path)
            if not hasattr(app.state, "area_model"):
                app.state.area_model = area_detection.load_model()
            result = app.state.gpu_pipeline.run(
                video_path, cabinet_id=cabinet_id, person_flags=person_flags, area_model=app.state.area_model
            )
            best = accept_ranking(result["ranking"], result["track_points"])
            return GPURecognitionResponse(
                cabinet_id=cabinet_id,
                status="ok" if best else "not_found",
                product=f"product_{best['sku']}" if best else None,
                confidence=round(best["combined"], 6) if best else None,
                vector_score=round(best["vector"], 6) if best else None,
                color_score=round(best["color"], 6) if best else None,
                track_points=result["track_points"],
                frames=result["frames"],
                video_path=str(video_path.relative_to(ROOT)),
                duration=round(time.monotonic() - start, 3),
            )
        except HTTPException:
            raise
        except ValueError as e:
            LOG.warning("GPU 视频输入错误：%s", e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            LOG.exception("GPU 识别流程失败")
            raise HTTPException(status_code=500, detail="GPU 识别失败，请查看容器日志")
        finally:
            video.file.close()
