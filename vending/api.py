"""兼容旷视 OpenAPI 的 FastAPI 入口。"""

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from vending.config import Settings
from vending.protocol import MAX_BODY, PREFIX, ApiError, allowed_url, authenticate, bad
from vending.schemas import (
    AuditResult,
    CallbackAcknowledgment,
    CallerSku,
    CatalogPage,
    CatalogSku,
    Envelope,
    ErrorResponse,
    Id,
    RecognitionCallback,
    RecognitionCreate,
    RecognitionStatus,
    SkuAuditCallback,
    SkuCreate,
    SkuEdit,
    SubmittedRecognition,
    SubmittedSku,
)
from vending.store import Store

LOG = logging.getLogger("vending.api")
router = APIRouter(
    prefix=PREFIX,
    responses={
        400: {"model": ErrorResponse, "description": "参数不符合契约"},
        401: {"model": ErrorResponse, "description": "签名校验失败"},
        403: {"model": ErrorResponse, "description": "访问或并发受限"},
        404: {"model": ErrorResponse, "description": "路径不存在"},
        413: {"content": {"text/plain": {}}, "description": "请求体超过 2 MB"},
        500: {"model": ErrorResponse, "description": "服务内部错误"},
    },
)
PathID = Annotated[Id, Path()]
SAFE_RESOURCE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")


def is_openapi(path: str) -> bool:
    return path == PREFIX or path.startswith(PREFIX + "/")


def envelope(request: Request, **fields) -> dict:
    return {
        "request_id": request.state.request_id,
        "cost": round((time.monotonic() - request.state.started) * 1000),
        **fields,
    }


def failure(request: Request, error: ApiError):
    body = {"msg": error.msg}
    return JSONResponse(body if error.bare else envelope(request, **body), status_code=error.status)


def principal(request: Request) -> str:
    return request.state.owner


def store(request: Request) -> Store:
    return request.app.state.store


def settings(request: Request) -> Settings:
    return request.app.state.settings


Owner = Annotated[str, Depends(principal)]
Database = Annotated[Store, Depends(store)]
Configuration = Annotated[Settings, Depends(settings)]


def checked_sku(payload: SkuEdit, config: Settings) -> dict:
    body = payload.model_dump(exclude_unset=True, exclude_none=True, exclude={"sku_id"})
    if payload.img_urls:
        for url in payload.img_urls:
            allowed_url(url, config.asset_hosts, "img_urls")
    if payload.notify_url:
        allowed_url(payload.notify_url, config.callback_hosts, "notify_url")
    return body


def validate_resources(data: RecognitionCreate, config: Settings) -> None:
    if data.notify_url:
        allowed_url(data.notify_url, config.callback_hosts, "notify_url")
    if data.resource_urls and data.resource_infos:
        raise ApiError(400, "COEXISTENCE_ARGUMENTS")
    if not data.resource_urls and not data.resource_infos:
        raise ApiError(400, "MISSING_ARGUMENTS: resource_urls")
    if data.resource_urls:
        if data.resource_type != 1:
            raise bad("resource_type", "资源 ID 需通过 resource_infos 提交")
        for url in data.resource_urls:
            allowed_url(url, config.asset_hosts, "resource_urls")
        return
    for item in data.resource_infos or ():
        if data.resource_type == 1:
            if not item.url:
                raise ApiError(400, "MISSING_ARGUMENTS: resource_infos.url")
            allowed_url(item.url, config.asset_hosts, "resource_infos.url")
        elif not item.resource_id or SAFE_RESOURCE.fullmatch(item.resource_id) is None:
            raise bad("resource_infos.resource_id", "资源 ID 无效")


@router.post("/sku", response_model=Envelope[SubmittedSku], response_model_exclude_unset=True)
@router.post("/sku/", response_model=Envelope[SubmittedSku], include_in_schema=False)
def create_sku(
    request: Request,
    payload: SkuCreate,
    owner: Owner,
    db: Database,
    config: Configuration,
):
    sku_id = db.create_sku(owner, payload.sku_id, checked_sku(payload, config), config.max_pending)
    return envelope(request, data={"ks_sku_id": sku_id})


@router.get(
    "/sku/sku_list", response_model=Envelope[CatalogPage], response_model_exclude_unset=True
)
def sku_list(
    request: Request,
    owner: Owner,
    db: Database,
    config: Configuration,
    search: Annotated[str, Query(max_length=50)] = "",
    only_standard: bool = False,
    only_single: bool = False,
    page_num: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=50)] = 10,
):
    if not config.accounts[owner].sku_search:
        raise ApiError(403, "ACCESS_FORBIDDEN", bare=True)
    data = db.products(
        owner,
        search=search,
        only_standard=only_standard,
        only_single=only_single,
        page_num=page_num,
        page_size=page_size,
    )
    return envelope(request, data=data)


@router.get(
    "/sku/all_sku/{barcode}",
    response_model=Envelope[list[CatalogSku]],
    response_model_exclude_unset=True,
)
def find_barcode(request: Request, barcode: PathID, owner: Owner, db: Database):
    return envelope(request, data=db.products(owner, barcode=barcode))


@router.get(
    "/sku/skus/{ks_sku_id}", response_model=Envelope[CatalogSku], response_model_exclude_unset=True
)
def get_product(request: Request, ks_sku_id: PathID, owner: Owner, db: Database):
    return envelope(request, data=db.product(owner, ks_sku_id))


@router.get(
    "/sku/{ks_sku_id}", response_model=Envelope[AuditResult], response_model_exclude_unset=True
)
def get_audit(request: Request, ks_sku_id: PathID, owner: Owner, db: Database):
    return envelope(request, data=db.audit(owner, ks_sku_id))


@router.patch(
    "/sku/{ks_sku_id}", response_model=Envelope[CallerSku], response_model_exclude_unset=True
)
def edit_sku(
    request: Request,
    ks_sku_id: PathID,
    payload: SkuEdit,
    owner: Owner,
    db: Database,
    config: Configuration,
):
    sku_id = db.edit_sku(owner, ks_sku_id, checked_sku(payload, config), config.max_pending)
    return envelope(request, data={"sku_id": sku_id})


@router.delete(
    "/sku/{ks_sku_id}", response_model=Envelope[CallerSku], response_model_exclude_unset=True
)
def delete_sku(request: Request, ks_sku_id: PathID, owner: Owner, db: Database):
    return envelope(request, data={"sku_id": db.delete_sku(owner, ks_sku_id)})


@router.post(
    "/recognition", response_model=Envelope[SubmittedRecognition], response_model_exclude_unset=True
)
def submit_recognition(
    request: Request,
    payload: RecognitionCreate,
    owner: Owner,
    db: Database,
    config: Configuration,
):
    validate_resources(payload, config)
    recog_id = db.submit(owner, payload.model_dump(exclude_none=True), config.max_pending)
    return envelope(request, data={"ks_recog_id": recog_id})


@router.get(
    "/recognition/{ks_recog_id}",
    response_model=Envelope[RecognitionStatus],
    response_model_exclude_unset=True,
)
def get_recognition(
    request: Request,
    ks_recog_id: PathID,
    owner: Owner,
    db: Database,
):
    return envelope(request, data=db.recognition(owner, ks_recog_id))


class VerifyRequests:
    def __init__(self, app, config: Settings) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not is_openapi(scope["path"]):
            return await self.app(scope, receive, send)
        request = Request(scope, receive)
        request.state.started = time.monotonic()
        supplied_id = request.headers.get("x-request-id", "")
        request.state.request_id = supplied_id[:128] if supplied_id else uuid.uuid4().hex
        try:
            if request.url.scheme != "https":
                raise ApiError(403, "ACCESS_FORBIDDEN", bare=True)
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BODY:
                    return await PlainTextResponse("REQUEST ENTITY TOO LARGE", status_code=413)(
                        scope,
                        receive,
                        send,
                    )
                body.extend(chunk)
            payload = bytes(body)
            request.state.owner = authenticate(request, payload, self.config)
        except ClientDisconnect:
            return await failure(request, bad("body", "请求已断开"))(scope, receive, send)
        except ApiError as error:
            return await failure(request, error)(scope, receive, send)
        except (UnicodeError, ValueError):
            return await failure(request, bad("request", "请求编码无效"))(scope, receive, send)

        replayed = False

        async def replay():
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": payload, "more_body": False}

        return await self.app(scope, replay, send)


def create_app(config: Settings | None = None) -> FastAPI:
    config = config or Settings.from_env()
    db = Store(config.data_dir, config.public_url)

    @asynccontextmanager
    async def lifespan(app):
        db.initialize()
        yield

    app = FastAPI(
        title="Visual Vending OpenAPI",
        version="2026.09.22",
        lifespan=lifespan,
        redirect_slashes=False,
    )
    app.state.settings = config
    app.state.store = db

    @app.get("/health", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @app.get("/covers/{digest}.jpg", include_in_schema=False)
    def cover(digest: Annotated[str, Path(pattern=r"^[a-f0-9]{64}$")]):
        path = config.data_dir / "covers" / f"{digest}.jpg"
        if not config.public_url or not path.is_file():
            raise HTTPException(404, "封面不存在")
        return FileResponse(path, media_type="image/jpeg")

    app.add_middleware(VerifyRequests, config=config)

    @app.exception_handler(ApiError)
    async def api_error(request, error):
        return failure(request, error)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        first = error.errors()[0]
        path = [str(part) for part in first["loc"] if part not in {"path", "query", "body"}]
        key = ".".join(path) or "body"
        if first["type"] == "missing":
            return failure(request, ApiError(400, f"MISSING_ARGUMENTS: {key}"))
        return failure(request, bad(key, first["msg"]))

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        if is_openapi(request.url.path):
            return failure(request, ApiError(404, "API_NOT_FOUND", bare=True))
        return JSONResponse({"detail": error.detail}, status_code=error.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        LOG.error("上游接口处理失败", exc_info=error)
        if is_openapi(request.url.path):
            return failure(request, ApiError(500, "INTERNAL_ERROR"))
        return JSONResponse({"detail": "INTERNAL_ERROR"}, status_code=500)

    @app.webhooks.post("skuAuditResult", response_model=CallbackAcknowledgment)
    def sku_audit_result(body: SkuAuditCallback):
        """平台向提交 SKU 时传入的 notify_url 发送审核结果。"""

    @app.webhooks.post("recognitionResult", response_model=CallbackAcknowledgment)
    def recognition_result(body: RecognitionCallback):
        """平台向提交识别时传入的 notify_url 发送识别结果。"""

    app.include_router(router)
    native_openapi = app.openapi

    def contract_openapi():
        schema = native_openapi()
        components = schema["components"]
        components["securitySchemes"] = {
            "SignedRequest": {
                "type": "apiKey",
                "in": "header",
                "name": "Authorization",
                "description": "HMAC-SHA1 <AccessKey>:<Signature>",
            },
        }
        components["parameters"] = {
            key: {
                "name": key,
                "in": "header",
                "required": required,
                "schema": {"type": "string"},
                "description": description,
            }
            for key, required, description in (
                ("X-Request-Date", True, "UTC 毫秒时间戳，参与签名"),
                ("Referer", True, "请求来源域名"),
                ("X-Request-Id", False, "调用方申请 ID，缺省时由服务生成"),
            )
        }
        for routes in (schema["paths"], schema["webhooks"]):
            for route in routes.values():
                for operation in route.values():
                    operation["responses"].pop("422", None)
                    operation["security"] = [{"SignedRequest": []}]
                    parameters = operation.setdefault("parameters", [])
                    for key in components["parameters"]:
                        ref = {"$ref": f"#/components/parameters/{key}"}
                        if ref not in parameters:
                            parameters.append(ref)
        return schema

    app.openapi = contract_openapi
    return app


app = create_app()
