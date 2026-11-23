"""对接字段及文档长度约束。"""

import json
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    StrictInt,
    StringConstraints,
    model_validator,
)

Id = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=32)]
Name = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=50)
]
SafeId = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]{1,32}$")]
Url = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=2048)]


def json_field(limit: int) -> BeforeValidator:
    def parse(value):
        encoded = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        if len(encoded.encode("utf-8")) > limit:
            raise ValueError(f"JSON 字段超过 {limit} 字节")
        return json.loads(value) if isinstance(value, str) else value

    return BeforeValidator(parse)


class Specs(BaseModel):
    taste: str | None = None
    capacity: str | None = None
    weight: str | None = None


class SkuEdit(BaseModel):
    sku_name: Name
    barcode: Annotated[str, StringConstraints(strict=True, max_length=200)] | None = None
    img_urls: Annotated[list[Url], Field(min_length=1, max_length=30), json_field(2000)] | None = (
        None
    )
    notify_url: Annotated[Url, StringConstraints(max_length=128)] | None = None
    specs: Annotated[Specs, json_field(2000)] | None = None


class SkuCreate(SkuEdit):
    sku_id: Id
    img_urls: Annotated[list[Url], Field(min_length=1, max_length=30), json_field(5000)]


class SkuScope(BaseModel):
    sku_id: Id
    floor_num: Annotated[StrictInt, Field(gt=0)] | None = None


class ExtraParams(BaseModel):
    origin_time: StrictInt | str
    sku_scope: Annotated[list[SkuScope], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_origin_time(self):
        value = str(self.origin_time)
        if not value.isascii() or not value.isdecimal() or len(value) != 13:
            raise ValueError("origin_time 必须是毫秒时间戳")
        return self


class ResourceInfo(BaseModel):
    url: Url | None = None
    resource_id: Annotated[str, StringConstraints(strict=True, max_length=128)] | None = None
    camera_id: Annotated[str, StringConstraints(strict=True, max_length=64)] | None = None


class RecognitionCreate(BaseModel):
    recog_id: Id
    extra_params: Annotated[ExtraParams, json_field(4096)]
    container_id: SafeId
    resource_type: Annotated[StrictInt, Field(ge=1, le=2)] = 1
    resource_urls: Annotated[list[Url], Field(min_length=1), json_field(2048)] | None = None
    resource_infos: Annotated[list[ResourceInfo], Field(min_length=1), json_field(2048)] | None = (
        None
    )
    notify_url: Url | None = None
    sub_mch_id: Annotated[str, StringConstraints(strict=True, max_length=128)] | None = None


class ErrorResponse(BaseModel):
    msg: str
    request_id: str | None = None
    cost: int | None = None


class SubmittedSku(BaseModel):
    ks_sku_id: Id


class CallerSku(BaseModel):
    sku_id: Id


class SubmittedRecognition(BaseModel):
    ks_recog_id: Id


class CodeDescription(BaseModel):
    code: int
    desc: str


class AuditResult(BaseModel):
    result: CodeDescription
    comments: str | None = None


class SkuAuditCallback(AuditResult):
    sku_id: Id
    ks_sku_id: Id


class SkuItem(BaseModel):
    ks_sku_id: Id
    sku_num: int


class RecognitionResult(BaseModel):
    status: CodeDescription
    result: list[SkuItem] | None = None
    comments: str | None = None


class RecognitionStatus(BaseModel):
    state: CodeDescription
    result: RecognitionResult


class RecognitionCallback(RecognitionStatus):
    ks_recog_id: Id


class CallbackAcknowledgment(BaseModel):
    code: Literal["10000"]
    msg: Literal["请求成功"]
    data: dict


class CatalogSku(BaseModel):
    ks_sku_id: Id
    barcode: str | None = None
    sku_name: Name
    sku_id: Id
    img_urls: list[str] | None = None
    cover: str
    sku_type: Literal["0", "1"]
    specs: dict
    has_sub_sku: bool
    sub_sku: list[dict]


class CatalogPage(BaseModel):
    page_num: int
    page_size: int
    total: int
    resultList: list[CatalogSku]


DataT = TypeVar("DataT")


class Envelope(BaseModel, Generic[DataT]):
    request_id: str
    cost: int
    data: DataT
