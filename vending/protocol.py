"""旷视签名、访问控制和请求错误。"""

import hashlib
import hmac
import time
from urllib.parse import parse_qsl, urlencode, urlsplit

from fastapi import Request

from vending.config import Settings

PREFIX = "/openapi/sc/v1"
MAX_BODY = 2 * 1024 * 1024
ALGORITHM = "HMAC-SHA1"


class ApiError(Exception):
    def __init__(self, status: int, msg: str, *, bare: bool = False) -> None:
        super().__init__(msg)
        self.status = status
        self.msg = msg
        self.bare = bare


def bad(field: str, reason: str) -> ApiError:
    return ApiError(400, f"BAD_ARGUMENTS:{field}:{reason}")


def canonical_query(raw: str) -> str:
    return urlencode(sorted(parse_qsl(raw, keep_blank_values=True, errors="strict")))


def authorization(
    ak: str,
    secret: str,
    host: str,
    method: str,
    path: str,
    query: str,
    date: str,
    body: bytes,
    content_type: str = "application/json",
) -> str:
    headers = f"{host.lower()};{content_type.lower()};{date}"
    canonical = "\n".join(
        (method, path, canonical_query(query), headers, hashlib.md5(body).hexdigest())
    )
    to_sign = f"{ALGORITHM}\n{hashlib.sha1(canonical.encode('utf-8')).hexdigest()}"
    signature = hmac.new(secret.encode("utf-8"), to_sign.encode("utf-8"), "sha1").hexdigest()
    return f"{ALGORITHM} {ak}:{signature}"


def authenticate(request: Request, body: bytes, settings: Settings) -> str:
    auth = request.headers.get("authorization", "")
    date = request.headers.get("x-request-date", "")
    referer = request.headers.get("referer")
    content_type = request.headers.get("content-type", "")
    error = ApiError(401, "AUTHENTICATION_ERROR", bare=True)
    if not auth:
        raise ApiError(400, "MISSING_ARGUMENTS: access_key", bare=True)
    if not auth.startswith(ALGORITHM + " "):
        raise error
    ak, separator, digest = auth[len(ALGORITHM) + 1 :].partition(":")
    account = settings.accounts.get(ak)
    if not ak or not separator or not digest:
        raise error
    if account is None or not date.isascii() or not date.isdecimal() or len(date) != 13:
        raise error
    if abs(time.time() - int(date) / 1000) > settings.signature_age:
        raise error
    if not referer:
        raise ApiError(400, "MISSING_ARGUMENTS: Referer")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise bad("Content-Type", "必须为 application/json")
    host = settings.signing_host or request.headers.get("host", "")
    raw_path = request.scope.get("raw_path") or request.url.path.encode()
    expected = authorization(
        ak,
        account.secret,
        host,
        request.method,
        raw_path.decode("ascii"),
        request.url.query,
        date,
        body,
        content_type,
    )
    if not hmac.compare_digest(expected.rsplit(":", 1)[1].encode(), digest.lower().encode()):
        raise error
    return ak


def allowed_url(raw: str, hosts: frozenset[str], field: str) -> str:
    try:
        url = urlsplit(raw)
        port = url.port
        host = url.hostname or ""
        authority = f"{host}:{port}" if port else host
    except ValueError as exc:
        raise bad(field, "URL 格式无效") from exc
    if (
        url.scheme not in {"http", "https"}
        or not host
        or url.username is not None
        or url.password is not None
        or url.fragment
        or any(char.isspace() for char in raw)
    ):
        raise bad(field, "URL 必须为有效的 HTTP 或 HTTPS 地址")
    if authority not in hosts and not (host in hosts and port in (None, 80, 443)):
        raise bad(field, "目标主机未配置")
    return raw
