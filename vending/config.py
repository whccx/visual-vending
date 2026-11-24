"""独立部署的上游服务配置。"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Account:
    secret: str = field(repr=False)
    sku_search: bool = False


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    accounts: dict[str, Account] = field(repr=False)
    signing_host: str = ""
    asset_hosts: frozenset[str] = frozenset()
    callback_hosts: frozenset[str] = frozenset()
    resource_dir: Path | None = None
    public_url: str = ""
    callback_delay: float = 3.0
    signature_age: int = 300
    max_pending: int = 100
    check_image_size: bool = True
    review_detector: str = "onnx"
    review_onnx_python: str = "/opt/comfy-conda/bin/python"

    @classmethod
    def from_env(cls) -> "Settings":
        config_file = os.environ.get("VENDING_ACCOUNTS_FILE")
        configured = json.loads(Path(config_file).read_text()) if config_file else {}
        accounts = {name: Account(**fields) for name, fields in configured.items()}
        if any(not account.secret for account in accounts.values()):
            raise ValueError("上游账户必须配置有效密钥")
        data_dir = Path(os.environ.get("VENDING_DATA_DIR", ROOT / "data")).resolve()
        return cls(
            data_dir=data_dir,
            accounts=accounts,
            signing_host=os.environ.get("VENDING_SIGNING_HOST", "").lower(),
            asset_hosts=read_hosts("VENDING_ASSET_HOSTS"),
            callback_hosts=read_hosts("VENDING_CALLBACK_HOSTS"),
            resource_dir=Path(
                os.environ.get("VENDING_RESOURCE_DIR", data_dir / "resources")
            ).resolve(),
            public_url=os.environ.get("VENDING_PUBLIC_URL", "").rstrip("/"),
            callback_delay=float(os.environ.get("VENDING_CALLBACK_DELAY", "3")),
            max_pending=int(os.environ.get("VENDING_MAX_PENDING", "100")),
            check_image_size=os.environ.get("VENDING_CHECK_IMAGE_SIZE", "1") == "1",
            review_detector=os.environ.get("VENDING_REVIEW_DETECTOR", "onnx").strip().lower(),
            review_onnx_python=os.environ.get(
                "VENDING_REVIEW_ONNX_PYTHON", "/opt/comfy-conda/bin/python"
            ),
        )


def read_hosts(name: str) -> frozenset[str]:
    return frozenset(
        host.strip().lower() for host in os.environ.get(name, "").split(",") if host.strip()
    )
