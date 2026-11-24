"""图片规则、人工复核和 SKU 生效的兼容测试。"""

import io
import json
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from vending.config import Settings
from vending.image_review import assess, review_image
from vending.store import Store
from vending.worker import Worker


def sample_bytes(invert: bool = False) -> bytes:
    grid = (np.indices((400, 400)) // 8).sum(axis=0) % 2
    if invert:
        grid = 1 - grid
    rgb = np.repeat((grid * 180 + 30)[:, :, None], 3, axis=2).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def detection(box=None, *, confidence: float = 0.7) -> list[dict]:
    return [{"class": "snack", "confidence": confidence, "bbox": box or [80, 40, 320, 360]}]


@pytest.mark.parametrize(
    ("boxes", "decision", "code"),
    [
        (detection(), "approved", "clear_center_product"),
        (detection([0, 40, 100, 340]), "rejected", "off_center"),
        (detection(confidence=0.2), "needs_review", "weak_detection"),
        ([], "rejected", "no_product"),
    ],
)
def test_product_image_rules(boxes, decision, code):
    result = review_image(sample_bytes(), lambda _: boxes)
    assert result["decision"] == decision
    assert result["reason_code"] == code


def test_blurred_item_and_invalid_file():
    blurred = Image.new("RGB", (400, 400), "white")
    assert assess(blurred, detection())["reason_code"] == "blurred_subject"
    detector = Mock()
    assert review_image(b"bad image", detector)["decision"] == "rejected"
    detector.assert_not_called()


def test_transparent_cutout_requires_product_detection():
    cutout = Image.new("RGBA", (400, 400))
    cutout.paste((20, 60, 90, 255), (80, 40, 320, 360))
    buffer = io.BytesIO()
    cutout.save(buffer, format="PNG")
    detector = Mock(return_value=[])

    assert review_image(buffer.getvalue(), detector)["reason_code"] == "no_product"
    processed = detector.call_args.args[0]
    assert processed.mode == "RGB"
    assert processed.getpixel((0, 0)) == (255, 255, 255)


@pytest.fixture
def audit(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.initialize()
    pipeline = Mock()
    pipeline.build.return_value = b"trained"
    detector = Mock(return_value=detection())
    content = [sample_bytes()]
    monkeypatch.setattr("vending.worker.images_from_urls", lambda urls, cfg: content)
    config = Settings(data_dir=tmp_path, accounts={}, callback_delay=0)
    worker = Worker(config, store, pipeline, image_detector=detector)
    ident = store.create_sku(
        "shop",
        "test-product",
        {
            "sku_name": "测试商品",
            "img_urls": ["http://test.local/item.png"],
            "notify_url": "http://test.local/notify",
        },
        10,
    )
    return store, worker, pipeline, detector, content, ident


def test_approved_image_is_modeled_then_callback_is_produced(audit):
    store, worker, pipeline, _, _, ident = audit
    assert worker.process_job()
    pipeline.build.assert_called_once()
    assert store.audit("shop", ident)["result"]["code"] == 2
    assert json.loads(store.claim_delivery()["body"])["result"]["code"] == 2


def test_rejected_image_has_reason_and_is_not_modeled(audit):
    store, worker, pipeline, detector, _, ident = audit
    detector.return_value = detection([0, 40, 100, 340])
    assert worker.process_job()
    pipeline.build.assert_not_called()
    assert store.audit("shop", ident)["result"]["code"] == 3
    callback = json.loads(store.claim_delivery()["body"])
    assert callback["result"]["code"] == 3
    assert "第 1 张参考图" in callback["comments"]


def test_manual_hold_survives_restart_and_can_be_approved(audit):
    store, worker, pipeline, detector, _, ident = audit
    detector.return_value = detection(confidence=0.2)
    assert worker.process_job()
    pipeline.build.assert_not_called()
    store.requeue_jobs()
    assert store.audit("shop", ident)["result"] == {"code": 0, "desc": "待审核"}
    assert store.pending_image_reviews()[0]["findings"][0]["reason_code"] == "weak_detection"
    assert store.claim_delivery() is None
    assert not worker.process_job()
    store.decide_image_review(ident, "approve")
    assert worker.process_job()
    pipeline.build.assert_called_once()
    assert detector.call_count == 1
    assert store.audit("shop", ident)["result"]["code"] == 2
    assert json.loads(store.claim_delivery()["body"])["result"]["code"] == 2


def test_manual_approval_applies_only_to_the_checked_images(audit):
    store, worker, pipeline, detector, content, ident = audit
    detector.return_value = detection(confidence=0.2)
    assert worker.process_job()
    store.decide_image_review(ident, "approve")
    content[:] = [sample_bytes(invert=True)]
    assert worker.process_job()
    assert detector.call_count == 2
    pipeline.build.assert_not_called()
    assert store.audit("shop", ident)["result"]["code"] == 0
    assert store.claim_delivery() is None
    store.decide_image_review(ident, "reject", "商品放错")
    assert worker.process_job()
    assert store.audit("shop", ident) == {
        "result": {"code": 3, "desc": "审核不通过"},
        "comments": "商品放错",
    }


def test_detection_error_goes_to_review_instead_of_modeling(audit):
    store, worker, pipeline, detector, _, ident = audit
    detector.side_effect = OSError("detector unavailable")
    assert worker.process_job()
    pipeline.build.assert_not_called()
    assert store.audit("shop", ident)["result"]["code"] == 0
    assert store.pending_image_reviews()[0]["findings"][0]["reason_code"] == "detector_error"
