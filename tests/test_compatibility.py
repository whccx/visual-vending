"""兼容协议与持久任务的回归用例。"""

import json
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from vending.api import create_app
from vending.config import Account, Settings
from vending.protocol import MAX_BODY, PREFIX, authorization
from vending.store import outcome
from vending.worker import ACK, Worker


@pytest.fixture
def context(tmp_path):
    config = Settings(
        data_dir=tmp_path,
        accounts={"demo": Account("test-only", True), "other": Account("other")},
        asset_hosts=frozenset({"assets.example"}),
        callback_hosts=frozenset({"java.example"}),
        resource_dir=tmp_path,
        callback_delay=0,
    )
    app = create_app(config)
    with TestClient(app, base_url="https://svc.example") as client:
        yield client, app.state.store, config


def signed(client, method, url, data=None, *, account="demo", secret="test-only"):
    path = url.partition("?")
    payload = json.dumps(data, ensure_ascii=False).encode() if data is not None else b""
    date = str(time.time_ns() // 1_000_000)
    return client.request(
        method,
        url,
        content=payload,
        headers={
            "Content-Type": "application/json",
            "X-Request-Date": date,
            "Referer": "https://svc.example",
            "Authorization": authorization(
                account, secret, "svc.example", method, path[0], path[2], date, payload
            ),
        },
    )


def sku_body():
    return {
        "sku_id": "vision-mock",
        "sku_name": "冰红茶",
        "barcode": "6920000000000",
        "img_urls": ["https://assets.example/tea.jpg"],
        "notify_url": "https://java.example/notifySku",
    }


def task_body(sku_id):
    return {
        "recog_id": "TE-Vision-01",
        "container_id": "container-a",
        "extra_params": {"origin_time": 1770000000000, "sku_scope": [{"sku_id": sku_id}]},
        "resource_urls": ["https://assets.example/order.mp4"],
        "notify_url": "https://java.example/notifyRecog",
    }


def test_authenticated_routes_and_error_envelope(context):
    client, _, _ = context
    assert client.post(PREFIX + "/sku", json={}).json() == {"msg": "MISSING_ARGUMENTS: access_key"}
    assert signed(client, "POST", PREFIX + "/sku", {}, secret="wrong").status_code == 401

    missing = signed(client, "POST", PREFIX + "/sku", {})
    assert missing.status_code == 400 and missing.json()["msg"].startswith("MISSING_ARGUMENTS")
    unknown = signed(client, "GET", PREFIX + "/nope")
    assert unknown.status_code == 404 and unknown.json() == {"msg": "API_NOT_FOUND"}
    large = client.post(PREFIX + "/sku", content=b"x" * (MAX_BODY + 1))
    assert large.status_code == 413 and large.text == "REQUEST ENTITY TOO LARGE"


def test_sku_lifecycle_and_outbox(context, monkeypatch):
    client, db, config = context
    monkeypatch.setattr("vending.worker.images_from_urls", lambda urls, cfg: [urls[0].encode()])
    monkeypatch.setattr("vending.worker.check_reference_images", lambda *_: None)
    pipeline = Mock()
    pipeline.build.return_value = b"features"
    worker = Worker(config, db, pipeline)
    create = signed(client, "POST", PREFIX + "/sku", sku_body())
    assert create.status_code == 200
    sku_id = create.json()["data"]["ks_sku_id"]
    retry = signed(client, "POST", PREFIX + "/sku/", sku_body())
    assert retry.json()["data"]["ks_sku_id"] == sku_id

    audit_path = f"{PREFIX}/sku/{sku_id}"
    detail_path = f"{PREFIX}/sku/skus/{sku_id}"
    assert signed(client, "GET", audit_path).json()["data"]["result"]["code"] == 0
    assert signed(client, "POST", PREFIX + "/recognition", task_body(sku_id)).status_code == 400
    assert worker.process_job()
    assert signed(client, "GET", audit_path).json()["data"]["result"]["code"] == 2
    edit = signed(
        client,
        "PATCH",
        audit_path,
        {
            "sku_name": "新名称",
            "img_urls": sku_body()["img_urls"],
        },
    )
    assert edit.json()["data"]["sku_id"] == "vision-mock"
    assert signed(client, "GET", detail_path).json()["data"]["sku_name"] == "冰红茶"
    assert worker.process_job()
    assert pipeline.build.call_count == 1
    assert signed(client, "GET", detail_path).json()["data"]["sku_name"] == "新名称"
    assert signed(client, "GET", detail_path, account="other", secret="other").status_code == 400
    replacement = signed(
        client,
        "PATCH",
        audit_path,
        {
            "sku_name": "替换包装",
            "img_urls": ["https://assets.example/new.jpg"],
        },
    )
    assert replacement.status_code == 200
    assert worker.process_job()
    assert signed(client, "GET", audit_path).json()["data"]["result"]["code"] == 3
    assert signed(client, "GET", detail_path).json()["data"]["sku_name"] == "新名称"

    event = db.claim_delivery()
    assert json.loads(event["body"])["ks_sku_id"] == sku_id
    db.finish_delivery(event, False)
    with db.connection() as conn:
        state = conn.execute("SELECT * FROM delivery WHERE id=?", (event["id"],)).fetchone()
    assert state["attempts"] == 1 and state["next_at"] > time.time() + 9
    assert signed(client, "DELETE", audit_path).json()["data"]["sku_id"] == "vision-mock"
    assert signed(client, "GET", detail_path).status_code == 400


def test_training_submission_status_and_artifact(context, monkeypatch):
    _, db, config = context
    config = replace(config, matching_model="dinov3_vit7b_arcface")
    for sku in ("a", "b"):
        db.import_sku(
            owner="demo", caller_id=sku, sku_id=sku, info={"sku_name": sku}, features=b"test"
        )
    body = {
        "positive_samples": [
            {"sku_id": "a", "image_urls": ["https://assets.example/a.png"]},
            {"sku_id": "b", "image_urls": ["https://assets.example/b.png"]},
        ],
        "negative_image_urls": ["https://assets.example/negative.png"],
    }
    with TestClient(create_app(config), base_url="https://svc.example") as client:
        url = PREFIX + "/training"
        assert signed(client, "POST", url, body, secret="bad").status_code == 401
        assert signed(client, "POST", url, {
            **body, "positive_samples": body["positive_samples"][:1]
        }).status_code == 400
        submitted = signed(client, "POST", url, body)
        assert submitted.status_code == 200
        identifier = submitted.json()["data"]["ks_training_id"]
        assert signed(client, "POST", url, body).json()["data"]["ks_training_id"] == identifier
        url += f"/{identifier}"
        assert signed(client, "GET", url).json()["data"]["state"] == "queued"
        assert signed(client, "GET", url, account="other", secret="other").status_code == 400
        assert signed(client, "GET", url + "/artifact").status_code == 404
        assert db.claim_job() is None
        job = db.claim_job(training=True)
        assert job["ref"] == identifier
        assert signed(client, "GET", url).json()["data"]["state"] == "running"
        directory = config.data_dir / "training" / identifier
        directory.mkdir(parents=True)
        (directory / "arcface_patch.safetensors").write_bytes(b"candidate")
        monkeypatch.setattr("vending.worker.run_training", lambda *_: {
            "state": "succeeded", "artifact": {"sha256": "test", "size": 9}
        })
        Worker(config, db, Mock()).train(job)
        result = signed(client, "GET", url).json()["data"]
        assert result["state"] == "succeeded"
        assert signed(client, "GET", url + "/artifact").content == b"candidate"


def test_legacy_sku_recognition_scope_and_signed_callback(context, monkeypatch):
    client, db, config = context
    legacy = "001234567890123456789"
    db.import_sku(
        owner="demo",
        caller_id="legacy-platform-sku",
        sku_id=legacy,
        info=sku_body(),
        features=b"legacy-vector",
        aliases=["legacy-java-key"],
    )
    payload = task_body(legacy)
    payload["extra_params"] = json.dumps(payload["extra_params"])
    submission = signed(client, "POST", PREFIX + "/recognition", payload)
    assert submission.status_code == 200
    rid = submission.json()["data"]["ks_recog_id"]
    assert (
        signed(client, "POST", PREFIX + "/recognition", payload).json()["data"]["ks_recog_id"]
        == rid
    )
    recognition = PREFIX + "/recognition/" + rid
    assert signed(client, "GET", recognition).json()["data"]["state"]["code"] == 1

    pipeline = Mock()
    pipeline.recognize.return_value = outcome(2, items=[{"ks_sku_id": legacy, "sku_num": 2}])

    def fake_videos(body, directory, settings):
        path = directory / "example.mp4"
        path.write_bytes(b"video")
        return [path]

    monkeypatch.setattr("vending.worker.video_files", fake_videos)
    worker = Worker(config, db, pipeline)
    assert worker.process_job()
    assert pipeline.recognize.call_args.args[2] == {legacy: b"legacy-vector"}
    queried = signed(client, "GET", recognition).json()["data"]
    assert queried["result"]["result"] == [{"ks_sku_id": legacy, "sku_num": 2}]

    session = MagicMock()
    monkeypatch.setattr("vending.worker.http_session", lambda: session)
    post = session.__enter__.return_value.post
    post.return_value.status_code = 200
    post.return_value.json.return_value = ACK
    assert worker.deliver()
    sent = post.call_args.kwargs
    assert json.loads(sent["data"]) == {"ks_recog_id": rid, **queried}
    timestamp = sent["headers"]["X-Request-Date"]
    assert sent["headers"]["Authorization"] == authorization(
        "demo",
        "test-only",
        "java.example",
        "POST",
        "/notifyRecog",
        "",
        timestamp,
        sent["data"],
    )
    assert not worker.deliver()


def test_standard_catalog_and_pagination(context):
    client, db, _ = context
    db.import_sku(
        owner="",
        caller_id="std",
        sku_id="std-id",
        info=sku_body(),
        features=None,
        standard=True,
    )
    db.import_sku(
        owner="demo", caller_id="private", sku_id="private-id", info=sku_body(), features=b"x"
    )
    barcode = f"{PREFIX}/sku/all_sku/6920000000000"
    assert [row["sku_type"] for row in signed(client, "GET", barcode).json()["data"]] == ["0", "1"]
    assert len(signed(client, "GET", barcode, account="other", secret="other").json()["data"]) == 1
    listing = signed(client, "GET", f"{PREFIX}/sku/sku_list?page_size=1&page_num=2").json()["data"]
    assert listing["total"] == 2 and len(listing["resultList"]) == 1
    assert (
        signed(client, "GET", PREFIX + "/sku/sku_list", account="other", secret="other").status_code
        == 403
    )


def test_resource_exclusivity_and_local_file(context):
    from vending.media import video_files

    client, db, config = context
    db.import_sku(owner="demo", caller_id="legacy", sku_id="old", info=sku_body(), features=b"v")
    task = task_body("old")
    task["resource_infos"] = [{"resource_id": "old.mp4", "camera_id": "01TM"}]
    response = signed(client, "POST", PREFIX + "/recognition", task)
    assert response.json()["msg"] == "COEXISTENCE_ARGUMENTS"
    task.pop("resource_urls")
    task["resource_type"] = 2
    assert signed(client, "POST", PREFIX + "/recognition", task).status_code == 200
    (config.resource_dir / "old.mp4").write_bytes(b"local resource")
    dest = config.data_dir / "spool"
    dest.mkdir()
    assert video_files(task, dest, config)[0].read_bytes() == b"local resource"


def test_gpu_pipeline_preserves_external_sku_mapping(monkeypatch):
    from vending.engine import Pipeline

    monkeypatch.setattr("vending.engine.unpack_features", lambda data: {"features": data})
    pipeline = Pipeline()
    view = Mock(return_value=outcome(2, items=[{"ks_sku_id": "001", "sku_num": 1}]))
    monkeypatch.setattr(pipeline, "_recognize_view", view)
    pipeline.recognize(
        [Path("cabinet.mp4")],
        "A",
        {"001": b"one", "002": b"two"},
        comparison_assets={"outside": b"three"},
    )
    assert view.call_args.args[2] == {
        1: {"features": b"one"},
        2: {"features": b"two"},
        3: {"features": b"three"},
    }
    assert view.call_args.args[3] == {1: "001", 2: "002"}


def test_gpu_pipeline_requires_agreement_on_out_of_scope_result(monkeypatch):
    from vending.engine import Pipeline

    pipeline = Pipeline()
    monkeypatch.setattr(
        pipeline, "_recognize_view", Mock(side_effect=[outcome(7), outcome(4)])
    )
    result = pipeline.recognize([Path("main.mp4"), Path("aux.mp4")], "A", {})
    assert result["status"]["code"] == 4


def test_cabinet_gate_measures_distance_to_segmentation_mask():
    import numpy as np

    from app.detection.hand_area_gate import filter_hand_linked_candidates

    mask = np.zeros((240, 240), dtype=bool)
    mask[10:20, 10:20] = True
    mask[220:230, 220:230] = True
    hand = {"box": [110, 110, 120, 120]}
    candidate = {"box": [110, 110, 120, 120], "hand_indices": [0], "score": 0.9}
    hands = [[hand], [hand]]
    candidates = [[candidate], [candidate]]
    boxes = [[10, 10, 230, 230]] * 2

    _, box_result = filter_hand_linked_candidates(hands, candidates, boxes)
    _, mask_result = filter_hand_linked_candidates(
        hands, candidates, boxes, area_masks_by_frame=[mask, mask]
    )
    assert box_result["taken_tracks"] == 1
    assert mask_result["taken_tracks"] == 0


def test_unmatched_cabinet_mask_frames_must_align():
    from app.detection.hand_area_gate import cabinet_masks_for_candidates

    with pytest.raises(ValueError, match="帧数不一致"):
        cabinet_masks_for_candidates([], (10, 10), [[]], [[]])


@pytest.mark.parametrize(
    "status,departure,taken_tracks,door_opened,code,comments",
    [
        ("door_not_opened", None, 0, False, 3, "未开门"),
        ("door_opened_no_taken", None, 0, True, 3, "开门未拿商品"),
        ("unknown_product", "degenerate", 0, True, 3, "开门未拿商品"),
        ("unknown_product", "degenerate", 0, False, 4, None),
        ("unknown_product", "degenerate", 1, True, 4, "拿取后缺少可用于识别商品的有效轨迹"),
        ("unknown_product", "tracked", 0, True, 4, "拿取后缺少可用于识别商品的有效轨迹"),
        ("unknown_product", "sparse", 0, True, 4, None),
        ("not_found", None, 1, True, 4, None),
    ],
)
def test_unmatched_video_status_mapping(status, departure, taken_tracks, door_opened, code, comments):
    from vending.engine import unmatched_video_result

    result = unmatched_video_result(status, departure, taken_tracks, door_opened=door_opened)
    assert result["status"]["code"] == code
    if comments is not None:
        assert result["comments"] == comments


def test_documented_operations_and_callback_schema(context):
    client, _, _ = context
    document = client.get("/openapi.json").json()
    legacy = {path: methods for path, methods in document["paths"].items() if "/training" not in path}
    assert sum(len(methods) for methods in legacy.values()) == 9
    assert set(document["paths"][PREFIX + "/training"]) == {"post"}
    assert set(document["paths"][PREFIX + "/training/{ks_training_id}"]) == {"get"}
    assert set(document["paths"][PREFIX + "/training/{ks_training_id}/artifact"]) == {"get"}
    assert set(document["webhooks"]) == {"skuAuditResult", "recognitionResult"}
    create = document["paths"][PREFIX + "/sku"]["post"]
    assert "$ref" in create["responses"]["200"]["content"]["application/json"]["schema"]
    assert "422" not in create["responses"]


def test_queued_recognition_keeps_original_catalog_version(context):
    client, db, _ = context
    db.import_sku(owner="demo", caller_id="legacy", sku_id="001", info=sku_body(), features=b"old")
    response = signed(client, "POST", PREFIX + "/recognition", task_body("001"))
    ident = response.json()["data"]["ks_recog_id"]
    signed(client, "DELETE", PREFIX + "/sku/001")
    assert db.recognition_data(ident)[1] == {"001": b"old"}
    assert signed(client, "GET", PREFIX + "/sku/skus/001").status_code == 400


def test_recognition_snapshots_visible_comparison_versions_and_excludes_scope_alias(context):
    client, db, _ = context
    db.import_sku(
        owner="demo", caller_id="inside", sku_id="001", info=sku_body(),
        features=b"inside", aliases=["inside-alias"],
    )
    db.import_sku(
        owner="demo", caller_id="outside", sku_id="002", info=sku_body(), features=b"outside",
    )
    db.import_sku(
        owner="other", caller_id="private", sku_id="003", info=sku_body(), features=b"private",
    )
    db.import_sku(
        owner="", caller_id="public", sku_id="004", info=sku_body(),
        features=b"public", standard=True,
    )

    response = signed(client, "POST", PREFIX + "/recognition", task_body("inside-alias"))
    ident = response.json()["data"]["ks_recog_id"]
    assert signed(client, "DELETE", PREFIX + "/sku/002").status_code == 200

    _, scope, comparison = db.recognition_data(ident)
    assert scope == {"inside-alias": b"inside"}
    assert comparison == {"002": b"outside", "004": b"public"}

    with db.connection(write=True) as conn:
        conn.execute("DELETE FROM recognition_comparison WHERE id=?", (ident,))
    assert db.recognition_data(ident)[1:] == (scope, {})


def test_catalog_import_can_retry_without_changing_existing_sku(context):
    _, db, _ = context
    fields = {
        "owner": "demo",
        "caller_id": "legacy",
        "sku_id": "001",
        "info": sku_body(),
        "features": b"old",
        "aliases": ["alias"],
    }
    db.import_sku(**fields)
    db.import_sku(**fields)
    assert db.product("demo", "alias")["ks_sku_id"] == "001"
    with pytest.raises(ValueError, match="冲突"):
        db.import_sku(**{**fields, "features": b"different"})


def test_signature_matches_java_for_utf8_body():
    body = '{"sku_id":"vision-123","sku_name":"冰红茶"}'.encode()
    assert (
        authorization(
            "fixture",
            "fixture-secret",
            "EXAMPLE.TEST",
            "POST",
            "/openapi/sc/v1/sku",
            "",
            "1795419000000",
            body,
        )
        == "HMAC-SHA1 fixture:1ba93b0a48c93eee4e4a91f8a88f26b793201993"
    )
