"""识别队列并发、视频名额与故障重排回归。"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from vending.config import Account, Settings
from vending.store import Store, outcome
from vending.worker import Worker


def test_five_orders_are_claimed_once_and_sixth_waits(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.initialize()
    store.import_sku(
        owner="demo",
        caller_id="demo",
        sku_id="sku",
        info={"sku_name": "商品"},
        features=b"features",
    )
    for index in range(6):
        store.submit(
            "demo",
            {
                "recog_id": f"job-{index}",
                "container_id": "cabinet",
                "resource_urls": ["https://example/video.mp4"],
                "extra_params": {"sku_scope": [{"sku_id": "sku"}]},
            },
            100,
        )
    config = Settings(data_dir=tmp_path, accounts={"demo": Account("test")})
    release = threading.Event()
    all_running = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def recognize(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 5:
                all_running.set()
        assert release.wait(10)
        with lock:
            active -= 1
        return outcome(2, items=[{"ks_sku_id": "sku", "sku_num": 1}])

    def video_files(body, directory, settings):
        path = directory / "video.mp4"
        path.write_bytes(b"fixture")
        return [path]

    monkeypatch.setattr("vending.worker.video_files", video_files)
    pipeline = Mock()
    pipeline.recognize.side_effect = recognize
    worker = Worker(config, store, Mock(), recognition_pipeline=pipeline)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(worker.process_job) for _ in range(6)]
        try:
            assert all_running.wait(10)
            with store.connection() as db:
                assert (
                    db.execute("SELECT count(*) FROM job WHERE state='running'").fetchone()[0] == 5
                )
                assert (
                    db.execute("SELECT count(*) FROM job WHERE state='pending'").fetchone()[0] == 1
                )
        finally:
            release.set()
        assert all(future.result() for future in futures)
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM job WHERE state='done'").fetchone()[0] == 6
    assert peak == 5
    assert pipeline.recognize.call_count == 6


def test_failed_claim_is_requeued(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.initialize()
    store.import_sku(
        owner="demo",
        caller_id="demo",
        sku_id="sku",
        info={"sku_name": "商品"},
        features=b"features",
    )
    store.submit(
        "demo",
        {
            "recog_id": "job",
            "container_id": "cabinet",
            "resource_urls": ["https://example/video.mp4"],
            "extra_params": {"sku_scope": [{"sku_id": "sku"}]},
        },
        100,
    )
    worker = Worker(Settings(data_dir=tmp_path, accounts={"demo": Account("test")}), store, Mock())
    monkeypatch.setattr(worker, "recognize", Mock(side_effect=RuntimeError("模拟事务故障")))
    try:
        worker.process_job()
    except RuntimeError:
        pass
    else:
        raise AssertionError("故障必须抛出")
    with store.connection() as db:
        assert db.execute("SELECT state FROM job").fetchone()[0] == "pending"


def test_multiple_views_share_five_video_slots(tmp_path, monkeypatch):
    from vending.frame_pipeline import FrameRuntime

    monkeypatch.setattr("vending.frame_pipeline.unpack_features", lambda data: {"features": data})
    runtime = FrameRuntime(5)
    release = threading.Event()
    all_running = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def recognize_view(path, cabinet, catalog, sku_ids):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 5:
                all_running.set()
        assert release.wait(10)
        with lock:
            active -= 1
        return outcome(2, items=[{"ks_sku_id": "sku", "sku_num": 1}])

    monkeypatch.setattr(runtime, "recognize_view", recognize_view)
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(
                runtime.recognize,
                [tmp_path / f"{i}.mp4" for i in range(6)],
                "cabinet",
                {"sku": b"features"},
            )
            try:
                assert all_running.wait(10)
                assert peak == 5
            finally:
                release.set()
            assert future.result()["result"] == [{"ks_sku_id": "sku", "sku_num": 1}]
    finally:
        runtime.close()
