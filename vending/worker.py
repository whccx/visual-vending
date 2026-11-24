"""处理持久任务，后台独立发送两种签名回调。"""

import fcntl
import json
import logging
import signal
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import requests

from vending.config import Settings
from vending.engine import ImageRejected, Pipeline, image_hashes
from vending.image_review import ImageReviewRequired, check_reference_images, make_product_detector
from vending.media import MediaError, http_session, images_from_urls, save_cover, video_files
from vending.protocol import ApiError, allowed_url, authorization
from vending.store import Store, outcome

LOG = logging.getLogger("vending.worker")
ACK = {"code": "10000", "msg": "请求成功", "data": {}}


class Worker:
    def __init__(
        self,
        config: Settings,
        store: Store,
        pipeline: Pipeline,
        *,
        image_detector=None,
        recognition_pipeline=None,
    ) -> None:
        self.config = config
        self.store = store
        self.pipeline = pipeline
        self.recognition_pipeline = recognition_pipeline or pipeline
        self.review_lock = threading.Lock()
        self.image_detector = (
            make_product_detector(config) if image_detector is None else image_detector
        )

    def process_job(self) -> bool:
        job = self.store.claim_job()
        if job is None:
            return False
        try:
            if job["kind"] == "sku":
                with self.review_lock:
                    self.review(job)
            else:
                self.recognize(job)
        except Exception:
            self.store.retry_job(job)
            raise
        return True

    def review(self, job) -> None:
        row = self.store.review_data(int(job["ref"]))
        body = json.loads(row["body"])
        features = hashes = None
        comments, rejected = "", False
        try:
            if row["deleted"]:
                self.store.finish_review(
                    job, features=None, hashes=None, delay=self.config.callback_delay
                )
                return
            if row["image_decision"] == "reject":
                raise ImageRejected(row["image_decision_comments"] or "人工复核拒绝参考图")
            active = json.loads(row["active_body"]) if row["active_body"] else None
            if active and body.get("img_urls") == active.get("img_urls"):
                features = row["active_features"]
                hashes = json.loads(row["active_hashes"]) if row["active_hashes"] else None
            else:
                images = images_from_urls(body.get("img_urls") or [], self.config)
                if not images:
                    raise ImageRejected("没有可用于建模的参考图")
                hashes = image_hashes(images)
                if active:
                    if hashes != json.loads(row["active_hashes"] or "null"):
                        raise ImageRejected("已审核通过的 SKU 不允许替换参考图")
                    features = row["active_features"]
                else:
                    approved = row["image_decision"] == "approve" and hashes == json.loads(
                        row["approved_image_hashes"]
                    )
                    if not approved:
                        check_reference_images(images, self.image_detector)
                    features = self.pipeline.build(images, check_size=self.config.check_image_size)
                save_cover(images[0], hashes[0], self.config)
            if features is None:
                raise ImageRejected("商品缺少可用的建模特征")
        except ImageReviewRequired as error:
            self.store.hold_image_review(job, hashes, error.findings)
            return
        except (MediaError, ImageRejected) as error:
            comments, rejected = str(error), True
        except Exception:
            LOG.exception("SKU 建模未完成，编号 %s", row["sku"])
            comments = "商品建模失败"
        self.store.finish_review(
            job,
            features=features,
            hashes=hashes,
            comments=comments[:256],
            image_rejected=rejected,
            delay=self.config.callback_delay,
        )

    def recognize(self, job) -> None:
        body, features, comparisons = self.store.recognition_data(job["ref"])
        try:
            spool = self.config.data_dir / "spool"
            spool.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="recognition-", dir=spool) as tmp:
                shared = Path(tmp)
                shared.chmod(0o750)
                videos = video_files(body, shared, self.config)
                for path in videos:
                    path.chmod(0o640)
                result = self.recognition_pipeline.recognize(
                    videos, body["container_id"], features, comparison_assets=comparisons
                )
        except MediaError as error:
            result = outcome(10, str(error)[:256])
        except Exception:
            LOG.exception("识别任务执行失败，编号 %s", job["ref"])
            result = outcome(4, "识别服务未能完成本单处理")
        self.store.finish_recognition(job, result, self.config.callback_delay)

    def deliver(self) -> bool:
        event = self.store.claim_delivery()
        if event is None:
            return False
        success = False
        try:
            url = allowed_url(event["url"], self.config.callback_hosts, "notify_url")
            target = urlsplit(url)
            account = self.config.accounts[event["owner"]]
            date = str(time.time_ns() // 1_000_000)
            payload = event["body"].encode("utf-8")
            request = requests.Request("POST", url, data=payload).prepare()
            path, _, query = request.path_url.partition("?")
            signature = authorization(
                event["owner"],
                account.secret,
                target.netloc,
                "POST",
                path,
                query,
                date,
                payload,
            )
            with http_session() as session:
                response = session.post(
                    url,
                    data=payload,
                    allow_redirects=False,
                    timeout=(5, 15),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": signature,
                        "Referer": f"{target.scheme}://{target.netloc}",
                        "X-Request-Date": date,
                        "X-Request-Id": uuid.uuid4().hex,
                    },
                )
            success = response.status_code == 200 and response.json() == ACK
        except (ApiError, KeyError, ValueError, requests.RequestException):
            LOG.warning("回调未确认，事件 %s，第 %s 次尝试", event["id"], event["attempts"] + 1)
        self.store.finish_delivery(event, success)
        return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = Settings.from_env()
    store = Store(config.data_dir)
    store.initialize()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with (config.data_dir / ".worker.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("该数据目录已有 GPU worker") from exc
        store.requeue_jobs()
        from vending.frame_pipeline import FrameRuntime

        runtime = FrameRuntime(config.video_concurrency)
        worker = Worker(config, store, Pipeline(), recognition_pipeline=runtime)

        def poll(method):
            while not stop.is_set():
                try:
                    if method():
                        continue
                except Exception:
                    LOG.exception("任务队列暂时不可用")
                stop.wait(0.25)

        callbacks = threading.Thread(target=poll, args=(worker.deliver,), name="callbacks")
        callbacks.start()
        LOG.info("视频识别并发上限：%s", config.video_concurrency)
        try:
            with ThreadPoolExecutor(
                max_workers=config.video_concurrency, thread_name_prefix="订单"
            ) as jobs:
                futures = [
                    jobs.submit(poll, worker.process_job) for _ in range(config.video_concurrency)
                ]
                for future in futures:
                    future.result()
        finally:
            stop.set()
            callbacks.join()
            runtime.close()


if __name__ == "__main__":
    main()
