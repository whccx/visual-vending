"""SQLite 商品版本、识别队列和可靠回调。"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from vending.protocol import ApiError, bad

AUDIT = {0: "待审核", 1: "审核中", 2: "审核通过", 3: "审核不通过"}
OUTCOME = {
    1: "识别中",
    2: "成功",
    3: "成功：未拿商品",
    4: "失败：无法识别",
    6: "告警：故意遮挡",
    7: "告警：柜中无此SKU",
    8: "告警：其他告警",
    9: "告警：异物拿放",
    10: "告警：疑似设备故障",
}
RETRY_DELAYS = (10, 30, 60, 120, 180)


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def outcome(code: int, comments: str = "", items: list[dict] | None = None) -> dict:
    result = {"status": {"code": code, "desc": OUTCOME[code]}}
    if code == 2:
        result["result"] = items or []
    if comments:
        result["comments"] = comments
    return result


def recognition_view(result: dict | None) -> dict:
    return {
        "state": {
            "code": 1 if result is None else 2,
            "desc": "识别中" if result is None else "识别完成",
        },
        "result": outcome(1) if result is None else result,
    }


class Store:
    def __init__(self, directory: Path, public_url: str = "") -> None:
        self.directory = directory
        self.public_url = public_url
        self.path = directory / "platform.sqlite3"

    @contextmanager
    def connection(self, *, write: bool = False):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            with db:
                yield db
        finally:
            db.close()

    def initialize(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sku (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    caller_id TEXT NOT NULL,
                    standard INTEGER NOT NULL DEFAULT 0,
                    active_version INTEGER,
                    latest_version INTEGER NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (owner, caller_id)
                );
                CREATE TABLE IF NOT EXISTS sku_version (
                    id INTEGER PRIMARY KEY,
                    sku TEXT NOT NULL,
                    body TEXT NOT NULL,
                    state INTEGER NOT NULL DEFAULT 0,
                    comments TEXT NOT NULL DEFAULT '',
                    image_rejected INTEGER NOT NULL DEFAULT 0,
                    features BLOB,
                    image_hashes TEXT
                );
                CREATE INDEX IF NOT EXISTS sku_version_sku ON sku_version (sku, id);
                CREATE TABLE IF NOT EXISTS sku_image_hold (
                    version INTEGER PRIMARY KEY,
                    hashes TEXT NOT NULL,
                    findings TEXT NOT NULL,
                    decision TEXT NOT NULL DEFAULT 'waiting',
                    comments TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS sku_alias (
                    alias TEXT PRIMARY KEY,
                    sku TEXT NOT NULL REFERENCES sku(id)
                );
                CREATE TABLE IF NOT EXISTS recognition (
                    id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    caller_id TEXT NOT NULL,
                    body TEXT NOT NULL,
                    candidates TEXT NOT NULL,
                    result TEXT,
                    UNIQUE (owner, caller_id)
                );
                CREATE TABLE IF NOT EXISTS recognition_comparison (
                    id TEXT PRIMARY KEY REFERENCES recognition(id),
                    candidates TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job (
                    id INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE TABLE IF NOT EXISTS delivery (
                    id INTEGER PRIMARY KEY,
                    owner TEXT NOT NULL,
                    url TEXT NOT NULL,
                    body TEXT NOT NULL,
                    next_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'pending'
                );
            """)

    @staticmethod
    def _limit(db, maximum: int) -> None:
        if (
            db.execute("SELECT count(*) FROM job WHERE state IN ('pending','running')").fetchone()[
                0
            ]
            >= maximum
        ):
            raise ApiError(403, "CONCURRENCY_LIMIT_EXCEEDED", bare=True)

    @staticmethod
    def _new_id(db, table: str, prefix: str) -> str:
        while True:
            value = prefix + uuid.uuid4().hex[:30]
            if table == "sku":
                row = db.execute(
                    "SELECT id FROM sku WHERE id=? UNION ALL SELECT sku FROM sku_alias WHERE alias=?",
                    (value, value),
                ).fetchone()
            else:
                row = db.execute("SELECT id FROM recognition WHERE id=?", (value,)).fetchone()
            if row is None:
                return value

    def create_sku(self, owner: str, caller_id: str, body: dict, limit: int) -> str:
        payload = encode(body)
        with self.connection(write=True) as db:
            row = db.execute(
                """
                SELECT id, deleted FROM sku WHERE owner=? AND caller_id=?
            """,
                (owner, caller_id),
            ).fetchone()
            if row:
                original = db.execute(
                    "SELECT body FROM sku_version WHERE sku=? ORDER BY id LIMIT 1", (row["id"],)
                ).fetchone()
                if not row["deleted"] and original["body"] == payload:
                    return row["id"]
                raise bad("sku_id", "商品已存在")
            self._limit(db, limit)
            sku_id = self._new_id(db, "sku", "S-")
            version = db.execute(
                "INSERT INTO sku_version (sku,body) VALUES (?,?)", (sku_id, payload)
            ).lastrowid
            db.execute(
                "INSERT INTO sku (id,owner,caller_id,latest_version) VALUES (?,?,?,?)",
                (sku_id, owner, caller_id, version),
            )
            db.execute("INSERT INTO job (kind,ref) VALUES ('sku',?)", (str(version),))
        return sku_id

    @staticmethod
    def _sku(db, owner: str, sku_id: str, *, private: bool = False):
        row = db.execute(
            """
            SELECT sku.*, latest.state, latest.comments, latest.image_rejected,
                   latest.body AS latest_body, active.body AS active_body,
                   active.image_hashes AS active_hashes,
                   EXISTS (
                       SELECT 1 FROM sku_image_hold AS held
                       WHERE held.version=latest.id AND held.decision='waiting'
                   ) AS waiting_for_image_review
            FROM sku
            JOIN sku_version AS latest ON sku.latest_version=latest.id
            LEFT JOIN sku_version AS active ON sku.active_version=active.id
            WHERE sku.deleted=0
              AND (sku.id=? OR sku.id=(SELECT sku FROM sku_alias WHERE alias=?))
              AND ((sku.owner=? AND sku.standard=0) OR (sku.standard=1 AND ?=0))
        """,
            (sku_id, sku_id, owner, private),
        ).fetchone()
        if row is None:
            raise bad("ks_sku_id", "商品不存在")
        return row

    def edit_sku(self, owner: str, sku_id: str, patch: dict, limit: int) -> str:
        with self.connection(write=True) as db:
            row = self._sku(db, owner, sku_id, private=True)
            latest = json.loads(row["latest_body"])
            base = json.loads(row["active_body"]) if row["active_body"] else latest
            changed = {**base, **patch}
            if row["state"] in (0, 1):
                if changed == latest:
                    return row["caller_id"]
                raise bad("ks_sku_id", "当前版本正在审核")
            if changed == latest and row["state"] == 2:
                return row["caller_id"]
            if (
                not row["active_version"]
                and not row["image_rejected"]
                and changed.get("img_urls") != latest.get("img_urls")
            ):
                raise bad("img_urls", "仅允许为因图片被拒的 SKU 替换参考图")
            self._limit(db, limit)
            version = db.execute(
                "INSERT INTO sku_version (sku,body) VALUES (?,?)", (row["id"], encode(changed))
            ).lastrowid
            db.execute("UPDATE sku SET latest_version=? WHERE id=?", (version, row["id"]))
            db.execute("INSERT INTO job (kind,ref) VALUES ('sku',?)", (str(version),))
            return row["caller_id"]

    def audit(self, owner: str, sku_id: str) -> dict:
        with self.connection() as db:
            row = self._sku(db, owner, sku_id)
        state = 0 if row["waiting_for_image_review"] else row["state"]
        payload = {"result": {"code": state, "desc": AUDIT[state]}}
        if row["comments"]:
            payload["comments"] = row["comments"]
        return payload

    def delete_sku(self, owner: str, sku_id: str) -> str:
        with self.connection(write=True) as db:
            row = self._sku(db, owner, sku_id, private=True)
            db.execute("UPDATE sku SET deleted=1 WHERE id=?", (row["id"],))
            db.execute(
                """
                UPDATE job SET state='done' WHERE kind='sku' AND state IN ('pending','image_review')
                AND ref IN (SELECT CAST(id AS TEXT) FROM sku_version WHERE sku=?)
            """,
                (row["id"],),
            )
            return row["caller_id"]

    def _product(self, row, body: dict, *, include_images: bool = True) -> dict:
        cover = (body.get("img_urls") or [""])[0]
        hashes = json.loads(row["active_hashes"] or "null")
        if (
            self.public_url
            and hashes
            and (self.directory / "covers" / f"{hashes[0]}.jpg").is_file()
        ):
            cover = f"{self.public_url}/covers/{hashes[0]}.jpg"
        product = {
            "ks_sku_id": row["id"],
            "sku_id": row["caller_id"],
            "sku_name": body["sku_name"],
            "sku_type": "0" if row["standard"] else "1",
            "specs": body.get("specs") or {},
            "cover": cover,
            "has_sub_sku": bool(body.get("sub_sku")),
            "sub_sku": body.get("sub_sku", []),
        }
        if body.get("barcode") is not None and not product["has_sub_sku"]:
            product["barcode"] = body["barcode"]
        if include_images:
            product["img_urls"] = body.get("img_urls") or []
        return product

    def product(self, owner: str, sku_id: str) -> dict:
        with self.connection() as db:
            row = self._sku(db, owner, sku_id)
        if not row["active_body"]:
            raise bad("ks_sku_id", "商品尚未生效")
        return self._product(row, json.loads(row["active_body"]))

    def products(
        self,
        owner: str,
        *,
        barcode: str | None = None,
        search: str = "",
        only_standard: bool = False,
        only_single: bool = False,
        page_num: int = 1,
        page_size: int = 10,
    ):
        conditions = [
            "sku.deleted=0",
            "(sku.owner=? OR sku.standard=1)",
        ]
        args: list = [owner]
        if barcode is not None:
            conditions.append("json_extract(active.body,'$.barcode')=?")
            args.append(barcode)
        if search:
            conditions.append("""
                (instr(lower(json_extract(active.body,'$.sku_name')), lower(?)) > 0
                OR instr(json_extract(active.body,'$.barcode'), ?) > 0)
            """)
            args.extend((search, search))
        if only_standard:
            conditions.append("sku.standard=1")
        if only_single:
            conditions.append("coalesce(json_array_length(active.body,'$.sub_sku'), 0)=0")
        source = """
            FROM sku JOIN sku_version AS active ON sku.active_version=active.id WHERE
        """ + " AND ".join(conditions)
        query = (
            "SELECT sku.*, active.body, active.image_hashes AS active_hashes "
            + source
            + " ORDER BY sku.standard DESC, sku.id"
        )
        with self.connection() as db:
            if barcode is not None:
                rows = db.execute(query, args).fetchall()
                if not rows:
                    raise bad("barcode", "商品不存在")
                return [self._product(row, json.loads(row["body"])) for row in rows]
            total = db.execute("SELECT count(*) " + source, args).fetchone()[0]
            offset = (page_num - 1) * page_size
            rows = (
                []
                if offset >= total
                else db.execute(query + " LIMIT ? OFFSET ?", [*args, page_size, offset]).fetchall()
            )
        return {
            "page_num": page_num,
            "page_size": page_size,
            "total": total,
            "resultList": [
                self._product(row, json.loads(row["body"]), include_images=False) for row in rows
            ],
        }

    def submit(self, owner: str, body: dict, limit: int) -> str:
        caller_id = body["recog_id"]
        packed = encode(body)
        with self.connection(write=True) as db:
            existing = db.execute(
                "SELECT id,body FROM recognition WHERE owner=? AND caller_id=?", (owner, caller_id)
            ).fetchone()
            if existing:
                if existing["body"] != packed:
                    raise bad("recog_id", "相同任务编号的请求内容不同")
                return existing["id"]
            self._limit(db, limit)
            candidates = {}
            for candidate in body["extra_params"]["sku_scope"]:
                candidate_id = candidate["sku_id"]
                try:
                    row = self._sku(db, owner, candidate_id)
                except ApiError as exc:
                    raise bad("sku_scope", "包含不存在的商品") from exc
                if not row["active_version"]:
                    raise bad("sku_scope", "包含未生效的商品")
                if not db.execute(
                    "SELECT features IS NOT NULL FROM sku_version WHERE id=?",
                    (row["active_version"],),
                ).fetchone()[0]:
                    raise bad("sku_scope", "包含尚无识别特征的商品")
                version = str(row["active_version"])
                if version in candidates and candidates[version] != candidate_id:
                    raise bad("sku_scope", "同一商品传入了多个外部 ID")
                candidates[version] = candidate_id
            recog_id = self._new_id(db, "recognition", "R-")
            db.execute(
                "INSERT INTO recognition (id,owner,caller_id,body,candidates) VALUES (?,?,?,?,?)",
                (recog_id, owner, caller_id, packed, encode(candidates)),
            )
            comparisons = {
                str(row["active_version"]): row["id"]
                for row in db.execute(
                    """
                    SELECT sku.id, sku.active_version
                    FROM sku JOIN sku_version AS active ON active.id=sku.active_version
                    WHERE sku.deleted=0 AND active.state=2 AND active.features IS NOT NULL
                      AND (sku.owner=? OR sku.standard=1)
                    """,
                    (owner,),
                )
                if str(row["active_version"]) not in candidates
            }
            if comparisons:
                db.execute(
                    "INSERT INTO recognition_comparison (id,candidates) VALUES (?,?)",
                    (recog_id, encode(comparisons)),
                )
            db.execute("INSERT INTO job (kind,ref) VALUES ('recognition',?)", (recog_id,))
            return recog_id

    def recognition(self, owner: str, recog_id: str) -> dict:
        with self.connection() as db:
            row = db.execute(
                "SELECT result FROM recognition WHERE owner=? AND id=?", (owner, recog_id)
            ).fetchone()
        if row is None:
            raise bad("ks_recog_id", "识别任务不存在")
        return recognition_view(json.loads(row["result"]) if row["result"] else None)

    def requeue_jobs(self) -> None:
        with self.connection(write=True) as db:
            db.execute("""
                UPDATE sku_version SET state=0 WHERE id IN (
                    SELECT ref FROM job WHERE kind='sku' AND state='running'
                )
            """)
            db.execute("UPDATE job SET state='pending' WHERE state='running'")
            db.execute("UPDATE delivery SET state='pending' WHERE state='sending'")

    def claim_job(self):
        with self.connection(write=True) as db:
            job = db.execute(
                "SELECT * FROM job WHERE state='pending' ORDER BY id LIMIT 1"
            ).fetchone()
            if job:
                db.execute("UPDATE job SET state='running' WHERE id=?", (job["id"],))
                if job["kind"] == "sku":
                    db.execute("UPDATE sku_version SET state=1 WHERE id=?", (job["ref"],))
            return job

    def review_data(self, version: int):
        with self.connection() as db:
            row = db.execute(
                """
                SELECT pending.*, sku.owner, sku.caller_id, sku.deleted, sku.active_version,
                       active.body AS active_body, active.image_hashes AS active_hashes,
                       active.features AS active_features, held.decision AS image_decision,
                       held.hashes AS approved_image_hashes, held.comments AS image_decision_comments
                FROM sku_version AS pending JOIN sku ON pending.sku=sku.id
                LEFT JOIN sku_version AS active ON sku.active_version=active.id
                LEFT JOIN sku_image_hold AS held ON pending.id=held.version
                WHERE pending.id=?
            """,
                (version,),
            ).fetchone()
        return row

    def hold_image_review(self, job, hashes: list[str], findings: list[dict]) -> None:
        with self.connection(write=True) as db:
            row = db.execute(
                "SELECT deleted FROM sku WHERE latest_version=?",
                (job["ref"],),
            ).fetchone()
            if row is None or row["deleted"]:
                db.execute("UPDATE job SET state='done' WHERE id=?", (job["id"],))
                return
            db.execute(
                """
                INSERT INTO sku_image_hold (version,hashes,findings) VALUES (?,?,?)
                ON CONFLICT(version) DO UPDATE SET hashes=excluded.hashes,
                    findings=excluded.findings,decision='waiting',comments=''
            """,
                (job["ref"], encode(hashes), encode(findings)),
            )
            db.execute("UPDATE job SET state='image_review' WHERE id=?", (job["id"],))

    def pending_image_reviews(self) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""
                SELECT sku.id,version.body,held.findings FROM sku
                JOIN sku_version AS version ON sku.latest_version=version.id
                JOIN sku_image_hold AS held ON held.version=version.id
                JOIN job ON job.kind='sku' AND job.ref=CAST(version.id AS TEXT)
                WHERE sku.deleted=0 AND job.state='image_review'
                ORDER BY job.id
            """).fetchall()
        reviews = []
        for row in rows:
            product = json.loads(row["body"])
            reviews.append(
                {
                    "ks_sku_id": row["id"],
                    "sku_name": product["sku_name"],
                    "img_urls": product["img_urls"],
                    "findings": json.loads(row["findings"]),
                }
            )
        return reviews

    def decide_image_review(self, sku_id: str, decision: str, comments: str = "") -> None:
        if decision not in {"approve", "reject", "retry"}:
            raise ValueError("无效的人工审核操作")
        with self.connection(write=True) as db:
            row = db.execute(
                """
                SELECT sku.latest_version,job.id AS job_id FROM sku
                JOIN job ON job.kind='sku' AND job.ref=CAST(sku.latest_version AS TEXT)
                WHERE sku.id=? AND sku.deleted=0 AND job.state='image_review'
            """,
                (sku_id,),
            ).fetchone()
            if row is None:
                raise ValueError("没有找到该 SKU 待人工处理的图片审核")
            db.execute(
                "UPDATE sku_image_hold SET decision=?,comments=? WHERE version=?",
                (
                    decision,
                    comments,
                    row["latest_version"],
                ),
            )
            db.execute("UPDATE sku_version SET state=0 WHERE id=?", (row["latest_version"],))
            db.execute("UPDATE job SET state='pending' WHERE id=?", (row["job_id"],))

    @staticmethod
    def _deliver(db, owner: str, url: str | None, body: dict, delay: float) -> None:
        if url and owner:
            db.execute(
                "INSERT INTO delivery (owner,url,body,next_at) VALUES (?,?,?,?)",
                (owner, url, encode(body), time.time() + delay),
            )

    def finish_review(
        self,
        job,
        *,
        features: bytes | None,
        hashes: list[str] | None,
        comments: str = "",
        image_rejected: bool = False,
        delay: float,
    ) -> None:
        with self.connection(write=True) as db:
            row = db.execute(
                """
                SELECT sku.*, sku_version.body FROM sku JOIN sku_version ON sku.latest_version=sku_version.id
                WHERE sku_version.id=?
            """,
                (job["ref"],),
            ).fetchone()
            state = 3 if comments else 2
            db.execute(
                """
                UPDATE sku_version SET state=?,comments=?,image_rejected=?,features=?,image_hashes=?
                WHERE id=?
            """,
                (state, comments, image_rejected, features, encode(hashes), job["ref"]),
            )
            db.execute("UPDATE job SET state='done' WHERE id=?", (job["id"],))
            if row is None or row["deleted"]:
                return
            if state == 2:
                db.execute("UPDATE sku SET active_version=? WHERE id=?", (job["ref"], row["id"]))
            event = {
                "sku_id": row["caller_id"],
                "ks_sku_id": row["id"],
                "result": {"code": state, "desc": AUDIT[state]},
            }
            if comments:
                event["comments"] = comments
            self._deliver(db, row["owner"], json.loads(row["body"]).get("notify_url"), event, delay)

    def recognition_data(self, recog_id: str) -> tuple[dict, dict[str, bytes], dict[str, bytes]]:
        with self.connection() as db:
            row = db.execute(
                """
                SELECT recognition.body, recognition.candidates,
                       recognition_comparison.candidates AS comparisons
                FROM recognition LEFT JOIN recognition_comparison USING (id)
                WHERE recognition.id=?
                """,
                (recog_id,),
            ).fetchone()
            feature_query = """
                SELECT snapshot.value, sku_version.features
                FROM json_each(?) AS snapshot
                JOIN sku_version ON sku_version.id=CAST(snapshot.key AS INTEGER)
            """
            features = dict(db.execute(feature_query, (row["candidates"],)))
            comparisons = dict(db.execute(feature_query, (row["comparisons"],)))
        return json.loads(row["body"]), features, comparisons

    def finish_recognition(self, job, result: dict, delay: float) -> None:
        with self.connection(write=True) as db:
            row = db.execute("SELECT * FROM recognition WHERE id=?", (job["ref"],)).fetchone()
            db.execute("UPDATE recognition SET result=? WHERE id=?", (encode(result), row["id"]))
            db.execute("UPDATE job SET state='done' WHERE id=?", (job["id"],))
            event = {"ks_recog_id": row["id"], **recognition_view(result)}
            self._deliver(db, row["owner"], json.loads(row["body"]).get("notify_url"), event, delay)

    def claim_delivery(self):
        with self.connection(write=True) as db:
            event = db.execute(
                """
                SELECT * FROM delivery AS pending WHERE pending.state='pending' AND pending.next_at<=?
                  AND NOT EXISTS (
                      SELECT 1 FROM delivery AS older
                      WHERE older.id < pending.id AND older.state IN ('pending', 'sending')
                        AND older.owner=pending.owner
                        AND json_extract(older.body, '$.ks_sku_id')=json_extract(pending.body, '$.ks_sku_id')
                  )
                ORDER BY pending.next_at,pending.id LIMIT 1
            """,
                (time.time(),),
            ).fetchone()
            if event:
                db.execute("UPDATE delivery SET state='sending' WHERE id=?", (event["id"],))
            return event

    def finish_delivery(self, event, success: bool) -> None:
        attempts = event["attempts"] + 1
        next_at = time.time()
        state = "delivered" if success else "failed"
        if not success and attempts <= len(RETRY_DELAYS):
            next_at += RETRY_DELAYS[attempts - 1]
            state = "pending"
        with self.connection(write=True) as db:
            db.execute(
                "UPDATE delivery SET state=?,attempts=?,next_at=? WHERE id=?",
                (state, attempts, next_at, event["id"]),
            )

    def import_sku(
        self,
        *,
        owner: str,
        caller_id: str,
        sku_id: str,
        info: dict,
        features: bytes | None,
        image_hashes: list[str] | None = None,
        state: int = 2,
        comments: str = "",
        standard: bool = False,
        aliases: list[str] = (),
    ) -> None:
        with self.connection(write=True) as db:
            existing = db.execute(
                """
                SELECT sku.owner,sku.caller_id,sku.standard,sku.deleted,version.body,
                       version.state,version.comments,version.features,version.image_hashes
                FROM sku JOIN sku_version AS version ON sku.latest_version=version.id
                WHERE sku.id=?
            """,
                (sku_id,),
            ).fetchone()
            if existing:
                current_aliases = {
                    row["alias"]
                    for row in db.execute("SELECT alias FROM sku_alias WHERE sku=?", (sku_id,))
                }
                expected = (
                    owner,
                    caller_id,
                    int(standard),
                    0,
                    encode(info),
                    state,
                    comments,
                    features,
                    encode(image_hashes),
                )
                if tuple(existing) == expected and current_aliases == set(aliases):
                    return
                raise ValueError("已有商品的导入信息与当前记录冲突")
            names = [sku_id, *aliases]
            if len(set(names)) != len(names) or any(
                db.execute(
                    "SELECT id FROM sku WHERE id=? UNION ALL SELECT sku FROM sku_alias WHERE alias=?",
                    (name, name),
                ).fetchone()
                for name in names
            ):
                raise ValueError("导入商品编号或别名发生冲突")
            version = db.execute(
                """
                INSERT INTO sku_version (sku,body,state,comments,image_rejected,features,image_hashes)
                VALUES (?,?,?,?,?,?,?)
            """,
                (
                    sku_id,
                    encode(info),
                    state,
                    comments,
                    bool(info.get("image_rejected")),
                    features,
                    encode(image_hashes),
                ),
            ).lastrowid
            db.execute(
                """
                INSERT INTO sku (id,owner,caller_id,standard,active_version,latest_version)
                VALUES (?,?,?,?,?,?)
            """,
                (sku_id, owner, caller_id, standard, version if state == 2 else None, version),
            )
            db.executemany(
                "INSERT INTO sku_alias (alias,sku) VALUES (?,?)",
                [(alias, sku_id) for alias in aliases],
            )
