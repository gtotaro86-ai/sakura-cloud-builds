"""Persistent job state machine. Restarts must not drop checkpoints."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STATES = (
    "CREATED",
    "INSPECTING",
    "PLANNING",
    "DEVELOPING",
    "TESTING",
    "QUANTIZING",
    "VALIDATING",
    "UPLOADING",
    "VERIFYING",
    "COMPLETED",
    "WAITING_FOR_WORKER",
    "ESCALATING",
    "PAUSED",
    "BLOCKED",
    "FAILED",
    "CANCELLED",
)

TERMINAL = frozenset({"COMPLETED", "BLOCKED", "FAILED", "CANCELLED"})

TRANSITIONS: dict[str, frozenset[str]] = {
    "CREATED": frozenset({"INSPECTING", "PAUSED", "CANCELLED"}),
    "INSPECTING": frozenset({"PLANNING", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"}),
    "PLANNING": frozenset({
        "DEVELOPING", "TESTING", "QUANTIZING", "WAITING_FOR_WORKER",
        "BLOCKED", "FAILED", "CANCELLED", "PAUSED", "ESCALATING", "COMPLETED",
    }),
    "DEVELOPING": frozenset({"TESTING", "ESCALATING", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"}),
    "TESTING": frozenset({"QUANTIZING", "DEVELOPING", "ESCALATING", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"}),
    "QUANTIZING": frozenset({"VALIDATING", "WAITING_FOR_WORKER", "FAILED", "BLOCKED", "CANCELLED", "PAUSED"}),
    "VALIDATING": frozenset({"UPLOADING", "QUANTIZING", "COMPLETED", "BLOCKED", "FAILED", "CANCELLED", "PAUSED"}),
    "UPLOADING": frozenset({"VERIFYING", "FAILED", "BLOCKED", "CANCELLED", "PAUSED"}),
    "VERIFYING": frozenset({"COMPLETED", "FAILED", "BLOCKED", "UPLOADING", "PAUSED", "CANCELLED"}),
    "WAITING_FOR_WORKER": frozenset({
        "QUANTIZING", "VALIDATING", "UPLOADING", "VERIFYING",
        "BLOCKED", "FAILED", "CANCELLED", "PAUSED", "COMPLETED",
    }),
    "ESCALATING": frozenset({"DEVELOPING", "TESTING", "BLOCKED", "FAILED", "CANCELLED"}),
    "PAUSED": frozenset({
        "CREATED", "INSPECTING", "PLANNING", "DEVELOPING", "TESTING", "QUANTIZING",
        "VALIDATING", "UPLOADING", "VERIFYING", "ESCALATING", "WAITING_FOR_WORKER", "CANCELLED",
    }),
    "BLOCKED": frozenset({"PLANNING", "DEVELOPING", "CANCELLED"}),
    "FAILED": frozenset({"PLANNING", "QUANTIZING", "CANCELLED"}),
    "COMPLETED": frozenset(),
    "CANCELLED": frozenset(),
}


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TransitionError(ValueError):
    pass


def dedupe_key(repo_id: str, revision: str | None, quants: list[str], variants: list[str], publish: bool, execution_backend: str = "auto") -> str | None:
    """Mutable branches/tags must be resolved before they can identify reusable work."""
    if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return None
    return json.dumps([repo_id.lower(), revision.lower(), sorted(set(quants)), sorted(set(variants)), bool(publish), execution_backend], separators=(",", ":"))


class JobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    @contextmanager
    def _write(self):
        # Serializes both threads sharing this connection and other processes.
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def __enter__(self) -> "JobStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              source_url TEXT NOT NULL,
              repo_id TEXT,
              revision TEXT,
              quants TEXT NOT NULL,
              state TEXT NOT NULL,
              dedupe_key TEXT,
              analysis_json TEXT,
              plan_json TEXT,
              result_json TEXT,
              error TEXT,
              publish INTEGER NOT NULL DEFAULT 0,
              paused_from TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id TEXT NOT NULL,
              ts TEXT NOT NULL,
              from_state TEXT,
              to_state TEXT,
              message TEXT,
              data_json TEXT
            );
            CREATE TABLE IF NOT EXISTS shards (
              job_id TEXT NOT NULL,
              quant TEXT NOT NULL,
              shard_index INTEGER NOT NULL,
              state TEXT NOT NULL,
              checksum TEXT,
              nbytes INTEGER,
              path TEXT,
              detail_json TEXT,
              PRIMARY KEY (job_id, quant, shard_index)
            );
            CREATE TABLE IF NOT EXISTS job_leases (
              job_id TEXT PRIMARY KEY,
              owner TEXT NOT NULL,
              expires_at REAL NOT NULL
            );
            """
        )
        self.conn.commit()
        # Upgrade legacy order-dependent keys without deleting any job history.
        from hfqa.heretic import normalize_variants
        with self._write():
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(jobs)")}
            if "variants" not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN variants TEXT NOT NULL DEFAULT '[\"normal\"]'")
            if "execution_backend" not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN execution_backend TEXT NOT NULL DEFAULT 'auto'")
            for row in self.conn.execute("SELECT * FROM jobs").fetchall():
                variants = normalize_variants(json.loads(row["variants"]))
                key = dedupe_key(row["repo_id"], row["revision"], json.loads(row["quants"]), variants, row["publish"], row["execution_backend"])
                encoded = json.dumps(variants)
                if row["dedupe_key"] != key or row["variants"] != encoded:
                    self.conn.execute("UPDATE jobs SET dedupe_key=?, variants=? WHERE id=?", (key, encoded, row["id"]))

    def create(
        self,
        source_url: str,
        repo_id: str,
        quants: list[str],
        publish: bool = False,
        revision: str | None = None,
        variants: list[str] | None = None,
        execution_backend: str = "auto",
    ) -> tuple[dict, bool]:
        from hfqa.heretic import normalize_variants
        chosen = normalize_variants(variants)
        if execution_backend not in {"auto", "oracle", "github"}:
            raise ValueError("unsupported execution backend")
        dedupe = dedupe_key(repo_id, revision, quants, chosen, publish, execution_backend)
        with self._write():
            if dedupe:
                row = self._duplicate(dedupe)
                if row:
                    return self._job(row), False
            jid = uuid.uuid4().hex[:16]
            now = utcnow()
            self.conn.execute(
                """INSERT INTO jobs
                   (id, source_url, repo_id, revision, quants, variants, state, dedupe_key, publish, created_at, updated_at, execution_backend)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (jid, source_url, repo_id, revision, json.dumps(quants), json.dumps(chosen), "CREATED", dedupe, int(bool(publish)), now, now, execution_backend),
            )
            self._event(jid, None, "CREATED", "job created", {"repo_id": repo_id, "quants": quants, "variants": chosen})
            return self.get(jid), True

    def _duplicate(self, key: str, exclude: str = ""):
        return self.conn.execute(
            """SELECT * FROM jobs WHERE dedupe_key=? AND id!=? AND state NOT IN ('FAILED','CANCELLED')
               ORDER BY created_at, rowid LIMIT 1""", (key, exclude),
        ).fetchone()

    def transition(self, job_id: str, to_state: str, message: str = "", data: dict | None = None, *, expected_state: str | None = None) -> dict:
        with self._write():
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if expected_state is not None and job["state"] != expected_state:
                raise TransitionError(f"expected {expected_state}, found {job['state']}")
            if to_state not in TRANSITIONS.get(job["state"], frozenset()):
                raise TransitionError(f"illegal transition {job['state']} -> {to_state}")
            paused_from = job["state"] if to_state == "PAUSED" else None
            self.conn.execute(
                "UPDATE jobs SET state=?, updated_at=?, paused_from=? WHERE id=?",
                (to_state, utcnow(), paused_from, job_id),
            )
            self._event(job_id, job["state"], to_state, message, data)
            return self.get(job_id)

    def event(self, job_id: str, frm: str | None, to: str | None, message: str, data: dict | None = None) -> None:
        with self._write():
            self._event(job_id, frm, to, message, data)

    def _event(self, job_id: str, frm: str | None, to: str | None, message: str, data: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (job_id, ts, from_state, to_state, message, data_json) VALUES (?,?,?,?,?,?)",
            (job_id, utcnow(), frm, to, message, json.dumps(data) if data is not None else None),
        )

    def update_fields(self, job_id: str, *, expected_state: str | None = None, **fields: object) -> dict:
        with self._write():
            return self._update_fields(job_id, expected_state=expected_state, **fields)

    def _update_fields(self, job_id: str, *, expected_state: str | None = None, **fields: object) -> dict:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if expected_state is not None and job["state"] != expected_state:
            raise TransitionError(f"expected {expected_state}, found {job['state']}")
        allowed = {
            "repo_id", "revision", "analysis_json", "plan_json", "result_json",
            "error", "dedupe_key", "quants",
        }
        sets: list[str] = []
        vals: list[object] = []
        for key, value in fields.items():
            if key not in allowed:
                raise KeyError(key)
            if key.endswith("_json") and not isinstance(value, str):
                value = json.dumps(value)
            if key == "quants" and not isinstance(value, str):
                value = json.dumps(value)
            sets.append(f"{key}=?")
            vals.append(value)
        sets.append("updated_at=?")
        vals.append(utcnow())
        vals.append(job_id)
        self.conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", vals)
        return self.get(job_id)

    def pin_revision(self, job_id: str, revision: str, **fields: object) -> tuple[dict, bool]:
        """Pin metadata atomically, reusing an existing job for the same exact work.

        False means the caller must stop processing this job and return the
        canonical job supplied here. The duplicate is cancelled, with an event.
        """
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("a full resolved commit SHA is required")
        with self._write():
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["state"] in TERMINAL or job["state"] == "PAUSED":
                raise TransitionError(f"cannot pin a {job['state']} job")
            key = dedupe_key(str(fields.get("repo_id") or job["repo_id"]), revision, job["quants"], job["variants"], job["publish"], job.get("execution_backend", "auto"))
            duplicate = self._duplicate(key, job_id)
            if duplicate:
                self.conn.execute("UPDATE jobs SET state='CANCELLED', updated_at=?, error=? WHERE id=?", (utcnow(), f"duplicate of {duplicate['id']}", job_id))
                self._event(job_id, job["state"], "CANCELLED", "resolved revision already has a job", {"canonical_job_id": duplicate["id"]})
                return self._job(duplicate), False
            fields.update(revision=revision.lower(), dedupe_key=key)
            return self._update_fields(job_id, **fields), True

    def retry(self, job_id: str) -> dict:
        """Retry planning while retaining checksummed artifacts and upload receipts."""
        with self._write():
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["state"] not in {"FAILED", "BLOCKED"}:
                raise TransitionError("only failed or blocked jobs can be retried")
            if job.get("dedupe_key") and self._duplicate(job["dedupe_key"], job_id):
                raise TransitionError("another job already owns this resolved revision and request")
            lease = self.conn.execute("SELECT 1 FROM job_leases WHERE job_id=? AND expires_at>?", (job_id, time.time())).fetchone()
            if lease:
                raise TransitionError("worker is still active; retry after it releases the job")
            self.conn.execute("UPDATE jobs SET state='PLANNING', error=NULL, plan_json=NULL, paused_from=NULL, updated_at=? WHERE id=?", (utcnow(), job_id))
            self._event(job_id, job["state"], "PLANNING", "operator retry; retained artifacts and checkpoints")
            return self.get(job_id)

    def claim(self, job_id: str, owner: str, ttl_seconds: float = 300) -> bool:
        if not owner or ttl_seconds <= 0:
            raise ValueError("owner and a positive lease duration are required")
        with self._write():
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["state"] in TERMINAL or job["state"] == "PAUSED":
                return False
            now = time.time()
            lease = self.conn.execute("SELECT * FROM job_leases WHERE job_id=?", (job_id,)).fetchone()
            if lease and lease["expires_at"] > now:
                return False
            self.conn.execute("INSERT INTO job_leases(job_id,owner,expires_at) VALUES (?,?,?) ON CONFLICT(job_id) DO UPDATE SET owner=excluded.owner, expires_at=excluded.expires_at", (job_id, owner, now + ttl_seconds))
            return True

    def renew_claim(self, job_id: str, owner: str, ttl_seconds: float = 300) -> bool:
        if ttl_seconds <= 0:
            raise ValueError("a positive lease duration is required")
        with self._write():
            now = time.time()
            updated = self.conn.execute("UPDATE job_leases SET expires_at=? WHERE job_id=? AND owner=? AND expires_at>?", (now + ttl_seconds, job_id, owner, now))
            return updated.rowcount == 1

    def release_claim(self, job_id: str, owner: str) -> None:
        with self._write():
            self.conn.execute("DELETE FROM job_leases WHERE job_id=? AND owner=?", (job_id, owner))

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._job(row) if row else None

    def list_jobs(self, state: str | None = None) -> list[dict]:
        with self._lock:
            return self._list_jobs(state)

    def _list_jobs(self, state: str | None = None) -> list[dict]:
        if state:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE state=? ORDER BY created_at DESC", (state,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [self._job(row) for row in rows]

    def events_for(self, job_id: str) -> list[dict]:
        with self._lock:
            return self._events_for(job_id)

    def _events_for(self, job_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE job_id=? ORDER BY id", (job_id,)
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            if item.get("data_json"):
                item["data"] = json.loads(item["data_json"])
            out.append(item)
        return out

    def upsert_shard(
        self,
        job_id: str,
        quant: str,
        index: int,
        state: str,
        checksum: str | None = None,
        nbytes: int | None = None,
        path: str | None = None,
        detail: dict | None = None,
    ) -> None:
        with self._write():
            self._upsert_shard(job_id, quant, index, state, checksum, nbytes, path, detail)

    def _upsert_shard(self, job_id, quant, index, state, checksum, nbytes, path, detail):
        self.conn.execute(
            """INSERT INTO shards (job_id, quant, shard_index, state, checksum, nbytes, path, detail_json)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(job_id, quant, shard_index) DO UPDATE SET
                 state=excluded.state,
                 checksum=COALESCE(excluded.checksum, shards.checksum),
                 nbytes=COALESCE(excluded.nbytes, shards.nbytes),
                 path=COALESCE(excluded.path, shards.path),
                 detail_json=COALESCE(excluded.detail_json, shards.detail_json)""",
            (
                job_id, quant, index, state, checksum, nbytes, path,
                json.dumps(detail) if detail is not None else None,
            ),
        )

    def shards(self, job_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM shards WHERE job_id=? ORDER BY quant, shard_index", (job_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def _job(self, row: sqlite3.Row) -> dict:
        data = dict(row)
        data["quants"] = json.loads(data["quants"])
        data["variants"] = json.loads(data.get("variants") or "[\"normal\"]")
        for key in ("analysis_json", "plan_json", "result_json"):
            raw = data.get(key)
            name = key[:-5]
            data[name] = json.loads(raw) if raw else None
        data["publish"] = bool(data["publish"])
        return data
