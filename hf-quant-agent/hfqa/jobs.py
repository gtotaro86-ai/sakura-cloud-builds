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
    "AWAITING_APPROVAL",
    "RETRY_WAIT",
)

TERMINAL = frozenset({"COMPLETED", "BLOCKED", "FAILED", "CANCELLED"})
ARCHIVABLE = TERMINAL
REORDERABLE = frozenset({"CREATED", "INSPECTING", "PLANNING", "WAITING_FOR_WORKER", "PAUSED","RETRY_WAIT"})
EXECUTING = frozenset({"QUANTIZING", "VALIDATING", "UPLOADING", "VERIFYING"})
QUALITY_MODES = frozenset({"normal", "imatrix_if_supported"})
MODEL_KINDS = frozenset({"auto", "text", "image"})

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
    "VALIDATING": frozenset({"UPLOADING", "QUANTIZING", "COMPLETED", "BLOCKED", "FAILED", "CANCELLED", "PAUSED","AWAITING_APPROVAL"}),
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
    "AWAITING_APPROVAL":frozenset({'VALIDATING','CANCELLED'}),
    "RETRY_WAIT":frozenset({'CREATED','INSPECTING','PLANNING','WAITING_FOR_WORKER','QUANTIZING','VALIDATING','UPLOADING','VERIFYING','PAUSED','CANCELLED','BLOCKED'}),
}
for _state in ('CREATED','INSPECTING','PLANNING','WAITING_FOR_WORKER','QUANTIZING','VALIDATING','UPLOADING','VERIFYING'):
    TRANSITIONS[_state]=TRANSITIONS[_state]|{'RETRY_WAIT'}
TRANSITIONS['PAUSED']=TRANSITIONS['PAUSED']|{'RETRY_WAIT'}
TRANSITIONS['CREATED']=TRANSITIONS['CREATED']|{'BLOCKED'}


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TransitionError(ValueError):
    pass


def dedupe_key(repo_id: str, revision: str | None, quants: list[str], variants: list[str], publish: bool, execution_backend: str = "auto", quality_mode: str = "normal", model_kind: str = "auto",release_options=None) -> str | None:
    """Mutable branches/tags must be resolved before they can identify reusable work."""
    if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return None
    parts = [repo_id.lower(), revision.lower(), sorted(set(quants)), sorted(set(variants)), bool(publish), execution_backend]
    # Normal requests retain the deployed key, including legacy auto jobs.
    if quality_mode != "normal" or model_kind != "auto":
        parts.append(quality_mode)
    if model_kind != "auto":
        parts.append(model_kind)
    from hfqa.release_options import normalize
    if normalize(release_options,repo_id):parts.append(normalize(release_options,repo_id))
    return json.dumps(parts, separators=(",", ":"))


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
            CREATE TABLE IF NOT EXISTS execution_leases (
              resource TEXT PRIMARY KEY,
              job_id TEXT NOT NULL,
              owner TEXT NOT NULL,
              expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS spark_leases (
              resource TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_turns (
              id TEXT PRIMARY KEY, job_id TEXT NOT NULL, request_key TEXT NOT NULL,
              user_text TEXT NOT NULL, answer TEXT, status TEXT NOT NULL,
              phase TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL, owner TEXT, expires_at REAL, question_id TEXT,
              research_mode TEXT NOT NULL DEFAULT 'auto', research_json TEXT,
              allow_external INTEGER NOT NULL DEFAULT 0,
              attempts INTEGER NOT NULL DEFAULT 0, UNIQUE(job_id, request_key)
            );
            CREATE INDEX IF NOT EXISTS chat_queue ON chat_turns(status, created_at);
            CREATE TABLE IF NOT EXISTS operator_questions (
              id TEXT PRIMARY KEY, job_id TEXT NOT NULL, fingerprint TEXT NOT NULL UNIQUE,
              role TEXT, text TEXT NOT NULL, status TEXT NOT NULL,
              created_at TEXT NOT NULL, answered_at TEXT, answer_turn_id TEXT
            );
            """
        )
        self.conn.commit()
        # Upgrade legacy order-dependent keys without deleting any job history.
        from hfqa.heretic import normalize_variants
        with self._write():
            chat_columns = {row[1] for row in self.conn.execute('PRAGMA table_info(chat_turns)')}
            if 'question_id' not in chat_columns:
                self.conn.execute('ALTER TABLE chat_turns ADD COLUMN question_id TEXT')
            if 'research_mode' not in chat_columns:
                self.conn.execute("ALTER TABLE chat_turns ADD COLUMN research_mode TEXT NOT NULL DEFAULT 'auto'")
            if 'research_json' not in chat_columns:
                self.conn.execute('ALTER TABLE chat_turns ADD COLUMN research_json TEXT')
            if 'allow_external' not in chat_columns:
                self.conn.execute('ALTER TABLE chat_turns ADD COLUMN allow_external INTEGER NOT NULL DEFAULT 0')
            if 'answer_provider' not in chat_columns:
                self.conn.execute('ALTER TABLE chat_turns ADD COLUMN answer_provider TEXT')
            if 'external_usage_json' not in chat_columns:
                self.conn.execute('ALTER TABLE chat_turns ADD COLUMN external_usage_json TEXT')
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(jobs)")}
            if "variants" not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN variants TEXT NOT NULL DEFAULT '[\"normal\"]'")
            if "execution_backend" not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN execution_backend TEXT NOT NULL DEFAULT 'auto'")
            if "quality_mode" not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN quality_mode TEXT NOT NULL DEFAULT 'normal'")
            if "model_kind" not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN model_kind TEXT NOT NULL DEFAULT 'auto'")
            if "queue_position" not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN queue_position INTEGER NOT NULL DEFAULT 0")
                for position, row in enumerate(self.conn.execute("SELECT id FROM jobs ORDER BY created_at, rowid").fetchall(), 1):
                    self.conn.execute("UPDATE jobs SET queue_position=? WHERE id=?", (position, row["id"]))
            if 'archived_at' not in columns:
                self.conn.execute('ALTER TABLE jobs ADD COLUMN archived_at TEXT')
            if 'release_options_json' not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN release_options_json TEXT NOT NULL DEFAULT '{}'")
            if 'retry_json' not in columns:
                self.conn.execute("ALTER TABLE jobs ADD COLUMN retry_json TEXT NOT NULL DEFAULT '{}'")
            # Startup migrations need identity fields, never complete tensor receipts
            # or chat/worker processes would reload all model history every poll.
            for row in self.conn.execute("SELECT id,repo_id,revision,quants,variants,publish,execution_backend,quality_mode,model_kind,dedupe_key,release_options_json FROM jobs").fetchall():
                variants = normalize_variants(json.loads(row["variants"]))
                key = dedupe_key(row["repo_id"], row["revision"], json.loads(row["quants"]), variants, row["publish"], row["execution_backend"], row["quality_mode"], row["model_kind"],json.loads(row['release_options_json']))
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
        quality_mode: str = "normal",
        model_kind: str = "auto",
        release_options: dict | None = None,
    ) -> tuple[dict, bool]:
        from hfqa.heretic import normalize_variants
        chosen = normalize_variants(variants)
        if execution_backend not in {"auto", "oracle", "github"}:
            raise ValueError("unsupported execution backend")
        if quality_mode not in QUALITY_MODES:
            raise ValueError("unsupported quality mode")
        if model_kind not in MODEL_KINDS:
            raise ValueError("unsupported model kind")
        from hfqa.release_options import normalize
        release_options=normalize(release_options,repo_id)
        dedupe = dedupe_key(repo_id, revision, quants, chosen, publish, execution_backend, quality_mode, model_kind,release_options)
        with self._write():
            if dedupe:
                row = self._duplicate(dedupe)
                if row:
                    self.conn.execute('UPDATE jobs SET archived_at=NULL WHERE id=?',(row['id'],))
                    row=self.conn.execute('SELECT * FROM jobs WHERE id=?',(row['id'],)).fetchone()
                    return self._job(row), False
            jid = uuid.uuid4().hex[:16]
            now = utcnow()
            position = self.conn.execute("SELECT COALESCE(MAX(queue_position),0)+1 FROM jobs").fetchone()[0]
            self.conn.execute(
                """INSERT INTO jobs
                   (id, source_url, repo_id, revision, quants, variants, state, dedupe_key, publish, created_at, updated_at, execution_backend, quality_mode, model_kind, queue_position,release_options_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (jid, source_url, repo_id, revision, json.dumps(quants), json.dumps(chosen), "CREATED", dedupe, int(bool(publish)), now, now, execution_backend, quality_mode, model_kind, position,json.dumps(release_options)),
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
            if to_state in {'COMPLETED','CANCELLED'} and job.get('retry'):
                retry={**job['retry'],'status':'recovered' if to_state=='COMPLETED' else 'cancelled','next_attempt_at':None,'next_attempt_epoch':None}
                self.conn.execute('UPDATE jobs SET retry_json=? WHERE id=?',(json.dumps(retry),job_id))
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
            "retry_json",
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
            key = dedupe_key(str(fields.get("repo_id") or job["repo_id"]), revision, job["quants"], job["variants"], job["publish"], job.get("execution_backend", "auto"), job.get("quality_mode", "normal"), job.get("model_kind", "auto"),job.get('release_options'))
            duplicate = self._duplicate(key, job_id)
            if duplicate:
                self.conn.execute('UPDATE jobs SET archived_at=NULL WHERE id=?',(duplicate['id'],))
                duplicate=self.conn.execute('SELECT * FROM jobs WHERE id=?',(duplicate['id'],)).fetchone()
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
            self.conn.execute("UPDATE jobs SET state='PLANNING', error=NULL, plan_json=NULL, paused_from=NULL, retry_json='{}', updated_at=? WHERE id=?", (utcnow(), job_id))
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

    def _ordered_queue(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM jobs ORDER BY queue_position, created_at, rowid").fetchall()
        locked = {row[0] for row in self.conn.execute("SELECT job_id FROM execution_leases WHERE expires_at>?", (time.time(),))}
        jobs = []
        for row in rows:
            if row["state"] in TERMINAL:
                continue
            job = self._job(row)
            job["queue_reorderable"] = job["state"] in REORDERABLE and job["id"] not in locked
            jobs.append(job)
        return jobs

    def ordered_queue(self) -> list[dict]:
        """Persistent user order, including paused jobs and the protected active job."""
        with self._lock:
            return self._ordered_queue()

    def reorder_queue(self, job_ids: list[str]) -> list[dict]:
        """Replace the complete reorderable set atomically; stale clients must refresh.

        Occupied positions remain reserved for active work. An incoming job or a
        job that starts during a phone edit produces a conflict instead of being
        silently dropped or moved.
        """
        with self._write():
            queued = self._ordered_queue()
            movable = [job for job in queued if job["queue_reorderable"]]
            if len(job_ids) != len(set(job_ids)) or set(job_ids) != {job["id"] for job in movable}:
                raise TransitionError("queue changed; provide every reorderable job exactly once")
            for position, job_id in zip(sorted(job["queue_position"] for job in movable), job_ids):
                self.conn.execute("UPDATE jobs SET queue_position=?, updated_at=? WHERE id=?", (position, utcnow(), job_id))
            return self._ordered_queue()

    def _execution_candidates(self) -> list[dict]:
        jobs = []
        for job in self._ordered_queue():
            if job['state']=='RETRY_WAIT':continue
            if job["state"] == "PAUSED" or job["execution_backend"] == "github":
                continue
            if job["state"] not in REORDERABLE | EXECUTING:
                continue
            if (job.get("plan") or {}).get("route") in {"existing_pipeline", "github_standard_unqualified", "agent_runtime"}:
                continue
            jobs.append(job)
        # An interrupted model must finish before a later timer starts another.
        return sorted(jobs, key=lambda job: (job["state"] not in EXECUTING, job["queue_position"]))

    def acquire_execution(self, job_id: str, owner: str, ttl_seconds: float = 3600) -> bool:
        """Claim the single Oracle heavy-work slot, across all worker processes.

        Use before the ordinary job claim; renew both while working and release
        both in finally. Planning leases remain independent of this CPU slot.
        """
        if not owner or ttl_seconds <= 0:
            raise ValueError("owner and a positive lease duration are required")
        with self._write():
            if self.get(job_id) is None:
                raise KeyError(job_id)
            now = time.time()
            lease = self.conn.execute("SELECT * FROM execution_leases WHERE resource='oracle'").fetchone()
            if lease and lease["expires_at"] > now:
                return False
            candidates = self._execution_candidates()
            if not candidates or candidates[0]["id"] != job_id or candidates[0]["state"] not in EXECUTING | {"WAITING_FOR_WORKER"}:
                return False
            self.conn.execute("INSERT INTO execution_leases(resource,job_id,owner,expires_at) VALUES ('oracle',?,?,?) ON CONFLICT(resource) DO UPDATE SET job_id=excluded.job_id, owner=excluded.owner, expires_at=excluded.expires_at", (job_id, owner, now + ttl_seconds))
            return True

    def renew_execution(self, job_id: str, owner: str, ttl_seconds: float = 3600) -> bool:
        if not owner or ttl_seconds <= 0:
            raise ValueError("owner and a positive lease duration are required")
        with self._write():
            now = time.time()
            updated = self.conn.execute("UPDATE execution_leases SET expires_at=? WHERE resource='oracle' AND job_id=? AND owner=? AND expires_at>?", (now + ttl_seconds, job_id, owner, now))
            return updated.rowcount == 1

    def release_execution(self, job_id: str, owner: str) -> None:
        with self._write():
            self.conn.execute("DELETE FROM execution_leases WHERE resource='oracle' AND job_id=? AND owner=?", (job_id, owner))

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._job(row) if row else None

    def list_jobs(self, state: str | None = None, *, include_archived: bool = False) -> list[dict]:
        with self._lock:
            return self._list_jobs(state, include_archived=include_archived)

    def _list_jobs(self, state: str | None = None, *, include_archived: bool = False) -> list[dict]:
        visible='' if include_archived else ' AND archived_at IS NULL'
        if state:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE state=?"+visible+" ORDER BY created_at DESC", (state,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM jobs WHERE 1=1"+visible+" ORDER BY created_at DESC").fetchall()
        return [self._job(row) for row in rows]

    def _assert_idle(self, job_id: str) -> None:
        if self.conn.execute('SELECT 1 FROM job_leases WHERE job_id=? AND expires_at>?',(job_id,time.time())).fetchone():
            raise TransitionError('worker is still releasing this job; try again shortly')
        if self.conn.execute('SELECT 1 FROM execution_leases WHERE job_id=? AND expires_at>?',(job_id,time.time())).fetchone():
            raise TransitionError('execution slot is still in use')

    def archive(self, job_id: str) -> dict:
        """Remove finished entries from the app list without deleting HF models."""
        with self._write():
            row=self.conn.execute('SELECT state,archived_at FROM jobs WHERE id=?',(job_id,)).fetchone()
            if row is None:raise KeyError(job_id)
            if row['state'] not in ARCHIVABLE:
                raise TransitionError('only completed, cancelled, failed or blocked jobs can be removed from the list')
            self._assert_idle(job_id)
            if not row['archived_at']:
                self.conn.execute('UPDATE jobs SET archived_at=? WHERE id=?',(utcnow(),job_id))
                self._event(job_id,row['state'],row['state'],'removed from app list; remote artifacts retained')
            return self.get(job_id)

    def restore(self, job_id: str) -> dict:
        """Bring an archived entry back into the app list."""
        with self._write():
            row=self.conn.execute('SELECT state,archived_at FROM jobs WHERE id=?',(job_id,)).fetchone()
            if row is None:raise KeyError(job_id)
            if row['archived_at']:
                self.conn.execute('UPDATE jobs SET archived_at=NULL WHERE id=?',(job_id,))
                self._event(job_id,row['state'],row['state'],'restored to app list')
            return self.get(job_id)

    def archived_jobs(self) -> list[dict]:
        with self._lock:
            rows=self.conn.execute('SELECT * FROM jobs WHERE archived_at IS NOT NULL ORDER BY archived_at DESC').fetchall()
            return [self._job(row) for row in rows]

    def archive_many(self, states, older_than_days: float = 0, *, dry_run: bool = False) -> dict:
        """Bulk-remove finished entries by state and age. Never touches active work or remote repositories."""
        chosen=set(states or ())
        if not chosen or not chosen<=ARCHIVABLE:
            raise TransitionError('bulk removal accepts only completed, cancelled, failed or blocked states')
        cutoff=datetime.now(timezone.utc).timestamp()-max(0.0,float(older_than_days))*86400
        with self._write():
            marks=','.join('?'*len(chosen))
            rows=self.conn.execute(f'SELECT id,state,updated_at FROM jobs WHERE archived_at IS NULL AND state IN ({marks}) ORDER BY updated_at',tuple(sorted(chosen))).fetchall()
            matched,skipped=[],[]
            for row in rows:
                try:updated=datetime.fromisoformat(row['updated_at']).timestamp()
                except (TypeError,ValueError):updated=0.0
                if updated>cutoff:continue
                try:self._assert_idle(row['id'])
                except TransitionError:
                    skipped.append(row['id']);continue
                matched.append(row)
            if not dry_run:
                now=utcnow()
                for row in matched:
                    self.conn.execute('UPDATE jobs SET archived_at=? WHERE id=?',(now,row['id']))
                    self._event(row['id'],row['state'],row['state'],'removed from app list; remote artifacts retained')
            return {'matched':len(matched),'archived':0 if dry_run else len(matched),'skipped_busy':len(skipped),
                    'dry_run':bool(dry_run),'job_ids':[row['id'] for row in matched],'remote_models_deleted':False}

    def purge(self, job_id: str) -> dict:
        """Permanently forget an already archived job (history, shards, chat). Hub repositories are never touched."""
        with self._write():
            row=self.conn.execute('SELECT state,archived_at FROM jobs WHERE id=?',(job_id,)).fetchone()
            if row is None:raise KeyError(job_id)
            if not row['archived_at'] or row['state'] not in ARCHIVABLE:
                raise TransitionError('only archived, finished jobs can be deleted permanently')
            self._assert_idle(job_id)
            for table in ('events','shards','chat_turns','operator_questions'):
                self.conn.execute(f'DELETE FROM {table} WHERE job_id=?',(job_id,))
            self.conn.execute('DELETE FROM jobs WHERE id=?',(job_id,))
            return {'purged':True,'job_id':job_id,'remote_models_deleted':False}

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
        data['release_options']=json.loads(data.get('release_options_json') or '{}')
        data['retry']=json.loads(data.get('retry_json') or '{}')
        for key in ("analysis_json", "plan_json", "result_json"):
            raw = data.get(key)
            name = key[:-5]
            data[name] = json.loads(raw) if raw else None
        data["publish"] = bool(data["publish"])
        return data
