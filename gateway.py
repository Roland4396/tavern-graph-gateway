#!/usr/bin/env python3
import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web
from multidict import CIMultiDict


LISTEN_HOST = os.environ.get("GRAPH_GATEWAY_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("GRAPH_GATEWAY_PORT", "18188"))
UPSTREAM_HOST = os.environ.get("GRAPH_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("GRAPH_UPSTREAM_PORT", "18189"))
UPSTREAM_PREFIX = os.environ.get("GRAPH_UPSTREAM_PREFIX", "").rstrip("/")
UPSTREAM_HTTP = f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}{UPSTREAM_PREFIX}"
UPSTREAM_WS = f"ws://{UPSTREAM_HOST}:{UPSTREAM_PORT}{UPSTREAM_PREFIX}"
IDLE_SECONDS = int(os.environ.get("GRAPH_IDLE_SECONDS", "1800"))
QUEUE_POLL_SECONDS = int(os.environ.get("GRAPH_QUEUE_POLL_SECONDS", "15"))
RESTART_BACKOFF_SECONDS = int(os.environ.get("GRAPH_RESTART_BACKOFF_SECONDS", "30"))
RESTART_MAX_BACKOFF_SECONDS = int(
    os.environ.get("GRAPH_RESTART_MAX_BACKOFF_SECONDS", "300")
)
UNLOAD_RETRY_SECONDS = int(os.environ.get("GRAPH_UNLOAD_RETRY_SECONDS", "60"))
REMOTE_TARGET = os.environ.get("GRAPH_REMOTE_TARGET", "gpu-a")
REMOTE_START = os.environ.get(
    "GRAPH_REMOTE_START", "/home/worker/.local/libexec/graph-stack/start.sh"
)
REMOTE_STOP = os.environ.get(
    "GRAPH_REMOTE_STOP", "/home/worker/.local/libexec/graph-stack/stop.sh"
)
CONTROL_URL = os.environ.get("GRAPH_CONTROL_URL", "").rstrip("/")
CACHE_DIR = Path(os.environ.get("GRAPH_CACHE_DIR", "/home/runner/.cache/graph-ui-service"))
STATE_DIR = Path(os.environ.get("GRAPH_STATE_DIR", "/var/lib/graph-ui-service"))
H3_CONTROL_URL = os.environ.get(
    "GRAPH_H3_CONTROL_URL", "http://gpu-orchestrator:8080/h3-control"
).rstrip("/")
STATUS_PATH = "/_graph_gateway/status"
PROMPT_STATUS_PREFIX = "/_graph_gateway/prompts/"
H3_OPERATION_NAMESPACE = uuid.UUID("00000000-0000-4000-8000-000000000004")
H3_SAMPLER_WORKLOADS = {
    "MiniMaxH3FL2VASampler": "h3",
    "MiniMaxH3Ref2VASampler": "h3_ref",
}

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
WS_HANDSHAKE_HEADERS = {
    "host",
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-key",
    "sec-websocket-protocol",
    "sec-websocket-version",
}
CACHED_HEADERS = {
    "content-type",
    "content-language",
    "content-disposition",
    "etag",
    "last-modified",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
LOG = logging.getLogger("graph-ui-gateway")


class DiskCache:
    @staticmethod
    def is_extension_asset(path: str) -> bool:
        # These namespaces serve package files, not workflows or generated media.
        # In particular /rgthree/config.js is dynamic and must stay uncached.
        path = path.split("?", 1)[0]
        return path.startswith((
            "/extensions/", "/scripts/", "/kjweb_async/",
            "/rgthree/common/", "/rgthree/lib/",
        )) and Path(path).suffix.lower() in {
            ".js", ".mjs", ".css", ".txt", ".wasm", ".woff", ".woff2",
            ".ttf", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".ico",
        }

    def __init__(self, root: Path) -> None:
        self.root = root
        self.locks: dict[str, asyncio.Lock] = {}
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    @staticmethod
    def policy(path: str) -> str | None:
        if DiskCache.is_extension_asset(path):
            return "private, max-age=300"
        if path in {"/", "/index.html"}:
            return "no-store"
        if path.startswith("/assets/"):
            return "public, max-age=31536000, immutable"
        if path.startswith("/api/global_subgraphs/"):
            return "private, max-age=31536000, immutable"
        if path.startswith("/fonts/") or path == "/materialdesignicons.min.css":
            return "public, max-age=86400"
        return None

    @staticmethod
    def key(rel_url: str) -> str:
        return hashlib.sha256(rel_url.encode("utf-8")).hexdigest()

    def paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.body", self.root / f"{key}.json"

    async def get(self, rel_url: str) -> tuple[bytes, dict[str, str]] | None:
        body_path, meta_path = self.paths(self.key(rel_url))

        def read() -> tuple[bytes, dict[str, str]] | None:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if metadata.get("url") != rel_url:
                    return None
                if self.ttl(rel_url) is not None:
                    created_at = float(metadata.get("created_at", 0))
                    if not created_at or time.time() - created_at > self.ttl(rel_url):
                        return None
                return body_path.read_bytes(), metadata.get("headers", {})
            except (OSError, ValueError, TypeError):
                return None

        return await asyncio.to_thread(read)

    async def put(self, rel_url: str, body: bytes, headers: dict[str, str]) -> None:
        key = self.key(rel_url)
        body_path, meta_path = self.paths(key)

        def write() -> None:
            body_tmp = body_path.with_suffix(".body.tmp")
            meta_tmp = meta_path.with_suffix(".json.tmp")
            body_tmp.write_bytes(body)
            meta_tmp.write_text(
                json.dumps(
                    {"url": rel_url, "headers": headers, "created_at": time.time()},
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            body_tmp.chmod(0o600)
            meta_tmp.chmod(0o600)
            body_tmp.replace(body_path)
            meta_tmp.replace(meta_path)

        await asyncio.to_thread(write)

    @staticmethod
    def ttl(path: str) -> int | None:
        if DiskCache.is_extension_asset(path):
            return 86400
        if path in {"/", "/index.html"}:
            return 86400
        return None

    def lock(self, rel_url: str) -> asyncio.Lock:
        key = self.key(rel_url)
        return self.locks.setdefault(key, asyncio.Lock())


class PromptConflict(ValueError):
    pass


class PromptJournal:
    """Durable journal for H3-bearing Comfy prompts.

    Comfy's in-memory PromptQueue disappears with the Graph process.  The
    gateway stores the exact normalized request before forwarding it and can
    therefore replay the same prompt id after a restart.  Stable operation ids
    make every H3 sampler replay idempotent at the GPU queue.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA busy_timeout=5000")
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    prompt_id TEXT PRIMARY KEY,
                    source_hash TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    client_id TEXT,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    forwarded_at REAL,
                    forward_attempts INTEGER NOT NULL DEFAULT 0,
                    terminal_at REAL,
                    error_json TEXT
                );
                CREATE INDEX IF NOT EXISTS prompt_recovery
                    ON prompts(status, updated_at);
                CREATE INDEX IF NOT EXISTS prompt_dedupe
                    ON prompts(client_id, source_hash, created_at);
                CREATE TABLE IF NOT EXISTS prompt_operations (
                    prompt_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL UNIQUE,
                    workload TEXT NOT NULL,
                    cancel_requested_at REAL,
                    cancel_confirmed_at REAL,
                    cancel_error_json TEXT,
                    acked_at REAL,
                    PRIMARY KEY(prompt_id, node_id),
                    FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id) ON DELETE CASCADE
                );
                PRAGMA user_version=2;
                """
            )
            prompt_columns = {
                str(row[1]) for row in self._db.execute("PRAGMA table_info(prompts)")
            }
            if "forward_attempts" not in prompt_columns:
                self._db.execute(
                    "ALTER TABLE prompts ADD COLUMN forward_attempts INTEGER NOT NULL DEFAULT 0"
                )
            operation_columns = {
                str(row[1])
                for row in self._db.execute("PRAGMA table_info(prompt_operations)")
            }
            for name, declaration in (
                ("cancel_requested_at", "REAL"),
                ("cancel_confirmed_at", "REAL"),
                ("cancel_error_json", "TEXT"),
            ):
                if name not in operation_columns:
                    self._db.execute(
                        f"ALTER TABLE prompt_operations ADD COLUMN {name} {declaration}"
                    )
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _canonical(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _hash(cls, value: object) -> str:
        return hashlib.sha256(cls._canonical(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        try:
            result["request"] = json.loads(result.pop("request_json"))
        except (TypeError, ValueError):
            result["request"] = {}
            result.pop("request_json", None)
        error_json = result.pop("error_json", None)
        try:
            result["error"] = json.loads(error_json) if error_json else None
        except (TypeError, ValueError):
            result["error"] = None
            result.pop("error_json", None)
        return result

    def prepare(self, raw: bytes) -> tuple[dict, bool]:
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("prompt request body must be valid JSON") from exc
        if not isinstance(body, dict) or not isinstance(body.get("prompt"), dict):
            raise ValueError("prompt request must contain a prompt object")

        source_hash = self._hash(body)
        client_id = str(body.get("client_id") or "") or None
        supplied_prompt_id = body.get("prompt_id")
        if supplied_prompt_id:
            try:
                prompt_id = str(uuid.UUID(str(supplied_prompt_id).strip().lower()))
            except (ValueError, AttributeError) as exc:
                raise ValueError("prompt_id must be a canonical UUID") from exc
            if prompt_id != str(supplied_prompt_id).strip().lower():
                raise ValueError("prompt_id must be a canonical UUID")
        else:
            prompt_id = ""
            if client_id:
                with self._lock:
                    recent = self._db.execute(
                        """
                        SELECT * FROM prompts
                        WHERE client_id=? AND source_hash=?
                          AND status NOT IN ('completed','failed','cancelled')
                          AND created_at >= ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (client_id, source_hash, time.time() - 30),
                    ).fetchone()
                existing = self._row(recent)
                if existing is not None:
                    existing["operations"] = self.operations(str(existing["prompt_id"]))
                    return existing, False
            prompt_id = str(uuid.uuid4())

        normalized = json.loads(self._canonical(body))
        normalized["prompt_id"] = prompt_id
        operations: list[tuple[str, str, str]] = []
        for node_id, node in normalized["prompt"].items():
            if not isinstance(node, dict):
                continue
            workload = H3_SAMPLER_WORKLOADS.get(str(node.get("class_type") or ""))
            if workload is None:
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                inputs = {}
                node["inputs"] = inputs
            operation_id = str(
                uuid.uuid5(
                    H3_OPERATION_NAMESPACE,
                    f"{prompt_id}:{node_id}:{workload}",
                )
            )
            inputs["_h3_operation_id"] = operation_id
            inputs["_graph_prompt_id"] = prompt_id
            inputs["_graph_node_id"] = str(node_id)
            operations.append((str(node_id), operation_id, workload))

        # Journal only the workflows that can be recovered idempotently.
        if not operations:
            return {
                "prompt_id": prompt_id,
                "request": normalized,
                "journaled": False,
            }, True

        request_hash = self._hash(normalized)
        now = time.time()
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT * FROM prompts WHERE prompt_id=?", (prompt_id,)
            ).fetchone()
            if row is not None:
                existing = self._row(row)
                assert existing is not None
                if existing["request_hash"] != request_hash:
                    raise PromptConflict(
                        "prompt_id is already bound to a different workflow request"
                    )
                existing["operations"] = self.operations(prompt_id)
                return existing, False
            self._db.execute(
                """
                INSERT INTO prompts(
                    prompt_id, source_hash, request_hash, client_id, request_json,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'accepted', ?, ?)
                """,
                (
                    prompt_id,
                    source_hash,
                    request_hash,
                    client_id,
                    self._canonical(normalized),
                    now,
                    now,
                ),
            )
            self._db.executemany(
                """
                INSERT INTO prompt_operations(prompt_id, node_id, operation_id, workload)
                VALUES (?, ?, ?, ?)
                """,
                [(prompt_id, node_id, operation_id, workload) for node_id, operation_id, workload in operations],
            )
            result = self.get(prompt_id)
            assert result is not None
            return result, True

    def get(self, prompt_id: str) -> dict | None:
        with self._lock:
            result = self._row(
                self._db.execute(
                    "SELECT * FROM prompts WHERE prompt_id=?", (prompt_id,)
                ).fetchone()
            )
            if result is not None:
                result["operations"] = self.operations(prompt_id)
            return result

    def operations(self, prompt_id: str, *, unacked_only: bool = False) -> list[dict]:
        with self._lock:
            where = "prompt_id=? AND acked_at IS NULL" if unacked_only else "prompt_id=?"
            rows = self._db.execute(
                f"SELECT * FROM prompt_operations WHERE {where} ORDER BY node_id",
                (prompt_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def nonterminal(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM prompts
                WHERE status NOT IN ('completed','failed','cancelled')
                ORDER BY created_at
                """
            ).fetchall()
            return [item for row in rows if (item := self._row(row)) is not None]

    def mark_forwarded(self, prompt_id: str) -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                """
                UPDATE prompts SET status='forwarded',
                    forwarded_at=COALESCE(forwarded_at, ?), updated_at=?
                WHERE prompt_id=? AND status NOT IN ('completed','failed','cancelled')
                """,
                (now, now, prompt_id),
            )

    def claim_forward(self, prompt_id: str, *, retry_after: float = 15) -> bool:
        """Atomically reserve one upstream forwarding attempt.

        A gateway request and the recovery loop may observe the same accepted
        prompt.  The claim timestamp prevents both from replaying it at once,
        while still allowing a new process to retry after a lost response.
        """

        now = time.time()
        with self._lock, self._db:
            cursor = self._db.execute(
                """
                UPDATE prompts
                SET status='forwarding', updated_at=?,
                    forward_attempts=forward_attempts+1
                WHERE prompt_id=?
                  AND status NOT IN ('completed','failed','cancelled')
                  AND (status != 'forwarding' OR updated_at <= ?)
                """,
                (now, prompt_id, now - retry_after),
            )
            return cursor.rowcount == 1

    def mark_terminal(self, prompt_id: str, status: str, error: object = None) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(f"invalid prompt terminal status: {status}")
        now = time.time()
        with self._lock, self._db:
            cursor = self._db.execute(
                """
                UPDATE prompts SET status=?, error_json=?, terminal_at=?, updated_at=?
                WHERE prompt_id=? AND status NOT IN ('completed','failed','cancelled')
                """,
                (
                    status,
                    None if error is None else self._canonical(error),
                    now,
                    now,
                    prompt_id,
                ),
            )
            if status == "cancelled" and cursor.rowcount == 1:
                self._db.execute(
                    """
                    UPDATE prompt_operations
                    SET cancel_requested_at=COALESCE(cancel_requested_at, ?)
                    WHERE prompt_id=? AND cancel_confirmed_at IS NULL
                    """,
                    (now, prompt_id),
                )

    def request_cancel(
        self, prompt_ids: list[str], *, reason: object = None
    ) -> list[dict]:
        now = time.time()
        unique_ids = list(dict.fromkeys(str(value) for value in prompt_ids if value))
        if not unique_ids:
            return []
        with self._lock, self._db:
            for prompt_id in unique_ids:
                cursor = self._db.execute(
                    """
                    UPDATE prompts
                    SET status='cancelled', error_json=?, terminal_at=COALESCE(terminal_at, ?),
                        updated_at=?
                    WHERE prompt_id=? AND status NOT IN ('completed','failed','cancelled')
                    """,
                    (
                        None if reason is None else self._canonical(reason),
                        now,
                        now,
                        prompt_id,
                    ),
                )
                status_row = self._db.execute(
                    "SELECT status FROM prompts WHERE prompt_id=?", (prompt_id,)
                ).fetchone()
                if cursor.rowcount == 1 or (
                    status_row is not None and status_row["status"] == "cancelled"
                ):
                    self._db.execute(
                        """
                        UPDATE prompt_operations
                        SET cancel_requested_at=COALESCE(cancel_requested_at, ?)
                        WHERE prompt_id=? AND cancel_confirmed_at IS NULL
                        """,
                        (now, prompt_id),
                    )
            placeholders = ",".join("?" for _ in unique_ids)
            rows = self._db.execute(
                f"""
                SELECT * FROM prompt_operations
                WHERE prompt_id IN ({placeholders})
                  AND cancel_requested_at IS NOT NULL
                  AND cancel_confirmed_at IS NULL
                ORDER BY prompt_id, node_id
                """,
                tuple(unique_ids),
            ).fetchall()
            return [dict(row) for row in rows]

    def pending_cancellations(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT * FROM prompt_operations
                WHERE cancel_requested_at IS NOT NULL
                  AND cancel_confirmed_at IS NULL
                ORDER BY cancel_requested_at, prompt_id, node_id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_cancel_confirmed(self, operation_id: str) -> None:
        now = time.time()
        with self._lock, self._db:
            self._db.execute(
                """
                UPDATE prompt_operations
                SET cancel_confirmed_at=COALESCE(cancel_confirmed_at, ?),
                    cancel_error_json=NULL
                WHERE operation_id=?
                """,
                (now, operation_id),
            )

    def mark_cancel_error(self, operation_id: str, error: object) -> None:
        with self._lock, self._db:
            self._db.execute(
                """
                UPDATE prompt_operations SET cancel_error_json=?
                WHERE operation_id=? AND cancel_confirmed_at IS NULL
                """,
                (self._canonical(error), operation_id),
            )

    def pending_ack_prompts(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT DISTINCT p.prompt_id
                FROM prompts AS p
                JOIN prompt_operations AS o ON o.prompt_id=p.prompt_id
                WHERE p.status IN ('completed','failed','cancelled')
                  AND o.acked_at IS NULL
                  AND (p.status != 'cancelled' OR o.cancel_confirmed_at IS NOT NULL)
                ORDER BY p.terminal_at
                """
            ).fetchall()
            return [str(row[0]) for row in rows]

    def acknowledgeable_operations(self, prompt_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT o.*
                FROM prompt_operations AS o
                JOIN prompts AS p ON p.prompt_id=o.prompt_id
                WHERE o.prompt_id=? AND o.acked_at IS NULL
                  AND p.status IN ('completed','failed','cancelled')
                  AND (p.status != 'cancelled' OR o.cancel_confirmed_at IS NOT NULL)
                ORDER BY o.node_id
                """,
                (prompt_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_acked(self, operation_ids: list[str]) -> None:
        if not operation_ids:
            return
        now = time.time()
        with self._lock, self._db:
            self._db.executemany(
                "UPDATE prompt_operations SET acked_at=COALESCE(acked_at, ?) WHERE operation_id=?",
                [(now, operation_id) for operation_id in operation_ids],
            )

    def active_count(self) -> int:
        with self._lock:
            return int(
                self._db.execute(
                    """
                    SELECT COUNT(*) FROM prompts
                    WHERE status NOT IN ('completed','failed','cancelled')
                    """
                ).fetchone()[0]
            )

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT status, COUNT(*) FROM prompts GROUP BY status"
            ).fetchall()
            return {str(row[0]): int(row[1]) for row in rows}


class GraphGateway:
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.lifecycle_lock = asyncio.Lock()
        self.monitor_task: asyncio.Task | None = None
        self.recovery_task: asyncio.Task | None = None
        self.last_generation_activity = time.monotonic()
        self.queue_was_active = False
        self.service_reachable = False
        self.next_stop_attempt = 0.0
        self.restart_failures = 0
        self.next_restart_attempt = 0.0
        self.cache = DiskCache(CACHE_DIR)
        self.journal = PromptJournal(STATE_DIR / "prompts.sqlite3")

    async def start(self, _app: web.Application) -> None:
        timeout = ClientTimeout(total=None, connect=15, sock_connect=15, sock_read=None)
        self.session = ClientSession(timeout=timeout, auto_decompress=False)
        queue = await self.probe_queue()
        self.service_reachable = queue is not None
        self.queue_was_active = self.queue_active(queue)
        self.monitor_task = asyncio.create_task(self.monitor_idle(), name="idle-monitor")
        self.recovery_task = asyncio.create_task(
            self.recover_prompts(), name="prompt-recovery"
        )
        LOG.info(
            "gateway_ready listen=%s:%d upstream=%s:%d idle_seconds=%d cache=%s service=%s",
            LISTEN_HOST,
            LISTEN_PORT,
            UPSTREAM_HOST,
            UPSTREAM_PORT,
            IDLE_SECONDS,
            CACHE_DIR,
            "running" if self.service_reachable else "unreachable",
        )

    async def close(self, _app: web.Application) -> None:
        if self.recovery_task:
            self.recovery_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.recovery_task
        if self.monitor_task:
            self.monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.monitor_task
        if self.session:
            await self.session.close()
        self.journal.close()

    @staticmethod
    def queue_active(queue: dict | None) -> bool:
        if not queue:
            return False
        return bool(queue.get("queue_running") or queue.get("queue_pending"))

    async def probe_queue(self, timeout_seconds: float = 5) -> dict | None:
        if not self.session:
            return None
        try:
            timeout = ClientTimeout(total=timeout_seconds)
            async with self.session.get(f"{UPSTREAM_HTTP}/queue", timeout=timeout) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                return data if isinstance(data, dict) else None
        except (asyncio.TimeoutError, ClientError, OSError, ValueError):
            return None

    async def remote_command(self, script: str, timeout_seconds: int) -> tuple[bool, str]:
        if CONTROL_URL:
            action = "start" if script == REMOTE_START else "stop"
            endpoint = "ensure" if action == "start" else "stop"
            if not self.session:
                return False, "HTTP session is unavailable"
            try:
                timeout = ClientTimeout(total=timeout_seconds)
                async with self.session.post(f"{CONTROL_URL}/{endpoint}", timeout=timeout) as response:
                    output = await response.text()
                    return 200 <= response.status < 300, output.strip()
            except (asyncio.TimeoutError, ClientError, OSError) as exc:
                return False, repr(exc)
        try:
            process = await asyncio.create_subprocess_exec(
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                REMOTE_TARGET,
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return False, repr(exc)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout_seconds)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return False, f"timeout after {timeout_seconds}s"
        output = stdout.decode("utf-8", errors="replace").strip()
        return process.returncode == 0, output

    def record_generation_activity(self, now: float | None = None) -> None:
        self.last_generation_activity = now if now is not None else time.monotonic()
        self.next_stop_attempt = 0.0

    def reset_restart_backoff(self) -> None:
        self.restart_failures = 0
        self.next_restart_attempt = 0.0

    def schedule_restart_retry(self) -> int:
        self.restart_failures += 1
        exponent = min(self.restart_failures - 1, 20)
        delay = min(RESTART_BACKOFF_SECONDS * (2**exponent), RESTART_MAX_BACKOFF_SECONDS)
        self.next_restart_attempt = time.monotonic() + delay
        return delay

    async def recover_service(self) -> bool:
        now = time.monotonic()
        if now < self.next_restart_attempt:
            return False
        async with self.lifecycle_lock:
            queue = await self.probe_queue()
            if queue is not None:
                self.service_reachable = True
                self.reset_restart_backoff()
                return True
            now = time.monotonic()
            if now < self.next_restart_attempt:
                return False
            LOG.warning("service_recovery_requested failures=%d", self.restart_failures)
            ok, output = await self.remote_command(REMOTE_START, 180)
            if not ok:
                self.service_reachable = False
                delay = self.schedule_restart_retry()
                LOG.error(
                    "service_recovery_failed retry_in_seconds=%d output=%s",
                    delay,
                    output or "<empty>",
                )
                return False
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                queue = await self.probe_queue()
                if queue is not None:
                    self.service_reachable = True
                    self.queue_was_active = self.queue_active(queue)
                    self.reset_restart_backoff()
                    LOG.info("service_recovery_completed output=%s", output or "<empty>")
                    return True
                await asyncio.sleep(1)
            self.service_reachable = False
            delay = self.schedule_restart_retry()
            LOG.error(
                "service_recovery_failed reason=upstream_health_timeout "
                "retry_in_seconds=%d output=%s",
                delay,
                output or "<empty>",
            )
            return False

    async def ensure_service(self) -> bool:
        queue = await self.probe_queue()
        if queue is not None:
            self.service_reachable = True
            self.reset_restart_backoff()
            return True
        self.service_reachable = False
        return await self.recover_service()

    async def stop_service_locked(self, reason: str) -> bool:
        LOG.info("service_stop_requested reason=%s", reason)
        ok, output = await self.remote_command(REMOTE_STOP, 60)
        if not ok:
            self.next_stop_attempt = time.monotonic() + UNLOAD_RETRY_SECONDS
            LOG.error(
                "service_stop_failed reason=%s retry_in_seconds=%d output=%s",
                reason,
                UNLOAD_RETRY_SECONDS,
                output or "<empty>",
            )
            return False
        self.service_reachable = False
        self.queue_was_active = False
        self.next_stop_attempt = 0.0
        LOG.info(
            "service_stop_completed reason=%s output=%s",
            reason,
            output or "<empty>",
        )
        return True

    async def stop_service(self, reason: str) -> bool:
        async with self.lifecycle_lock:
            return await self.stop_service_locked(reason)

    async def stop_service_if_idle(self) -> None:
        async with self.lifecycle_lock:
            now = time.monotonic()
            if (
                now < self.next_stop_attempt
                or now - self.last_generation_activity < IDLE_SECONDS
                or self.journal.active_count() > 0
            ):
                return
            queue = await self.probe_queue()
            if queue is None:
                self.service_reachable = False
                return
            self.service_reachable = True
            if self.queue_active(queue):
                self.queue_was_active = True
                self.record_generation_activity(now)
                return
            idle_for = int(now - self.last_generation_activity)
            await self.stop_service_locked(f"idle_{idle_for}s")

    async def monitor_idle(self) -> None:
        while True:
            await asyncio.sleep(QUEUE_POLL_SECONDS)
            queue = await self.probe_queue()
            if queue is None:
                self.service_reachable = False
                continue
            self.service_reachable = True
            self.reset_restart_backoff()
            now = time.monotonic()
            active = self.queue_active(queue) or self.journal.active_count() > 0
            if active:
                self.queue_was_active = True
                self.record_generation_activity(now)
            elif self.queue_was_active:
                self.queue_was_active = False
                self.record_generation_activity(now)
                LOG.info("generation_queue_became_idle")
            elif now - self.last_generation_activity >= IDLE_SECONDS:
                await self.stop_service_if_idle()

    @staticmethod
    def queued_prompt_ids(queue: dict | None) -> set[str]:
        return GraphGateway.queue_prompt_ids(queue, "queue_running", "queue_pending")

    @staticmethod
    def queue_prompt_ids(queue: dict | None, *names: str) -> set[str]:
        result: set[str] = set()
        if not isinstance(queue, dict):
            return result
        for name in names:
            items = queue.get(name)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) > 1:
                    result.add(str(item[1]))
                elif isinstance(item, dict) and item.get("prompt_id"):
                    result.add(str(item["prompt_id"]))
        return result

    async def prompt_history(self, prompt_id: str) -> dict | None:
        if not self.session:
            return None
        try:
            async with self.session.get(
                f"{UPSTREAM_HTTP}/history/{prompt_id}",
                timeout=ClientTimeout(total=5),
            ) as response:
                if response.status != 200:
                    return None
                payload = await response.json(content_type=None)
        except (asyncio.TimeoutError, ClientError, OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        record = payload.get(prompt_id)
        return record if isinstance(record, dict) else None

    @staticmethod
    def history_terminal_status(record: dict) -> tuple[str, object | None]:
        status = record.get("status")
        messages = status.get("messages") if isinstance(status, dict) else None
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, (list, tuple)) and message:
                    kind = str(message[0])
                    if kind == "execution_interrupted":
                        return "cancelled", message
                    if kind == "execution_error":
                        return "failed", message
        if isinstance(status, dict):
            status_text = str(status.get("status_str") or "").lower()
            if status_text in {"error", "failed"}:
                return "failed", status
        return "completed", None

    async def forward_journal_prompt(self, prompt: dict) -> bool:
        if not self.session:
            return False
        prompt_id = str(prompt["prompt_id"])
        if not self.journal.claim_forward(prompt_id):
            return False
        try:
            async with self.session.post(
                f"{UPSTREAM_HTTP}/prompt",
                json=prompt["request"],
                timeout=ClientTimeout(total=30),
            ) as response:
                body = await response.read()
                if 200 <= response.status < 300:
                    self.journal.mark_forwarded(prompt_id)
                    LOG.info("prompt_replayed prompt_id=%s", prompt_id)
                    return True
                if 400 <= response.status < 500:
                    self.journal.mark_terminal(
                        prompt_id,
                        "failed",
                        {
                            "upstream_status": response.status,
                            "body": body.decode("utf-8", errors="replace")[:4096],
                        },
                    )
                    LOG.error(
                        "prompt_replay_rejected prompt_id=%s status=%s",
                        prompt_id,
                        response.status,
                    )
                    return False
        except (asyncio.TimeoutError, ClientError, OSError) as exc:
            self.service_reachable = False
            LOG.warning("prompt_replay_transport_failed prompt_id=%s error=%r", prompt_id, exc)
        return False

    async def cancel_h3_operation(self, operation: dict) -> bool:
        if not self.session or not H3_CONTROL_URL:
            return False
        operation_id = str(operation.get("operation_id") or "")
        workload = str(operation.get("workload") or "")
        prefix = "h3" if workload == "h3" else "h3-ref" if workload == "h3_ref" else ""
        if not operation_id or not prefix:
            self.journal.mark_cancel_error(
                operation_id,
                {"error": "invalid durable H3 operation metadata", "workload": workload},
            )
            return False
        try:
            async with self.session.delete(
                f"{H3_CONTROL_URL.removesuffix('/h3-control')}/{prefix}/v1/videos/{operation_id}",
                timeout=ClientTimeout(total=10),
            ) as response:
                try:
                    payload = await response.json(content_type=None)
                except (ValueError, TypeError):
                    payload = {"raw": (await response.read()).decode("utf-8", errors="replace")[:4096]}
                if response.status == 404:
                    self.journal.mark_cancel_confirmed(operation_id)
                    return True
                state = str(payload.get("status") or "").lower() if isinstance(payload, dict) else ""
                if 200 <= response.status < 300 and state in {
                    "completed",
                    "failed",
                    "cancelled",
                    "canceled",
                }:
                    self.journal.mark_cancel_confirmed(operation_id)
                    return True
                self.journal.mark_cancel_error(
                    operation_id,
                    {"upstream_status": response.status, "response": payload},
                )
        except (asyncio.TimeoutError, ClientError, OSError) as exc:
            self.journal.mark_cancel_error(
                operation_id,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        return False

    async def cancel_prompt_ids(self, prompt_ids: list[str], *, reason: object) -> None:
        operations = self.journal.request_cancel(prompt_ids, reason=reason)
        for operation in operations:
            await self.cancel_h3_operation(operation)

    async def acknowledge_prompt_operations(self, prompt_id: str) -> None:
        if not self.session or not H3_CONTROL_URL:
            return
        operations = self.journal.acknowledgeable_operations(prompt_id)
        operation_ids = [str(item["operation_id"]) for item in operations]
        if not operation_ids:
            return
        try:
            async with self.session.post(
                f"{H3_CONTROL_URL}/v1/operations/ack",
                json={"operation_ids": operation_ids},
                timeout=ClientTimeout(total=10),
            ) as response:
                if 200 <= response.status < 300:
                    self.journal.mark_acked(operation_ids)
                    LOG.info(
                        "prompt_operations_acked prompt_id=%s operations=%d",
                        prompt_id,
                        len(operation_ids),
                    )
        except (asyncio.TimeoutError, ClientError, OSError):
            return

    async def recover_prompts(self) -> None:
        while True:
            try:
                for operation in self.journal.pending_cancellations():
                    await self.cancel_h3_operation(operation)
                pending = self.journal.nonterminal()
                if pending:
                    if await self.ensure_service():
                        queue = await self.probe_queue(timeout_seconds=5)
                        queued_ids = self.queued_prompt_ids(queue)
                        if queue is not None:
                            for prompt in pending:
                                prompt_id = str(prompt["prompt_id"])
                                history = await self.prompt_history(prompt_id)
                                if history is not None:
                                    terminal, error = self.history_terminal_status(history)
                                    self.journal.mark_terminal(prompt_id, terminal, error)
                                    continue
                                if (
                                    prompt_id not in queued_ids
                                    and time.time() - float(prompt.get("updated_at") or 0) >= 15
                                ):
                                    await self.forward_journal_prompt(prompt)
                for prompt_id in self.journal.pending_ack_prompts():
                    await self.acknowledge_prompt_operations(prompt_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("prompt_recovery_loop_failed")
            await asyncio.sleep(5)

    @staticmethod
    def request_headers(
        request: web.Request, websocket: bool = False, cacheable: bool = False
    ) -> CIMultiDict:
        headers = CIMultiDict()
        excluded = HOP_HEADERS | {"origin", "x-forwarded-prefix"}
        excluded |= WS_HANDSHAKE_HEADERS if websocket else {"content-length"}
        if cacheable:
            excluded.add("accept-encoding")
            excluded.add("range")
        for name, value in request.headers.items():
            if name.lower() not in excluded and name.lower() != "host":
                headers.add(name, value)
        return headers

    @staticmethod
    def upstream_path(request: web.Request) -> str:
        """Return the decoded path after removing a trusted reverse-proxy prefix."""
        path = request.path
        prefix = request.headers.get("X-Forwarded-Prefix", "").rstrip("/")
        if prefix and (path == prefix or path.startswith(f"{prefix}/")):
            path = path[len(prefix) :]
        return path or "/"

    @staticmethod
    def upstream_rel_url(request: web.Request) -> str:
        """Strip the public prefix without decoding escaped path separators."""
        raw_path = request.raw_path
        prefix = request.headers.get("X-Forwarded-Prefix", "").rstrip("/")
        if prefix and (raw_path == prefix or raw_path.startswith(f"{prefix}/")):
            raw_path = raw_path[len(prefix) :]
        return raw_path or "/"

    @staticmethod
    def response_headers(response) -> CIMultiDict:
        headers = CIMultiDict()
        for raw_name, raw_value in response.raw_headers:
            name = raw_name.decode("latin-1")
            if name.lower() not in HOP_HEADERS:
                headers.add(name, raw_value.decode("latin-1"))
        return headers

    @staticmethod
    def is_generation_request(method: str, upstream_path: str) -> bool:
        return method == "POST" and upstream_path.rstrip("/") in {
            "/prompt",
            "/api/prompt",
        }

    @staticmethod
    def is_free_request(method: str, upstream_path: str) -> bool:
        return method == "POST" and upstream_path.rstrip("/") in {"/free", "/api/free"}

    @staticmethod
    def is_cancellation_request(method: str, upstream_path: str) -> bool:
        path = upstream_path.rstrip("/")
        return method == "POST" and path in {
            "/queue",
            "/api/queue",
            "/interrupt",
            "/api/interrupt",
        }

    async def status(self, _request: web.Request) -> web.Response:
        queue = await self.probe_queue(timeout_seconds=2)
        reachable = queue is not None
        if reachable:
            self.service_reachable = True
        idle_for = max(0, int(time.monotonic() - self.last_generation_activity))
        restart_in = max(0, int(self.next_restart_attempt - time.monotonic()))
        return web.json_response(
            {
                "gateway": "running",
                "listen": f"{LISTEN_HOST}:{LISTEN_PORT}",
                "upstream": f"{UPSTREAM_HOST}:{UPSTREAM_PORT}",
                "service": "running" if reachable else "sleeping_or_unreachable",
                "queue_active": self.queue_active(queue),
                "journal_active": self.journal.active_count(),
                "journal_counts": self.journal.counts(),
                "idle_seconds": idle_for,
                "idle_limit_seconds": IDLE_SECONDS,
                "idle_remaining_seconds": max(0, IDLE_SECONDS - idle_for),
                "restart_failures": self.restart_failures,
                "restart_in_seconds": restart_in,
                "recovering": self.lifecycle_lock.locked() and not reachable,
            }
        )

    @staticmethod
    def unavailable() -> web.Response:
        return web.json_response(
            {"error": "graph service is starting or temporarily unreachable", "retry_after_seconds": 2},
            status=503,
            headers={"Retry-After": "2"},
        )

    async def cached_static(self, request: web.Request, policy: str) -> web.Response:
        rel_url = self.upstream_rel_url(request)
        cached = await self.cache.get(rel_url)
        if cached is not None:
            body, headers = cached
            headers.update({"Cache-Control": policy, "X-Graph-Cache": "HIT"})
            return web.Response(body=body, headers=headers)

        async with self.cache.lock(rel_url):
            cached = await self.cache.get(rel_url)
            if cached is not None:
                body, headers = cached
                headers.update({"Cache-Control": policy, "X-Graph-Cache": "HIT"})
                return web.Response(body=body, headers=headers)
            assert self.session is not None
            try:
                async with self.session.get(
                    f"{UPSTREAM_HTTP}{rel_url}",
                    headers=self.request_headers(request, cacheable=True),
                    allow_redirects=False,
                ) as upstream:
                    body = await upstream.read()
                    if upstream.status != 200:
                        return web.Response(
                            status=upstream.status,
                            body=body,
                            headers=self.response_headers(upstream),
                        )
                    headers = {
                        name: value
                        for name, value in upstream.headers.items()
                        if name.lower() in CACHED_HEADERS
                    }
                    await self.cache.put(rel_url, body, headers)
                    headers.update({"Cache-Control": policy, "X-Graph-Cache": "MISS"})
                    return web.Response(body=body, headers=headers)
            except (asyncio.TimeoutError, ClientError, OSError) as exc:
                self.service_reachable = False
                LOG.warning("static_proxy_failed path=%s error=%r", request.path, exc)
                return self.unavailable()

    async def proxy_http(self, request: web.Request) -> web.StreamResponse:
        upstream_path = self.upstream_path(request)
        upstream_rel_url = self.upstream_rel_url(request)
        ui_open = request.method == "GET" and upstream_path in {"/", "/index.html"}
        generation_request = self.is_generation_request(request.method, upstream_path)
        cancellation_request = self.is_cancellation_request(
            request.method, upstream_path
        )
        if ui_open:
            self.record_generation_activity()
            LOG.info("ui_open_received")
        if generation_request:
            self.record_generation_activity()
            LOG.info("generation_request_received")

        journal_prompt: dict | None = None
        journal_created = True
        forwarded_body: bytes | None = None
        queue_before: dict | None = None
        cancellation_targets: set[str] = set()
        if cancellation_request:
            queue_before = await self.probe_queue(timeout_seconds=2)
            path = upstream_path.rstrip("/")
            if path in {"/queue", "/api/queue"}:
                forwarded_body = await request.read()
                try:
                    command = json.loads(forwarded_body or b"{}")
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    command = {}
                if isinstance(command, dict):
                    deleted = command.get("delete")
                    if isinstance(deleted, list):
                        cancellation_targets.update(str(value) for value in deleted)
                    if command.get("clear") is True:
                        cancellation_targets.update(
                            self.queue_prompt_ids(queue_before, "queue_pending")
                        )
            else:
                cancellation_targets.update(
                    self.queue_prompt_ids(queue_before, "queue_running")
                )
                if queue_before is None and not cancellation_targets:
                    active = self.journal.nonterminal()
                    if len(active) == 1:
                        cancellation_targets.add(str(active[0]["prompt_id"]))
            if cancellation_targets:
                await self.cancel_prompt_ids(
                    sorted(cancellation_targets),
                    reason={"source": "graph_control", "path": upstream_path},
                )
            if queue_before is None and cancellation_targets:
                return web.json_response(
                    {
                        "status": "cancel_requested",
                        "prompt_ids": sorted(cancellation_targets),
                    },
                    status=202,
                )
        if generation_request:
            try:
                journal_prompt, journal_created = self.journal.prepare(await request.read())
            except PromptConflict as exc:
                return web.json_response({"error": str(exc)}, status=409)
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=400)
            forwarded_body = self.journal._canonical(journal_prompt["request"]).encode("utf-8")
            if journal_prompt.get("operations") and not journal_created:
                return web.json_response(
                    {
                        "prompt_id": journal_prompt["prompt_id"],
                        "number": -1,
                        "node_errors": {},
                        "gateway_status": journal_prompt.get("status", "accepted"),
                    }
                )

        policy = self.cache.policy(upstream_path)
        if request.method in {"GET", "HEAD"} and policy and "range" not in request.headers:
            if ui_open and not await self.ensure_service():
                return self.unavailable()
            return await self.cached_static(request, policy)
        if generation_request and not await self.ensure_service():
            if journal_prompt and journal_prompt.get("operations"):
                return web.json_response(
                    {
                        "prompt_id": journal_prompt["prompt_id"],
                        "number": -1,
                        "node_errors": {},
                        "gateway_status": "accepted",
                    },
                    status=202,
                )
            return self.unavailable()

        if (
            journal_prompt
            and journal_prompt.get("operations")
            and not self.journal.claim_forward(str(journal_prompt["prompt_id"]))
        ):
            return web.json_response(
                {
                    "prompt_id": journal_prompt["prompt_id"],
                    "number": -1,
                    "node_errors": {},
                    "gateway_status": "forwarding",
                },
                status=202,
            )

        assert self.session is not None
        body = (
            forwarded_body
            if forwarded_body is not None
            else request.content.iter_chunked(1024 * 1024) if request.can_read_body else None
        )
        prepared = False
        try:
            async with self.session.request(
                request.method,
                f"{UPSTREAM_HTTP}{upstream_rel_url}",
                headers=self.request_headers(request),
                data=body,
                allow_redirects=False,
            ) as upstream:
                if journal_prompt and journal_prompt.get("operations"):
                    if 200 <= upstream.status < 300:
                        self.journal.mark_forwarded(str(journal_prompt["prompt_id"]))
                headers = self.response_headers(upstream)
                headers["X-Graph-Cache"] = "BYPASS"
                response = web.StreamResponse(
                    status=upstream.status, reason=upstream.reason, headers=headers
                )
                await response.prepare(request)
                prepared = True
                async for chunk in upstream.content.iter_chunked(1024 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                if journal_prompt and journal_prompt.get("operations"):
                    if 400 <= upstream.status < 500:
                        self.journal.mark_terminal(
                            str(journal_prompt["prompt_id"]),
                            "failed",
                            {"upstream_status": upstream.status},
                        )
                if (
                    self.is_free_request(request.method, upstream_path)
                    and 200 <= upstream.status < 300
                ):
                    await self.stop_service("explicit_free")
                return response
        except (asyncio.TimeoutError, ClientError, OSError) as exc:
            self.service_reachable = False
            LOG.warning("http_proxy_failed method=%s path=%s error=%r", request.method, request.path, exc)
            if not prepared:
                return self.unavailable()
            raise

    async def proxy_websocket(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None
        protocols = [
            item.strip()
            for item in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
            if item.strip()
        ]
        try:
            upstream = await self.session.ws_connect(
                f"{UPSTREAM_WS}{self.upstream_rel_url(request)}",
                headers=self.request_headers(request, websocket=True),
                protocols=protocols,
                autoping=True,
                heartbeat=20,
            )
        except (asyncio.TimeoutError, ClientError, OSError) as exc:
            self.service_reachable = False
            LOG.warning(
                "websocket_upstream_connect_failed path=%s error=%s status=%s",
                request.path,
                type(exc).__name__,
                getattr(exc, "status", "n/a"),
            )
            return self.unavailable()

        browser = web.WebSocketResponse(protocols=protocols, autoping=True, heartbeat=20)
        await browser.prepare(request)

        async def browser_to_upstream() -> None:
            async for message in browser:
                if message.type == WSMsgType.TEXT:
                    await upstream.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await upstream.send_bytes(message.data)
                elif message.type == WSMsgType.CLOSE:
                    break

        async def upstream_to_browser() -> None:
            async for message in upstream:
                if message.type == WSMsgType.TEXT:
                    await browser.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await browser.send_bytes(message.data)
                elif message.type == WSMsgType.CLOSE:
                    break

        tasks = {
            asyncio.create_task(browser_to_upstream()),
            asyncio.create_task(upstream_to_browser()),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done | pending:
            with suppress(asyncio.CancelledError, ConnectionError, RuntimeError):
                await task
        await upstream.close()
        await browser.close()
        return browser

    async def handle(self, request: web.Request) -> web.StreamResponse:
        upstream_path = self.upstream_path(request)
        if upstream_path == STATUS_PATH:
            return await self.status(request)
        if request.method in {"GET", "DELETE"} and upstream_path.startswith(
            PROMPT_STATUS_PREFIX
        ):
            prompt_id = upstream_path[len(PROMPT_STATUS_PREFIX) :].strip("/")
            try:
                prompt_id = str(uuid.UUID(prompt_id))
            except ValueError:
                return web.json_response({"error": "invalid prompt id"}, status=400)
            prompt = self.journal.get(prompt_id)
            if prompt is None:
                return web.json_response({"error": "prompt not found"}, status=404)
            if request.method == "DELETE":
                await self.cancel_prompt_ids(
                    [prompt_id],
                    reason={"source": "graph_gateway_api", "path": upstream_path},
                )
                prompt = self.journal.get(prompt_id)
                assert prompt is not None
            return web.json_response(prompt)
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self.proxy_websocket(request)
        return await self.proxy_http(request)


def build_app() -> web.Application:
    gateway = GraphGateway()
    app = web.Application(client_max_size=1024**4)
    app.router.add_route("*", "/{path_info:.*}", gateway.handle)
    app.on_startup.append(gateway.start)
    app.on_cleanup.append(gateway.close)
    return app


if __name__ == "__main__":
    web.run_app(
        build_app(),
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        access_log=None,
        print=None,
        handle_signals=True,
    )
