"""ClickHouse access for the collector: schema migrations and a batching writer."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from pathlib import Path
from typing import Any, Protocol, Sequence

import clickhouse_connect

from ico.config import ClickHouse

log = logging.getLogger(__name__)

Row = Sequence[Any]

# Largest single insert. After an outage the backlog goes out in pieces.
_MAX_INSERT_ROWS = 50_000


class Client(Protocol):
    """The slice of clickhouse_connect's client the collector uses."""

    def command(self, cmd: str, *args: Any, **kwargs: Any) -> Any: ...

    def query(self, query: str, *args: Any, **kwargs: Any) -> Any: ...

    def insert(self, table: str, data: Sequence[Row], column_names: Sequence[str]) -> Any: ...


def connect(cfg: ClickHouse, *, database: str | None = None) -> Client:
    return clickhouse_connect.get_client(
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password,
        database=cfg.database if database is None else database,
        secure=cfg.secure,
    )


def split_statements(sql: str) -> list[str]:
    """Split a migration file into statements.

    Line comments go first, because a comment may contain a semicolon. The DDL
    in schema/ has no semicolons inside string literals, so a plain split on
    the rest is enough.
    """
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def migrate(cfg: ClickHouse, schema_dir: Path) -> list[str]:
    """Apply the .sql files in schema_dir that this database has not seen.

    Files run in name order and are recorded by name, so a file is applied
    once and must never be edited after it has shipped. Returns the names
    applied by this call.
    """
    files = sorted(schema_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no .sql files in {schema_dir}")

    # The target database may not exist yet, so the first statement must not
    # name it as the session database.
    connect(cfg, database="default").command(f"CREATE DATABASE IF NOT EXISTS `{cfg.database}`")
    client = connect(cfg)
    client.command(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(name String, applied_at DateTime DEFAULT now()) ENGINE = MergeTree ORDER BY name"
    )
    done = {row[0] for row in client.query("SELECT name FROM schema_migrations").result_rows}

    applied = []
    for f in files:
        if f.name in done:
            continue
        for statement in split_statements(f.read_text(encoding="utf-8")):
            client.command(statement)
        client.insert("schema_migrations", [[f.name]], column_names=["name"])
        applied.append(f.name)
        log.info("applied migration %s", f.name)
    return applied


class Writer:
    """Buffers rows per table and writes them to ClickHouse in batches.

    ``add`` never blocks and never raises on a full buffer: while ClickHouse
    is unreachable rows wait in memory, and past ``max_buffered_rows`` the
    oldest are dropped and counted. Telemetry is the thing to lose, never the
    caller's time.
    """

    def __init__(self, client: Client, *, max_buffered_rows: int, flush_interval_s: float) -> None:
        self._client = client
        self._max = max_buffered_rows
        self._interval = flush_interval_s
        self._columns: dict[str, Sequence[str]] = {}
        self._rows: dict[str, deque[Row]] = {}
        self._buffered = 0
        self.written = 0
        self.dropped = 0
        self._failing = False

    def add(self, table: str, columns: Sequence[str], rows: Sequence[Row]) -> None:
        self._columns[table] = columns
        buffer = self._rows.setdefault(table, deque())
        buffer.extend(rows)
        self._buffered += len(rows)
        while self._buffered > self._max:
            # Shed from the fullest table so one noisy source cannot starve the rest.
            fullest = max(self._rows.values(), key=len)
            fullest.popleft()
            self._buffered -= 1
            self.dropped += 1

    async def run(self) -> None:
        """Flush on a timer until cancelled, then flush once more."""
        try:
            while True:
                await asyncio.sleep(self._interval)
                await self.flush()
        finally:
            await self.flush()

    async def flush(self) -> None:
        for table, buffer in self._rows.items():
            while buffer:
                # The batch leaves the buffer before the insert, so add() can
                # keep appending and shedding while the insert runs.
                batch = [buffer.popleft() for _ in range(min(len(buffer), _MAX_INSERT_ROWS))]
                self._buffered -= len(batch)
                try:
                    await asyncio.to_thread(self._client.insert, table, batch, column_names=self._columns[table])
                except Exception as e:  # noqa: BLE001 - any failure means "keep the rows and retry"
                    buffer.extendleft(reversed(batch))
                    self._buffered += len(batch)
                    if not self._failing:
                        log.warning("ClickHouse write failed, rows are kept for retry: %s", e)
                        self._failing = True
                    return
                if self._failing:
                    log.info("ClickHouse writes are working again")
                    self._failing = False
                self.written += len(batch)
