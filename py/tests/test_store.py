import asyncio
from pathlib import Path

from ico.collector.store import Writer, split_statements

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


class FakeClient:
    """Records inserts; fails while ``down`` is set."""

    def __init__(self) -> None:
        self.inserts: list[tuple[str, list, tuple]] = []
        self.down = False

    def insert(self, table, data, column_names):
        if self.down:
            raise ConnectionError("clickhouse is down")
        self.inserts.append((table, list(data), tuple(column_names)))

    def rows(self, table: str) -> list:
        return [row for t, data, _ in self.inserts if t == table for row in data]


def new_writer(client: FakeClient, max_rows: int = 1000) -> Writer:
    return Writer(client, max_buffered_rows=max_rows, flush_interval_s=0.01)


def test_split_statements_ignores_semicolons_in_comments():
    sql = """
    -- first table; note the semicolon in this comment
    CREATE TABLE a (x UInt8) ENGINE = Memory;
    CREATE TABLE b (y UInt8) ENGINE = Memory; -- trailing; comment
    """
    assert split_statements(sql) == [
        "CREATE TABLE a (x UInt8) ENGINE = Memory",
        "CREATE TABLE b (y UInt8) ENGINE = Memory",
    ]


def test_shipped_schema_splits_into_one_statement_per_object():
    statements = split_statements((SCHEMA_DIR / "001_init.sql").read_text(encoding="utf-8"))
    created = [" ".join(s.split()[:6]) for s in statements]
    assert created == [
        "CREATE TABLE IF NOT EXISTS requests",
        "CREATE TABLE IF NOT EXISTS engine_spans",
        "CREATE VIEW IF NOT EXISTS request_spans",
        "CREATE TABLE IF NOT EXISTS engine_metrics",
        "CREATE TABLE IF NOT EXISTS gpu_metrics",
        "CREATE TABLE IF NOT EXISTS replay_runs",
        "CREATE TABLE IF NOT EXISTS replay_results",
    ]


def test_flush_writes_each_table_with_its_columns():
    client = FakeClient()
    w = new_writer(client)
    w.add("requests", ("a", "b"), [(1, 2), (3, 4)])
    w.add("gpu_metrics", ("x",), [(9,)])
    asyncio.run(w.flush())

    assert client.inserts == [
        ("requests", [(1, 2), (3, 4)], ("a", "b")),
        ("gpu_metrics", [(9,)], ("x",)),
    ]
    assert w.written == 3
    asyncio.run(w.flush())
    assert len(client.inserts) == 2, "an empty buffer must not produce an insert"


def test_rows_survive_an_outage_in_order():
    client = FakeClient()
    w = new_writer(client)
    client.down = True
    w.add("requests", ("n",), [(1,), (2,)])
    asyncio.run(w.flush())
    w.add("requests", ("n",), [(3,)])
    asyncio.run(w.flush())
    assert client.inserts == [] and w.written == 0 and w.dropped == 0

    client.down = False
    asyncio.run(w.flush())
    assert client.rows("requests") == [(1,), (2,), (3,)]
    assert w.written == 3


def test_overflow_drops_the_oldest_rows_and_counts_them():
    client = FakeClient()
    w = new_writer(client, max_rows=5)
    client.down = True
    w.add("requests", ("n",), [(i,) for i in range(8)])
    asyncio.run(w.flush())  # fails; rows go back, the bound still holds
    w.add("requests", ("n",), [(8,), (9,)])
    assert w.dropped == 5

    client.down = False
    asyncio.run(w.flush())
    assert client.rows("requests") == [(5,), (6,), (7,), (8,), (9,)]


def test_run_flushes_on_cancel():
    client = FakeClient()
    w = Writer(client, max_buffered_rows=10, flush_interval_s=3600)

    async def scenario():
        task = asyncio.create_task(w.run())
        await asyncio.sleep(0)
        w.add("requests", ("n",), [(1,)])
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert client.rows("requests") == [(1,)], "shutdown must not lose buffered rows"
