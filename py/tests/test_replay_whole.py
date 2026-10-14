"""The replay lab across real boundaries: the watchdog as its own process, and the command run whole."""

import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ico.config import Model
from ico.replay import results, roofline, watchdog
from ico.replay.load import StepResult
from ico.replay.run import FAILED_TO_START, FINISHED, RULED_OUT, SKIPPED, TESTED, RunResult, ConfigResult
from ico.replay.sweep import Candidate
from tests.http_fakes import TOKEN, FakeInstall

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"


@pytest.fixture
def install():
    fake = FakeInstall()
    yield fake
    fake.close()


def wait_for(condition, what, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.1)


# Roofline -------------------------------------------------------------------

QWEN = Model(params=751_650_164, weight_bytes_per_param=2, kv_bytes_per_token=114_688)
SIX_GIB = 6 * 2**30


def test_roofline_rules_out_a_memory_fraction_too_small_for_the_weights():
    # 1.4 GiB of weights do not fit in 20% of 6 GiB.
    reason = roofline.rule_out({"gpu-memory-utilization": 0.2}, QWEN, SIX_GIB, gpu_count=1)
    assert reason == "the weights need 1.4 GiB and this memory fraction allows 1.2 GiB"
    assert roofline.rule_out({"gpu-memory-utilization": 0.6}, QWEN, SIX_GIB, gpu_count=1) is None


def test_roofline_needs_room_for_one_full_context():
    # 30% of 6 GiB is 1.8 GiB: the weights fit, but 4,096 tokens of KV cache (0.4 GiB) no longer do.
    flags = {"gpu-memory-utilization": 0.3, "max-model-len": 4096}
    assert "one full context of KV cache (0.4 GiB)" in roofline.rule_out(flags, QWEN, SIX_GIB, gpu_count=1)
    assert roofline.rule_out({**flags, "max-model-len": 1024}, QWEN, SIX_GIB, gpu_count=1) is None
    # FP8 halves the weights, which makes room again.
    assert roofline.rule_out({**flags, "quantization": "fp8"}, QWEN, SIX_GIB, gpu_count=1) is None


def test_roofline_lets_everything_through_when_it_has_nothing_to_go_on():
    tiny = {"gpu-memory-utilization": 0.01}
    assert roofline.rule_out(tiny, None, SIX_GIB, 1) is None, "a model missing from the config's registry"
    assert roofline.rule_out(tiny, QWEN, None, 1) is None, "a host that never reported its memory size"


# Engine client ---------------------------------------------------------------


def test_time_to_first_token_does_not_count_opening_the_connection(install):
    import asyncio

    from ico.replay.engine import Engine
    from ico.replay.prompts import Prompt

    # By name, not by address: on Windows "localhost" tries IPv6 first and
    # takes about 300 ms to fall back, on every new connection. The first
    # real sweep counted that as engine latency and saw every config break.
    url = install.engine_url.replace("127.0.0.1", "localhost")
    install.service_s = 0.06  # 30 ms to the first token

    async def go():
        outcomes = []
        for _ in range(3):
            async with Engine(url) as engine:  # a fresh client, so a fresh connection each time
                outcomes.append(await engine.complete("fake-model", Prompt((1, 2, 3), 3)))
        return outcomes

    outcomes = asyncio.run(go())
    assert all(o.ok for o in outcomes)
    assert max(o.ttft_s for o in outcomes) < 0.2, [round(o.ttft_s, 3) for o in outcomes]


# Summary --------------------------------------------------------------------


def outcome_of(name, rate=None, cost=None, **more):
    flags = dict(kv.split("=") for kv in name.split(",")) if "=" in name else {}
    best = StepResult(rate, 40, 0, 300.0, 20.0, True, 10.0) if rate else None
    return ConfigResult(Candidate(name, flags, flags), more.pop("outcome", TESTED), best=best, tokens_s=(rate or 0) * 250, cost_per_mtok=cost, **more)


def test_summary_names_the_winner_and_what_it_saves():
    result = RunResult(
        uuid.uuid4(),
        FINISHED,
        (
            outcome_of("current", 12.4, 0.0090, is_baseline=True),
            outcome_of("gpu-memory-utilization=0.7", 15.5, 0.0072),
            outcome_of("quantization=fp8", 20.0, 0.0056),
            outcome_of("max-num-seqs=4", outcome=TESTED, note="held the limits at no rate tried"),
            outcome_of("gpu-memory-utilization=0.2", outcome=RULED_OUT, note="the weights need 1.4 GiB and this memory fraction allows 1.2 GiB"),
            outcome_of("max-num-seqs=512", outcome=FAILED_TO_START, note="the engine did not answer its health check within 120s"),
            outcome_of("max-num-seqs=96", outcome=SKIPPED, note="the time budget ran out"),
        ),
    )
    text = results.summary(result)
    lines = text.splitlines()
    assert "cheapest" in next(line for line in lines if line.startswith("gpu-memory-utilization=0.7"))
    assert "quality unverified" in next(line for line in lines if line.startswith("quantization=fp8"))
    assert next(line for line in lines if line.startswith("max-num-seqs=4 ")).endswith("held the limits at no rate tried")
    assert "ruled out: the weights need 1.4 GiB" in text
    assert "did not start: the engine did not answer" in text
    assert "not tested: the time budget ran out" in text
    assert (
        "Cheapest config that holds the limits: gpu-memory-utilization=0.7, $0.0072 per million tokens "
        "at 15.50 requests/s, 20% below the current config ($0.0090)." in text
    )
    assert "Configs that change precision are measured but cannot win" in text


def test_summary_says_so_when_the_current_config_is_already_best():
    result = RunResult(uuid.uuid4(), FINISHED, (outcome_of("current", 12.4, 0.0090, is_baseline=True), outcome_of("max-num-seqs=8", 6.0, 0.0186)))
    assert "The current config is the cheapest of those tested: $0.0090 per million tokens at 12.40 requests/s." in results.summary(result)


# Watchdog -------------------------------------------------------------------

PARENT = textwrap.dedent(
    """
    import json, os, sys, urllib.request
    from ico.config import LauncherConfig
    from ico.replay import watchdog

    engine_url, admin_url, token, launcher, last_words = sys.argv[1:6]
    dog = watchdog.spawn(
        replica="r1", engine_url=engine_url, admin_url=admin_url, admin_token=token,
        launcher=LauncherConfig(**json.loads(launcher), flags={"max-num-seqs": 2}),
        original={"max-num-seqs": 2}, health_timeout_s=5.0,
    )
    print(dog.log_path, flush=True)
    for state in last_words.split(","):
        if state == "done":
            dog.done()
        elif state:
            dog.report(state)
    if "stopped" in last_words:
        urllib.request.urlopen(engine_url + "/_stop")   # the engine is down, or on a test config
    os._exit(1)   # dies on the spot: no cleanup, no goodbye
    """
)


def die_after(install: FakeInstall, reports: str) -> Path:
    """Run a stand-in replay lab that reports these states to its watchdog and then dies. Returns the watchdog's log."""
    done = subprocess.run(
        [sys.executable, "-c", PARENT, install.engine_url, install.admin_url, TOKEN, json.dumps(install.launcher()), reports],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert done.returncode == 1, done.stderr
    return Path(done.stdout.strip())


def test_watchdog_restarts_and_undrains_when_the_replay_lab_dies_mid_test(install):
    install.drained["r1"] = True
    log = die_after(install, "drained,stopped")

    wait_for(lambda: install.drained["r1"] is False, "the watchdog to undrain r1")
    assert install.starts == ["--max-num-seqs 2"], "the engine was started once, with its original flags"
    assert install.engine_flags == "--max-num-seqs 2"
    assert "POST /admin/replicas/r1/undrain" in install.admin_calls
    wait_for(lambda: "r1 is back in service" in log.read_text(), "the watchdog to log the recovery")


def test_watchdog_only_undrains_when_the_engine_was_never_stopped(install):
    install.drained["r1"] = True
    die_after(install, "drained")
    wait_for(lambda: install.drained["r1"] is False, "the watchdog to undrain r1")
    assert install.starts == [], "an engine that was never stopped is not restarted"


def test_watchdog_stands_down_after_a_clean_finish(install):
    install.drained["r1"] = True  # as if left drained on purpose
    log = die_after(install, "drained,stopped,restored,done")
    time.sleep(1.5)
    assert install.drained["r1"] is True and install.starts == [], "after done, the watchdog must not touch anything"
    assert not log.exists() or "went away" not in log.read_text()


def test_watchdog_leaves_a_replica_drained_when_the_original_will_not_start(install):
    install.drained["r1"] = True
    install.broken_seqs = {2}
    log = die_after(install, "drained,stopped")
    wait_for(lambda: log.exists() and "stays drained" in log.read_text(), "the watchdog to give up", timeout_s=40)
    assert install.drained["r1"] is True, "no traffic for an engine that is down"


def test_watchdog_cannot_restart_for_a_manual_launcher_and_says_so(install, monkeypatch):
    monkeypatch.setenv("ICO_ADMIN_TOKEN", TOKEN)
    install.drained["r1"] = True
    state = {"replica": "r1", "engine_url": install.engine_url, "admin_url": install.admin_url,
             "launcher": {"kind": "manual", "start": "", "stop": "", "env": {}, "cwd": "", "flags": {}},
             "original": {}, "health_timeout_s": 5.0}  # fmt: skip
    watchdog.recover(state, watchdog.STOPPED)
    assert install.drained["r1"] is True and install.admin_calls == []


# The command, whole ---------------------------------------------------------

needs_clickhouse = pytest.mark.skipif("ICO_TEST_CLICKHOUSE_HOST" not in os.environ, reason="no ClickHouse to test against")


@pytest.mark.clickhouse
@needs_clickhouse
def test_replay_command_end_to_end(install, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from ico.cli import main
    from ico.collector import store
    from ico.collector.adapters.gpu import GPU_METRICS_COLUMNS
    from ico.collector.ingest import REQUESTS_COLUMNS, to_row
    from ico.config import ClickHouse
    from tests.test_ingest import sample_record

    ch = ClickHouse(
        host=os.environ["ICO_TEST_CLICKHOUSE_HOST"],
        port=int(os.environ.get("ICO_TEST_CLICKHOUSE_PORT", "8123")),
        username=os.environ.get("ICO_TEST_CLICKHOUSE_USER", "default"),
        password=os.environ.get("ICO_TEST_CLICKHOUSE_PASSWORD", ""),
        database=f"ico_test_{uuid.uuid4().hex[:12]}",
    )
    try:
        store.migrate(ch, SCHEMA_DIR)
        client = store.connect(ch)
        noon = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
        nanos = int(noon.timestamp()) * 1_000_000_000
        # An hour of recorded traffic: 80 requests, two in three sharing a long system prompt.
        client.insert("requests", [
            to_row(sample_record(
                request_id=f"q{i}", replica="r1", model="fake-model", status=200,
                arrival_unix_nanos=nanos + i * 40 * 1_000_000_000,
                prompt_tokens=64, completion_tokens=3, prompt_chars=256,
                prefix_hashes=[7, 8] if i % 3 else [100 + i],
            ))
            for i in range(80)
        ], column_names=REQUESTS_COLUMNS)  # fmt: skip
        client.insert("gpu_metrics", [(datetime.now(timezone.utc), "gpu-1", "GPU-1", "Fake 6GB", "memory_total_bytes", SIX_GIB, 2)], column_names=GPU_METRICS_COLUMNS)

        config = {
            "replicas": [
                {"name": "r1", "url": install.engine_url, "host": "gpu-1", "gpu_model": "L4", "gpu_count": 1,
                 "launcher": {**install.launcher(), "flags": {"max-num-seqs": 2}}},
                {"name": "r2", "url": "http://elsewhere:8000", "host": "gpu-2", "gpu_model": "L4"},
            ],
            "gpu_price_per_hour": {"L4": 0.5},
            "models": {"fake-model": {"params": QWEN.params, "weight_bytes_per_param": 2, "kv_bytes_per_token": 114_688}},
            "gateway": {"admin_url": install.admin_url, "prefix_hash": {"block_chars": 128}},
            "clickhouse": {"host": ch.host, "port": ch.port, "username": ch.username, "database": ch.database},
        }  # fmt: skip
        (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf-8")  # JSON is YAML
        (tmp_path / "sweep.yaml").write_text(
            "configs:\n"
            "  - max-num-seqs: 6\n"
            "  - max-num-seqs: 13\n"  # the fake engine will refuse to start with this
            "  - gpu-memory-utilization: 0.2\n",  # cannot hold the weights: ruled out on paper
            encoding="utf-8",
        )
        install.broken_seqs = {13}
        monkeypatch.setenv("ICO_ADMIN_TOKEN", TOKEN)
        monkeypatch.setenv("ICO_CLICKHOUSE_PASSWORD", ch.password)
        monkeypatch.delenv("ICO_CLICKHOUSE_HOST", raising=False)
        monkeypatch.delenv("ICO_GATEWAY_ADMIN", raising=False)

        outcome = CliRunner().invoke(main, [
            "replay", "--config", str(tmp_path / "config.yaml"), "--replica", "r1", "--configs", str(tmp_path / "sweep.yaml"),
            "--window", "60m", "--until", "2026-10-14T13:00", "--budget", "10m", "--yes",
            "--slo", "ttft_p95=200ms,itl_p95=500ms", "--step-seconds", "0.5", "--min-step-requests", "12", "--health-timeout", "4s",
        ])  # fmt: skip
        out = outcome.output
        assert outcome.exit_code == 0, out

        # The workload was read from the window and rebuilt with its prefix sharing.
        assert "r1 serves fake-model. 80 requests in the window; the replay uses a run of 80." in out
        assert "synthetic prompts repeat the start of an earlier prompt" in out
        # The roofline check spent no restart on the config that cannot fit.
        assert "ruled out before any restart: gpu-memory-utilization=0.2 (the weights need 1.4 GiB" in out
        assert not any("gpu-memory-utilization" in flags for flags in install.starts)
        # The plan came after the first load, and the broken config was survived.
        assert "Within the budget that leaves time for 2 of the 2 configs" in out
        assert "max-num-seqs=13: did not start" in out
        # Three times the slots sustain a clearly higher rate, so they win.
        assert "Cheapest config that holds the limits: max-num-seqs=6" in out
        # And the replica is back as it was.
        assert install.engine_flags == "--max-num-seqs 2" and install.starts[-1] == "--max-num-seqs 2"
        assert install.drained == {"r1": False, "r2": False}
        assert [c for c in install.admin_calls if c.startswith("POST")] == ["POST /admin/replicas/r1/drain", "POST /admin/replicas/r1/undrain"]

        # What it left in ClickHouse.
        runs = client.query("SELECT status, replica, mode, trigger, budget_s, isNotNull(finished_at) FROM replay_runs FINAL").result_rows
        assert runs == [("finished", "r1", "recorded_shape", "cli", 600, 1)]
        rows = {r[0]: r[1:] for r in client.query(
            "SELECT config_id, outcome, held_slo, baseline, winner, rate_rps, cost_per_mtok, config FROM replay_results"
        ).result_rows}  # fmt: skip
        assert {k: v[0] for k, v in rows.items()} == {
            "current": TESTED, "max-num-seqs=6": TESTED, "max-num-seqs=13": FAILED_TO_START, "gpu-memory-utilization=0.2": RULED_OUT,
        }  # fmt: skip
        current, faster = rows["current"], rows["max-num-seqs=6"]
        assert current[1:4] == (True, True, False) and faster[1:4] == (True, False, True)
        assert faster[4] > 1.5 * current[4], "rate: three times the slots"
        assert 0 < faster[5] < current[5], "and so a lower cost per million tokens"
        assert json.loads(faster[6]) == {"max-num-seqs": 6}

        # And the report picks the run up.
        from ico.analyzer import replays

        [shown] = replays.fetch(client, ["r1", "r2"], datetime.now(timezone.utc) + timedelta(minutes=1))
        assert shown.replica == "r1" and (shown.ttft_p95_ms, shown.itl_p95_ms) == (200, 500)
        assert [c.config_id for c in shown.configs[:2]] == ["current", "max-num-seqs=6"], "the current config first, then the cheapest"
        assert shown.winner.config_id == "max-num-seqs=6" and 0.3 < shown.saving < 0.9
        assert replays.fetch(client, ["r1"], noon - timedelta(days=30)) == [], "a report for a window before the run does not show it"
    finally:
        store.connect(ch, database="default").command(f"DROP DATABASE IF EXISTS `{ch.database}`")
