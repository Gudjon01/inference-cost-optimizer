"""What a run leaves behind: rows in ClickHouse and a summary for the terminal."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime

from ico.analyzer.measure import Window
from ico.collector.store import Client
from ico.replay.load import Limits
from ico.replay.run import FAILED_TO_START, RULED_OUT, SKIPPED, RunResult, ConfigResult

RUNS_COLUMNS = (
    "run_id", "trigger", "mode", "replica", "window_start", "window_end",
    "slo", "budget_s", "started_at", "finished_at", "status", "note",
)  # fmt: skip
RESULTS_COLUMNS = (
    "run_id", "config_id", "stage", "config", "rate_rps", "tokens_s", "ttft_p95_ms", "itl_p95_ms",
    "error_rate", "cost_per_mtok", "held_slo", "outcome", "note", "quality_unverified", "baseline", "winner",
)  # fmt: skip

# The only mode so far: synthetic prompts in the recorded shape. Live mirror comes later.
MODE = "recorded_shape"


def record_start(
    client: Client, run_id: uuid.UUID, *, trigger: str, replica: str, window: Window, limits: Limits, budget_s: float, started_at: datetime
) -> None:
    client.insert("replay_runs", [_run_row(run_id, trigger, replica, window, limits, budget_s, started_at, None, "running", "")], column_names=RUNS_COLUMNS)


def record_finish(
    client: Client, result: RunResult, *, trigger: str, replica: str, window: Window, limits: Limits, budget_s: float, started_at: datetime, finished_at: datetime
) -> None:
    # replay_runs keeps the last row written per run, so the finished row replaces the running one.
    client.insert(
        "replay_runs",
        [_run_row(result.run_id, trigger, replica, window, limits, budget_s, started_at, finished_at, result.status, result.note)],
        column_names=RUNS_COLUMNS,
    )
    if result.results:
        winner = result.winner
        client.insert("replay_results", [_result_row(result.run_id, t, t is winner) for t in result.results], column_names=RESULTS_COLUMNS)


def _run_row(run_id, trigger, replica, window, limits, budget_s, started_at, finished_at, status, note):
    slo = json.dumps({"ttft_p95_ms": limits.ttft_p95_ms, "itl_p95_ms": limits.itl_p95_ms})
    return (run_id, trigger, MODE, replica, window.since, window.until, slo, int(budget_s), started_at, finished_at, status, note)


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _result_row(run_id: uuid.UUID, t: ConfigResult, is_winner: bool):
    best = t.best
    return (
        run_id,
        t.candidate.id,
        t.stage,
        json.dumps(t.candidate.flags, sort_keys=True),
        best.rate_rps if best else 0.0,
        t.tokens_s,
        _finite(best.ttft_p95_ms) if best else 0.0,
        _finite(best.itl_p95_ms) if best else 0.0,
        best.error_rate if best else 0.0,
        t.cost_per_mtok or 0.0,
        best is not None,
        t.outcome,
        t.note,
        t.candidate.quality_unverified,
        t.is_baseline,
        is_winner,
    )


def summary(result: RunResult) -> str:
    """The table and the verdict printed at the end of a run."""
    from ico.replay.ramp import FULL, STAGES

    winner, current = result.winner, result.baseline
    shown = result.final
    width = max([len("config")] + [len(t.candidate.id) for t in shown])
    lines = [f"{'config':<{width}}  requests/s   tokens/s   per million tokens"]
    # The current config first, then what was measured from cheapest up, then what was not.
    for t in sorted(shown, key=lambda t: (not t.is_baseline, t.best is None, t.stage != FULL.number, t.cost_per_mtok or 0.0)):
        if t.best is None:
            why = {RULED_OUT: "ruled out", FAILED_TO_START: "did not start", SKIPPED: "not tested"}.get(t.outcome)
            lines.append(f"{t.candidate.id:<{width}}  {why}: {t.note}" if why else f"{t.candidate.id:<{width}}  {t.note or 'no rate held'}")
            continue
        marks = [
            m
            for m, on in (
                ("cheapest", t is winner),
                ("quality unverified", t.candidate.quality_unverified),
                (f"out after a {STAGES[t.stage].name}", t.stage != FULL.number),
            )
            if on
        ]
        row = f"{t.candidate.id:<{width}}  {t.best.rate_rps:10.2f}  {t.tokens_s:9,.0f}  ${t.cost_per_mtok:<10.4f}"
        lines.append((row + "  " + ", ".join(marks)).rstrip())

    lines.append("")
    if winner is None:
        lines.append("No config held the latency limits at any rate tried.")
    elif winner.is_baseline:
        lines.append(f"The current config is the cheapest of those tested: ${winner.cost_per_mtok:.4f} per million tokens at {winner.best.rate_rps:.2f} requests/s.")
    else:
        line = f"Cheapest config that holds the limits: {winner.candidate.id}, ${winner.cost_per_mtok:.4f} per million tokens at {winner.best.rate_rps:.2f} requests/s"
        if current is not None and current.cost_per_mtok:
            saved = 1 - winner.cost_per_mtok / current.cost_per_mtok
            line += f", {saved:.0%} below the current config (${current.cost_per_mtok:.4f})"
        lines.append(line + ".")
        if current is not None and current.best is None:
            lines.append("The current config held the limits at no rate tried, so there is no current cost to compare with.")
    if any(t.candidate.quality_unverified and t.best for t in result.results):
        lines.append("Configs that change precision are measured but cannot win: their answer quality is not checked yet.")
    return "\n".join(lines)
