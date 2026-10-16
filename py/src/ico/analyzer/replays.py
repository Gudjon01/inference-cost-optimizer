"""The replay results a report shows: the latest finished run of each replica."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from ico.analyzer.measure import stamp
from ico.collector.store import Client


@dataclass(frozen=True)
class ReplayConfig:
    config_id: str
    stage: int
    outcome: str
    note: str
    held: bool
    rate_rps: float
    tokens_s: float
    cost_per_mtok: float
    is_baseline: bool
    is_winner: bool
    quality_unverified: bool


@dataclass(frozen=True)
class ReplayRun:
    replica: str
    finished_at: datetime
    # The stretch of recorded traffic the replay was built from.
    traffic_since: datetime
    traffic_until: datetime
    ttft_p95_ms: float
    itl_p95_ms: float
    # The current config first, then measured configs from cheapest up, then the rest.
    configs: tuple[ReplayConfig, ...]
    # How much cheaper than the current config a config had to measure to win.
    min_saving: float = 0.0

    @property
    def current(self) -> ReplayConfig | None:
        return next((c for c in self.configs if c.is_baseline), None)

    @property
    def winner(self) -> ReplayConfig | None:
        return next((c for c in self.configs if c.is_winner), None)

    @property
    def saving(self) -> float | None:
        """How far the winner's cost is below the current config's, as a share; None if there is nothing to compare."""
        current, winner = self.current, self.winner
        if winner is None or current is None or winner.is_baseline or not current.held or not current.cost_per_mtok:
            return None
        return 1 - winner.cost_per_mtok / current.cost_per_mtok

    @property
    def close_calls(self) -> tuple[ReplayConfig, ...]:
        """Configs that measured cheaper than the current one, but by less than the run's margin."""
        current = self.current
        if current is None or not current.is_winner or not current.cost_per_mtok:
            return ()
        return tuple(
            c
            for c in self.configs
            if not c.is_baseline and c.held and c.stage == 3 and not c.quality_unverified
            and c.cost_per_mtok < current.cost_per_mtok
        )  # fmt: skip

    def below_current(self, config: ReplayConfig) -> float:
        """How far a config's cost is below the current config's, as a share."""
        return 1 - config.cost_per_mtok / self.current.cost_per_mtok


def fetch(client: Client, replicas: Sequence[str], since: datetime, until: datetime) -> list[ReplayRun]:
    """For each replica, the replay a report of since..until shows.

    A replay is built from recorded traffic, so it always finishes after that
    traffic: the newest finished replay of traffic inside the window is shown,
    whenever it finished. Failing that, the newest replay that finished by
    ``until``, as the best knowledge at the window's end. A report never shows
    a replay of traffic later than its window.
    """
    runs = client.query(
        "WITH window_start < {until:DateTime64(3, 'UTC')} AND window_end > {since:DateTime64(3, 'UTC')} AS of_window "
        "SELECT run_id, replica, finished_at, window_start, window_end, slo, min_saving FROM replay_runs FINAL "
        "WHERE status = 'finished' AND replica IN {replicas:Array(String)} "
        "AND (of_window OR finished_at <= {until:DateTime64(3, 'UTC')}) "
        "ORDER BY of_window DESC, finished_at DESC LIMIT 1 BY replica",
        parameters={"replicas": list(replicas), "since": stamp(since), "until": stamp(until)},
    ).result_rows
    if not runs:
        return []
    rows = client.query(
        "SELECT run_id, config_id, stage, outcome, note, held_slo, rate_rps, tokens_s, cost_per_mtok, baseline, winner, quality_unverified "
        "FROM replay_results WHERE run_id IN {runs:Array(UUID)}",
        parameters={"runs": [str(r[0]) for r in runs]},
    ).result_rows
    # A config tested at several stages of successive halving is shown at its furthest one.
    furthest: dict[tuple[object, str], ReplayConfig] = {}
    for run_id, *fields in rows:
        config = ReplayConfig(*fields)
        known = furthest.get((run_id, config.config_id))
        if known is None or (config.outcome == "tested", config.stage) > (known.outcome == "tested", known.stage):
            furthest[(run_id, config.config_id)] = config
    configs: dict[object, list[ReplayConfig]] = {}
    for (run_id, _), config in furthest.items():
        configs.setdefault(run_id, []).append(config)

    def utc(t: datetime) -> datetime:
        return t.replace(tzinfo=timezone.utc)

    found = []
    for run_id, replica, finished_at, since, until_, slo, min_saving in runs:
        limits = json.loads(slo)
        ordered = sorted(configs.get(run_id, []), key=lambda c: (not c.is_baseline, not c.held, c.stage != 3, c.cost_per_mtok))
        found.append(
            ReplayRun(
                replica, utc(finished_at), utc(since), utc(until_), limits["ttft_p95_ms"], limits["itl_p95_ms"],
                tuple(ordered), float(min_saving),
            )
        )  # fmt: skip
    return sorted(found, key=lambda r: r.replica)
