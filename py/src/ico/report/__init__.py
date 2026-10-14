"""Renders a cost report as one static HTML file: no scripts, nothing loaded from elsewhere."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ico.analyzer.cost import BY_PHASE_TIME, BY_PROFILING, BY_REQUESTS, BY_TOKENS, CostReport, TenantCost
from ico.analyzer.replays import ReplayRun

# Past this many rows the smallest tenants are folded into one line.
MAX_TENANT_ROWS = 100

ACTIVE_BASIS = {
    BY_PROFILING: "Compute cores active (DCGM profiling)",
    BY_REQUESTS: "Time with a request in flight",
}
SPLIT_BASIS = {
    BY_PHASE_TIME: "Prefill and decode time",
    BY_TOKENS: "Token count",
}
LEVELS = {
    3: ("L3", "profiling counters: compute cores, tensor cores and memory bandwidth"),
    2: ("L2", "basic counters: busy time, memory and power"),
    1: ("L1", "memory and power only"),
}


def money(x: float | None) -> str:
    """Dollars at a precision that fits the size: $1,234, $12.34, $0.012."""
    if x is None:
        return "n/a"
    size = abs(x)
    if size >= 1000:
        return f"${x:,.0f}"
    if size >= 1 or size == 0:
        return f"${x:,.2f}"
    # Below a dollar, keep two significant digits and never fewer than cents.
    return f"${x:.{max(2, 1 - math.floor(math.log10(size)))}f}"


def count(n: float) -> str:
    return f"{n:,.0f}"


def compact(n: float) -> str:
    """A count short enough for a tile: 1,284, 12.9K, 4.2M."""
    # Thousands stay written out until five digits: 1,284 reads better than 1.3K.
    for threshold, divisor, unit in ((1e9, 1e9, "B"), (1e6, 1e6, "M"), (1e4, 1e3, "K")):
        if abs(n) >= threshold:
            return f"{n / divisor:.1f}{unit}"
    return f"{n:,.0f}"


def percent(x: float) -> str:
    if 0 < x < 0.01:
        return "<1%"
    if 0.99 < x < 1:
        return ">99%"
    return f"{x * 100:.0f}%"


def hours(h: float) -> str:
    return f"{h:,.1f}" if h >= 10 else f"{h:,.2f}"


def span(seconds: float) -> str:
    """A length of time in words: 45 s, 12 min, 3 h 20 min, 2 d 4 h."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours_, minutes = divmod(minutes, 60)
    if hours_ < 24:
        return f"{hours_} h {minutes} min" if minutes else f"{hours_} h"
    days, hours_ = divmod(hours_, 24)
    return f"{days} d {hours_} h" if hours_ else f"{days} d"


def moment(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def gibibytes(b: float) -> str:
    return f"{b / 2**30:.1f} GiB"


_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
_env.filters.update(
    money=money, count=count, compact=compact, percent=percent, hours=hours, span=span, moment=moment, gibibytes=gibibytes
)


# Why a replayed config has no numbers, by its recorded outcome.
NOT_MEASURED = {"ruled_out": "Ruled out before any restart", "failed_to_start": "Did not start", "skipped": "Not tested"}


def render(report: CostReport, replays: Sequence[ReplayRun] = ()) -> str:
    shown, folded = report.tenants[:MAX_TENANT_ROWS], report.tenants[MAX_TENANT_ROWS:]
    return _env.get_template("report.html.j2").render(
        report=report,
        replays=replays,
        not_measured=NOT_MEASURED,
        tenants=shown,
        folded=_fold(folded),
        folded_count=len(folded),
        active_basis=ACTIVE_BASIS,
        split_basis=SPLIT_BASIS,
        levels=LEVELS,
    )


def _fold(rows: tuple[TenantCost, ...]) -> TenantCost | None:
    """The tail of the tenant list as one row, so the table still adds up."""
    if not rows:
        return None
    return TenantCost(
        tenant=None,
        model="",
        requests=sum(r.requests for r in rows),
        errors=sum(r.errors for r in rows),
        prompt_tokens=sum(r.prompt_tokens for r in rows),
        completion_tokens=sum(r.completion_tokens for r in rows),
        cost=sum(r.cost for r in rows),
    )
