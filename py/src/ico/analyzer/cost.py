"""The cost math of the report.

Cost is allocation-based: a replica costs its GPUs times the price per GPU-hour
for as long as it was there, whether or not it served anything. That cost is
then cut two ways:

- into active and idle, by the share of time the GPUs did work;
- across the replica's tenants and models, by their share of the work.

Everything here is plain arithmetic over what measure.fetch returned, so it
can be checked by hand.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

from ico.analyzer.measure import HostGpus, Measurements, Phases, Usage, Window
from ico.config import Config, ConfigError, Replica

# How a replica's active share was measured.
BY_PROFILING = "profiling"  # compute cores active, from DCGM
BY_REQUESTS = "requests"  # time with a request in flight

# How a replica's cost was split across its tenants.
BY_PHASE_TIME = "phase_time"  # prefill and decode time from engine spans
BY_TOKENS = "tokens"  # token count, prompt and completion weighted alike


def per_mtok(cost: float, tokens: int) -> float | None:
    """Dollars per million tokens, or None when there were no tokens."""
    return cost / tokens * 1_000_000 if tokens else None


@dataclass(frozen=True)
class ReplicaCost:
    name: str
    host: str
    gpu_model: str
    gpu_count: int
    price_per_gpu_hour: float
    # How long the replica was seen inside the window. It is billed for all of it.
    hours: float
    active_share: float
    active_basis: str
    split_basis: str
    # Share of the replica's requests the engine sent a span for; None without requests.
    span_coverage: float | None

    @property
    def gpu_hours(self) -> float:
        return self.gpu_count * self.hours

    @property
    def cost(self) -> float:
        return self.gpu_hours * self.price_per_gpu_hour

    @property
    def active_cost(self) -> float:
        return self.cost * self.active_share

    @property
    def idle_cost(self) -> float:
        return self.cost - self.active_cost

    @property
    def idle_gpu_hours(self) -> float:
        return self.gpu_hours * (1 - self.active_share)


@dataclass(frozen=True)
class ModelCost:
    model: str
    gpu_hours: float
    idle_gpu_hours: float
    cost: float
    active_cost: float
    requests: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def idle_cost(self) -> float:
        return self.cost - self.active_cost

    @property
    def active_share(self) -> float:
        return self.active_cost / self.cost if self.cost else 0.0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_per_mtok(self) -> float | None:
        return per_mtok(self.cost, self.tokens)


@dataclass(frozen=True)
class TenantCost:
    # "" is traffic sent without a tenant header. None is the cost of replicas
    # that served no tokens at all: there is nobody to charge it to.
    tenant: str | None
    model: str
    requests: int
    errors: int
    prompt_tokens: int
    completion_tokens: int
    cost: float

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_per_mtok(self) -> float | None:
        return per_mtok(self.cost, self.tokens)


@dataclass(frozen=True)
class CostReport:
    window: Window
    generated_at: datetime
    replicas: tuple[ReplicaCost, ...]
    models: tuple[ModelCost, ...]
    tenants: tuple[TenantCost, ...]
    # Hosts named in the config, in config order. None where a host sent no GPU telemetry.
    hosts: tuple[tuple[str, HostGpus | None], ...]
    gpu_price_per_hour: dict[str, float]
    notes: tuple[str, ...]

    @property
    def cost(self) -> float:
        return sum(r.cost for r in self.replicas)

    @property
    def active_cost(self) -> float:
        return sum(r.active_cost for r in self.replicas)

    @property
    def idle_cost(self) -> float:
        return self.cost - self.active_cost

    @property
    def idle_share(self) -> float:
        return self.idle_cost / self.cost if self.cost else 0.0

    @property
    def gpu_hours(self) -> float:
        return sum(r.gpu_hours for r in self.replicas)

    @property
    def idle_gpu_hours(self) -> float:
        return sum(r.idle_gpu_hours for r in self.replicas)

    @property
    def requests(self) -> int:
        return sum(m.requests for m in self.models)

    @property
    def tokens(self) -> int:
        return sum(m.tokens for m in self.models)

    @property
    def cost_per_mtok(self) -> float | None:
        return per_mtok(self.cost, self.tokens)

    @property
    def active_cost_per_mtok(self) -> float | None:
        """What the same tokens would have cost had the GPUs never been idle."""
        return per_mtok(self.active_cost, self.tokens)

    @property
    def telemetry_level(self) -> int | None:
        """The lowest level among hosts that sent GPU telemetry; None if none did."""
        levels = [h.level for _, h in self.hosts if h is not None]
        return min(levels) if levels else None


def analyze(cfg: Config, window: Window, m: Measurements, *, generated_at: datetime) -> CostReport:
    for r in cfg.replicas:
        if r.gpu_model not in cfg.gpu_price_per_hour:
            raise ConfigError(
                f"replica {r.name!r} runs on {r.gpu_model!r}, which has no price in gpu_price_per_hour"
            )

    notes: list[str] = []
    usage_of: dict[str, list[Usage]] = defaultdict(list)
    for u in m.usage:
        usage_of[u.replica].append(u)
    for name in sorted(set(usage_of) - {r.name for r in cfg.replicas}):
        count = sum(u.requests for u in usage_of[name])
        notes.append(f"{count:,} requests went to replica {name}, which is not in the config file. They carry no cost.")

    replicas_on_host = Counter(r.host for r in cfg.replicas)
    replicas: list[ReplicaCost] = []
    models: dict[str, _Sum] = defaultdict(_Sum)
    tenants: dict[tuple[str | None, str], _Sum] = defaultdict(_Sum)

    for r in cfg.replicas:
        seconds = _seconds_seen(r, window, m)
        if seconds == 0:
            notes.append(f"Replica {r.name} sent no telemetry in this window. It carries no cost.")
        alone = replicas_on_host[r.host] == 1
        host = m.hosts.get(r.host)
        if host is not None and host.sm_active is not None and not alone:
            notes.append(
                f"Host {r.host} has profiling counters but runs more than one replica, and the config does not say "
                f"which GPUs belong to {r.name}. Its active share is taken from request timing instead."
            )
        active_share, active_basis = _active_share(r, seconds, m, alone)
        prompt_weight, completion_weight, split_basis = _token_weights(m.phases.get(r.name))
        phases = m.phases.get(r.name)
        replica = ReplicaCost(
            name=r.name,
            host=r.host,
            gpu_model=r.gpu_model,
            gpu_count=r.gpu_count,
            price_per_gpu_hour=cfg.gpu_price_per_hour[r.gpu_model],
            hours=seconds / 3600,
            active_share=active_share,
            active_basis=active_basis,
            split_basis=split_basis,
            span_coverage=phases.with_span / phases.requests if phases and phases.requests else None,
        )
        replicas.append(replica)

        rows = usage_of.get(r.name, [])
        weights = [prompt_weight * u.prompt_tokens + completion_weight * u.completion_tokens for u in rows]
        total = sum(weights)
        for u, weight in zip(rows, weights):
            share = weight / total if total else 0.0
            models[u.model].add(replica, share, u)
            tenants[(u.tenant, u.model)].add(replica, share, u)
        if not total and replica.cost:
            # Nothing was served, so no tenant can carry the cost. It still
            # belongs to the model the replica was there to serve.
            engine = m.engines.get(r.name)
            model = (engine.served_model if engine else "") or (rows[0].model if rows else "")
            models[model].add(replica, 1.0)
            tenants[(None, model)].add(replica, 1.0)

    return CostReport(
        window=window,
        generated_at=generated_at,
        replicas=tuple(replicas),
        models=tuple(
            sorted(
                (
                    ModelCost(model, s.gpu_hours, s.idle_gpu_hours, s.cost, s.active_cost, s.requests, s.prompt_tokens, s.completion_tokens)
                    for model, s in models.items()
                ),
                key=lambda c: -c.cost,
            )
        ),
        tenants=tuple(
            sorted(
                (
                    TenantCost(tenant, model, s.requests, s.errors, s.prompt_tokens, s.completion_tokens, s.cost)
                    for (tenant, model), s in tenants.items()
                ),
                key=lambda c: -c.cost,
            )
        ),
        hosts=tuple((host, m.hosts.get(host)) for host in dict.fromkeys(r.host for r in cfg.replicas)),
        gpu_price_per_hour={r.gpu_model: cfg.gpu_price_per_hour[r.gpu_model] for r in cfg.replicas},
        notes=tuple(notes),
    )


class _Sum:
    """Running totals for one model, or one tenant on one model."""

    def __init__(self) -> None:
        self.gpu_hours = self.idle_gpu_hours = self.cost = self.active_cost = 0.0
        self.requests = self.errors = self.prompt_tokens = self.completion_tokens = 0

    def add(self, replica: ReplicaCost, share: float, usage: Usage | None = None) -> None:
        self.gpu_hours += replica.gpu_hours * share
        self.idle_gpu_hours += replica.idle_gpu_hours * share
        self.cost += replica.cost * share
        self.active_cost += replica.active_cost * share
        if usage is not None:
            self.requests += usage.requests
            self.errors += usage.errors
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens


def _seconds_seen(r: Replica, window: Window, m: Measurements) -> float:
    """How long the replica was there: from the first to the last sign of it.

    Its requests, its engine's metrics and its host's GPU exporter all count as
    signs. Gaps in between are billed: an engine that was down for an hour
    still had its GPUs rented.
    """
    sources = (m.traffic.get(r.name), m.engines.get(r.name), m.hosts.get(r.host))
    seen = [s.seen for s in sources if s is not None]
    if not seen:
        return 0.0
    first = max(min(s.first for s in seen), window.since)
    last = min(max(s.last for s in seen), window.until)
    return max(0.0, (last - first).total_seconds())


def _active_share(r: Replica, seconds: float, m: Measurements, alone_on_host: bool) -> tuple[float, str]:
    host = m.hosts.get(r.host)
    # The exporter reports per host. With one replica on the host its cards
    # are that replica's cards; with several the config cannot tell them apart.
    if host is not None and host.sm_active is not None and alone_on_host:
        return min(max(host.sm_active, 0.0), 1.0), BY_PROFILING
    traffic = m.traffic.get(r.name)
    if traffic is None or seconds <= 0:
        return 0.0, BY_REQUESTS
    return min(traffic.busy_seconds / seconds, 1.0), BY_REQUESTS


def _token_weights(phases: Phases | None) -> tuple[float, float, str]:
    """What a prompt token and a completion token weigh when a replica is split.

    With engine spans the weights are milliseconds of engine time per token,
    so a tenant pays for the engine time its tokens took. Without them every
    token weighs the same.
    """
    if phases and phases.prefill_ms > 0 and phases.decode_ms > 0 and phases.prompt_tokens and phases.completion_tokens:
        return phases.prefill_ms / phases.prompt_tokens, phases.decode_ms / phases.completion_tokens, BY_PHASE_TIME
    return 1.0, 1.0, BY_TOKENS
