"""The cost math, on numbers small enough to check by hand."""

from datetime import datetime, timedelta, timezone

import pytest

from ico import config
from ico.analyzer.cost import BY_PHASE_TIME, BY_PROFILING, BY_REQUESTS, BY_TOKENS, analyze
from ico.analyzer.measure import Engine, HostGpus, Measurements, Phases, Seen, Traffic, Usage, Window

T0 = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)
DAY = Window(T0, T0 + timedelta(hours=24))
WHOLE_DAY = Seen(T0, T0 + timedelta(hours=24))


def cfg(*replicas: dict, prices: dict | None = None) -> config.Config:
    return config.parse({"replicas": list(replicas), "gpu_price_per_hour": prices or {"H100": 2.0}})


def replica(name: str = "r1", host: str = "gpu-1", gpus: int = 2) -> dict:
    return {"name": name, "url": f"http://{name}:8000", "host": host, "gpu_model": "H100", "gpu_count": gpus}


def measurements(**parts) -> Measurements:
    return Measurements(
        usage=tuple(parts.get("usage", ())),
        traffic=parts.get("traffic", {}),
        phases=parts.get("phases", {}),
        engines=parts.get("engines", {}),
        hosts=parts.get("hosts", {}),
    )


def host(name: str = "gpu-1", level: int = 2, sm_active: float | None = None, seen: Seen = WHOLE_DAY) -> HostGpus:
    return HostGpus(name, 2, ("NVIDIA H100",), level, seen, sm_active, None, None, 0.9, 40e9, 300.0)


def run(c: config.Config, m: Measurements, window: Window = DAY):
    return analyze(c, window, m, generated_at=T0)


def test_allocation_cost_bills_idle_time():
    # 2 GPUs at $2 for 24 hours = $96. Busy 6 of 24 hours: a quarter of it is active.
    report = run(
        cfg(replica()),
        measurements(
            usage=[Usage("r1", "m", "acme", 10, 0, 3_000_000, 1_000_000)],
            traffic={"r1": Traffic(6 * 3600, WHOLE_DAY)},
        ),
    )
    [r] = report.replicas
    assert (r.gpu_hours, r.cost) == (48, 96)
    assert (r.active_share, r.active_basis) == (0.25, BY_REQUESTS)
    assert (r.active_cost, r.idle_cost, r.idle_gpu_hours) == (24, 72, 36)

    assert (report.cost, report.idle_cost, report.idle_share) == (96, 72, 0.75)
    assert report.tokens == 4_000_000
    assert report.cost_per_mtok == 24  # $96 over 4M tokens
    assert report.active_cost_per_mtok == 6  # what it would be with no idle time


def test_a_replica_is_billed_only_for_the_time_it_was_seen():
    # A 24-hour window, but the install is 3 hours old: 2 GPUs x $2 x 3 h = $12.
    seen = Seen(T0 + timedelta(hours=21), T0 + timedelta(hours=24))
    report = run(cfg(replica()), measurements(engines={"r1": Engine(seen, "m")}))
    assert report.replicas[0].hours == 3
    assert report.cost == 12


def test_gaps_between_signs_of_life_are_billed():
    # Requests in the first hour, GPU exporter heard in the last: the 22 hours
    # between are billed too. The GPUs were rented the whole time.
    report = run(
        cfg(replica()),
        measurements(
            traffic={"r1": Traffic(600, Seen(T0, T0 + timedelta(hours=1)))},
            hosts={"gpu-1": host(seen=Seen(T0 + timedelta(hours=23), T0 + timedelta(hours=24)))},
        ),
    )
    assert report.replicas[0].hours == 24


def test_profiling_counters_decide_the_active_share_when_present():
    usage = [Usage("r1", "m", "acme", 1, 0, 100, 100)]
    traffic = {"r1": Traffic(12 * 3600, WHOLE_DAY)}  # request timing would say 50%

    with_profiling = run(cfg(replica()), measurements(usage=usage, traffic=traffic, hosts={"gpu-1": host(level=3, sm_active=0.2)}))
    assert (with_profiling.replicas[0].active_share, with_profiling.replicas[0].active_basis) == (0.2, BY_PROFILING)
    assert with_profiling.telemetry_level == 3

    without = run(cfg(replica()), measurements(usage=usage, traffic=traffic, hosts={"gpu-1": host(level=2)}))
    assert (without.replicas[0].active_share, without.replicas[0].active_basis) == (0.5, BY_REQUESTS)
    assert without.telemetry_level == 2


def test_two_replicas_on_one_host_fall_back_to_request_timing():
    # The exporter reports per host; nothing says which cards are whose.
    report = run(
        cfg(replica("r1"), replica("r2")),
        measurements(
            traffic={"r1": Traffic(6 * 3600, WHOLE_DAY), "r2": Traffic(18 * 3600, WHOLE_DAY)},
            hosts={"gpu-1": host(level=3, sm_active=0.9)},
        ),
    )
    assert [(r.active_share, r.active_basis) for r in report.replicas] == [(0.25, BY_REQUESTS), (0.75, BY_REQUESTS)]
    assert sum("more than one replica" in note for note in report.notes) == 2


def test_tenants_split_a_replica_by_token_count_without_spans():
    # $96 replica. acme has 3M of the 4M tokens, globex 1M.
    report = run(
        cfg(replica()),
        measurements(
            usage=[
                Usage("r1", "m", "acme", 30, 2, 2_000_000, 1_000_000),
                Usage("r1", "m", "globex", 10, 0, 900_000, 100_000),
            ],
            traffic={"r1": Traffic(3600, WHOLE_DAY)},
        ),
    )
    assert report.replicas[0].split_basis == BY_TOKENS
    acme, globex = report.tenants
    assert (acme.tenant, acme.cost, acme.requests, acme.errors) == ("acme", 72, 30, 2)
    assert (globex.tenant, globex.cost) == ("globex", 24)
    assert acme.cost_per_mtok == globex.cost_per_mtok == 24, "by token count, every token costs the same"
    assert sum(t.cost for t in report.tenants) == report.cost


def test_tenants_split_by_engine_time_when_spans_report_phases():
    # Spans say a prompt token took 1 ms and a completion token 9 ms.
    # acme: 1M prompt + 1M completion = 10M ms. globex: 2M prompt = 2M ms.
    # acme carries 10/12 of the $96, globex 2/12.
    report = run(
        cfg(replica()),
        measurements(
            usage=[
                Usage("r1", "m", "acme", 5, 0, 1_000_000, 1_000_000),
                Usage("r1", "m", "globex", 5, 0, 2_000_000, 0),
            ],
            traffic={"r1": Traffic(3600, WHOLE_DAY)},
            phases={"r1": Phases(requests=10, with_span=8, prefill_ms=3_000, decode_ms=9_000, prompt_tokens=3_000, completion_tokens=1_000)},
        ),
    )
    assert report.replicas[0].split_basis == BY_PHASE_TIME
    assert report.replicas[0].span_coverage == 0.8
    acme, globex = report.tenants
    assert (acme.cost, globex.cost) == (pytest.approx(80), pytest.approx(16))
    # globex sent more tokens and pays less: its tokens were all cheap prompt tokens.
    assert globex.tokens > acme.tokens / 2 and globex.cost_per_mtok < acme.cost_per_mtok


def test_models_carry_their_replicas_including_idle_ones():
    report = run(
        cfg(replica("busy"), replica("spare")),
        measurements(
            usage=[Usage("busy", "big-model", "acme", 4, 0, 500_000, 500_000)],
            traffic={"busy": Traffic(12 * 3600, WHOLE_DAY)},
            engines={"busy": Engine(WHOLE_DAY, "big-model"), "spare": Engine(WHOLE_DAY, "small-model")},
        ),
    )
    by_model = {m.model: m for m in report.models}
    assert (by_model["big-model"].cost, by_model["big-model"].active_share) == (96, 0.5)
    # The spare replica served nothing: all idle, charged to the model it was there for.
    spare = by_model["small-model"]
    assert (spare.cost, spare.active_cost, spare.idle_gpu_hours, spare.tokens) == (96, 0, 48, 0)
    assert spare.cost_per_mtok is None
    # No tenant used it, so its cost shows up unassigned and the table still adds up.
    unassigned = next(t for t in report.tenants if t.tenant is None)
    assert (unassigned.model, unassigned.cost) == ("small-model", 96)
    assert sum(t.cost for t in report.tenants) == report.cost == 192


def test_one_model_on_two_replicas_adds_up():
    report = run(
        cfg(replica("r1", "gpu-1"), replica("r2", "gpu-2")),
        measurements(
            usage=[Usage("r1", "m", "acme", 1, 0, 1_000_000, 0), Usage("r2", "m", "acme", 1, 0, 1_000_000, 0)],
            traffic={"r1": Traffic(24 * 3600, WHOLE_DAY), "r2": Traffic(12 * 3600, WHOLE_DAY)},
        ),
    )
    [m] = report.models
    assert (m.cost, m.active_cost, m.gpu_hours, m.idle_gpu_hours) == (192, 144, 96, 24)
    [t] = report.tenants
    assert (t.cost, t.cost_per_mtok) == (192, 96)


def test_untagged_traffic_stays_visible():
    report = run(
        cfg(replica()),
        measurements(usage=[Usage("r1", "m", "", 3, 0, 500, 500)], traffic={"r1": Traffic(60, WHOLE_DAY)}),
    )
    assert [(t.tenant, t.cost) for t in report.tenants] == [("", 96)]


def test_requests_to_an_unconfigured_replica_are_noted_not_costed():
    report = run(
        cfg(replica()),
        measurements(
            usage=[Usage("r1", "m", "acme", 1, 0, 10, 10), Usage("old-replica", "m", "acme", 1234, 0, 10, 10)],
            traffic={"r1": Traffic(60, WHOLE_DAY)},
        ),
    )
    assert report.requests == 1
    assert any("1,234 requests went to replica old-replica" in note for note in report.notes)


def test_silent_replica_is_noted_and_costs_nothing():
    report = run(cfg(replica()), measurements())
    assert report.cost == 0 and report.cost_per_mtok is None
    assert any("r1 sent no telemetry" in note for note in report.notes)


def test_a_missing_price_is_an_error_not_a_zero():
    with pytest.raises(config.ConfigError, match="'H100', which has no price"):
        run(cfg(replica(), prices={"L4": 0.5}), measurements())
