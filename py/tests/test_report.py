import re
from datetime import timedelta

import pytest

from ico import report
from ico.analyzer.measure import Phases, Seen, Traffic, Usage
from tests.test_cost import T0, WHOLE_DAY, cfg, host, measurements, replica, run


def sample_report(**overrides):
    parts = dict(
        usage=[
            Usage("r1", "llama-70b", "acme", 30, 2, 2_000_000, 1_000_000),
            Usage("r1", "llama-70b", "globex", 10, 0, 900_000, 100_000),
        ],
        traffic={"r1": Traffic(6 * 3600, WHOLE_DAY)},
        hosts={"gpu-1": host(level=2)},
    )
    parts.update(overrides)
    return run(cfg(replica()), measurements(**parts))


def text_of(html: str) -> str:
    """The page as a reader sees it: tags dropped, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"(?s)<style>.*?</style>", "", html)))


@pytest.mark.parametrize(
    ("value", "shown"),
    [(None, "n/a"), (0, "$0.00"), (1234.5, "$1,234"), (96, "$96.00"), (12.345, "$12.35"), (0.5, "$0.50"), (0.0123, "$0.012"), (0.00456, "$0.0046")],
)
def test_money_keeps_a_precision_that_fits_the_size(value, shown):
    assert report.money(value) == shown


def test_small_formatters():
    assert [report.compact(n) for n in (1284, 12_900, 4_200_000)] == ["1,284", "12.9K", "4.2M"]
    assert [report.percent(x) for x in (0, 0.004, 0.25, 0.995, 1)] == ["0%", "<1%", "25%", ">99%", "100%"]
    assert [report.span(s) for s in (45, 720, 12_000, 86_400, 187_200)] == ["45 s", "12 min", "3 h 20 min", "1 d", "2 d 4 h"]
    assert report.moment(T0 + timedelta(hours=9, minutes=5)) == "2026-10-14 09:05"


def test_report_leads_with_cost_per_million_tokens_and_the_idle_share():
    page = text_of(report.render(sample_report()))
    # $96 for 4M tokens; 18 of 24 hours idle.
    assert "$24.00 per million tokens" in page
    assert "GPUs cost $96.00 in this window, and $72.00 of it (75%) paid for time with nothing to serve." in page
    assert "the same tokens would have cost $6.00 per million" in page
    assert "Idle GPU-hours 36.0 of 48.0" in page


def test_report_shows_every_part_the_spec_lists():
    page = text_of(report.render(sample_report()))
    # Idle vs active cost per model, and cost per million tokens.
    assert "llama-70b 48.0 $96.00 25% $72.00 36.0 2,900,000 1,100,000 $24.00" in page
    # The per-tenant split.
    assert "acme llama-70b 30 2 2,000,000 1,000,000 75% $72.00 $24.00" in page
    assert "globex llama-70b 10 0 900,000 100,000 25% $24.00 $24.00" in page
    # The telemetry level.
    assert "L2 , basic counters: busy time, memory and power" in page
    # The inputs used.
    assert "2026-10-14 00:00 to 2026-10-15 00:00 UTC (1 d)" in page
    assert "H100 at $2.00 per GPU-hour, from the config file" in page
    assert "Time with a request in flight" in page and "Token count" in page


def test_report_names_how_each_number_was_measured():
    with_spans = sample_report(
        hosts={"gpu-1": host(level=3, sm_active=0.4)},
        phases={"r1": Phases(40, 30, 3_000, 9_000, 3_000, 1_000)},
    )
    page = text_of(report.render(with_spans))
    assert "Compute cores active (DCGM profiling)" in page
    assert "Prefill and decode time" in page
    assert "L3 , profiling counters" in page
    assert "75%" in page  # 30 of 40 requests traced


def test_report_is_one_self_contained_file():
    html = report.render(sample_report())
    assert html.startswith("<!doctype html>")
    # Nothing is fetched and nothing runs: it must open the same on an air-gapped box.
    for forbidden in ("<script", "http://", "https://", "@import", "url(", "<link", "<img", "<iframe"):
        assert forbidden not in html, f"report contains {forbidden!r}"


def test_tenant_names_are_escaped():
    hostile = sample_report(usage=[Usage("r1", "m", "<script>alert(1)</script>", 1, 0, 100, 100)])
    html = report.render(hostile)
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_untagged_and_unassigned_cost_are_labelled():
    page = text_of(report.render(sample_report(usage=[Usage("r1", "m", "", 3, 0, 500, 500)])))
    assert "(no tenant header) m 3" in page
    idle_only = run(cfg(replica()), measurements(traffic={}, hosts={"gpu-1": host()}))
    page = text_of(report.render(idle_only))
    assert "$96.00 spent, no tokens served" in page
    assert "(no traffic to charge)" in page


def test_empty_window_says_so_instead_of_showing_zeros():
    page = text_of(report.render(run(cfg(replica()), measurements())))
    assert "No telemetry was recorded in this window" in page
    assert "per million tokens" not in page
    assert "Replica r1 sent no telemetry in this window." in page
    assert "No GPU telemetry in this window." in page


def test_long_tenant_lists_fold_into_one_row_that_keeps_the_total():
    many = [Usage("r1", "m", f"tenant-{i:03d}", 1, 0, 1000 + i, 0) for i in range(report.MAX_TENANT_ROWS + 25)]
    result = run(cfg(replica()), measurements(usage=many, traffic={"r1": Traffic(3600, Seen(T0, T0 + timedelta(hours=24)))}))
    html = report.render(result)
    page = text_of(html)
    assert "25 smaller tenant rows" in page
    assert html.count("<tr>") < report.MAX_TENANT_ROWS + 40
    assert "tenant-124" in page and "tenant-000" not in page, "the largest tenants are the ones kept"
