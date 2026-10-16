"""The ``ico`` command line."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

import click

from ico import config as config_module

config_option = click.option(
    "--config",
    "config_path",
    default=lambda: os.environ.get("ICO_CONFIG", "/etc/ico/config.yaml"),
    show_default="$ICO_CONFIG or /etc/ico/config.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML config file.",
)


def load_config(path: Path) -> config_module.Config:
    try:
        return config_module.load(path)
    except config_module.ConfigError as e:
        raise click.ClickException(f"{path}: {e}") from None


@click.group()
def main() -> None:
    """Self-hosted inference cost optimizer."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every scrape at INFO


@main.command()
@config_option
@click.option(
    "--schema-dir",
    default=lambda: os.environ.get("ICO_SCHEMA_DIR", "/opt/ico/schema"),
    show_default="$ICO_SCHEMA_DIR or /opt/ico/schema",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of ClickHouse migrations, applied at startup.",
)
def collector(config_path: Path, schema_dir: Path) -> None:
    """Receive telemetry and write it to ClickHouse."""
    from ico.collector import server

    cfg = load_config(config_path)
    server.migrate_when_ready(cfg, schema_dir)
    asyncio.run(server.serve(cfg))


_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def parse_window(text: str) -> timedelta:
    match = re.fullmatch(r"(\d+)([smhd])", text.strip())
    if not match or int(match.group(1)) == 0:
        raise click.BadParameter("give a length like 90m, 24h or 7d")
    return timedelta(**{_UNITS[match.group(2)]: int(match.group(1))})


def parse_moment(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise click.BadParameter("give an ISO time like 2026-10-14T09:00") from None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


@main.command()
@config_option
@click.option("--window", default="24h", show_default=True, help="How far back the report looks: 90m, 24h, 7d.")
@click.option("--until", default=None, help="Where the window ends, as an ISO time (UTC unless it says otherwise). Default: now.")
@click.option(
    "--out",
    default="report.html",
    show_default=True,
    type=click.File("w", encoding="utf-8", lazy=True),
    help="File to write. Use - for standard output.",
)
def report(config_path: Path, window: str, until: str | None, out: TextIO) -> None:
    """Write the cost and waste report as one static HTML file."""
    from ico import report as report_module
    from ico.analyzer import cost, measure, replays
    from ico.collector import store

    cfg = load_config(config_path)
    now = datetime.now(timezone.utc)
    end = parse_moment(until) if until else now
    span = measure.Window(end - parse_window(window), end)
    client = store.connect(cfg.clickhouse)
    try:
        result = cost.analyze(cfg, span, measure.fetch(client, span), generated_at=now)
    except config_module.ConfigError as e:
        raise click.ClickException(f"{config_path}: {e}") from None
    out.write(report_module.render(result, replays.fetch(client, [r.name for r in cfg.replicas], span.since, span.until)))

    # Standard error, so that `--out -` stays a clean HTML stream.
    summary = (
        f"{report_module.money(result.cost_per_mtok)} per million tokens, {report_module.percent(result.idle_share)} of GPU cost idle"
        if result.cost_per_mtok is not None
        else "no tokens served in the window"
    )
    click.echo(f"wrote {out.name}: {summary}", err=True)


def parse_slo(text: str, defaults: config_module.Slo) -> tuple[float, float]:
    """Latency limits such as ``ttft_p95=500ms,itl_p95=50ms``, in milliseconds. A missing one keeps its default."""
    limits = {"ttft_p95": defaults.ttft_p95_ms, "itl_p95": defaults.itl_p95_ms}
    for part in filter(None, (p.strip() for p in text.split(","))):
        match = re.fullmatch(r"(ttft_p95|itl_p95)=(\d+(?:\.\d+)?)(ms|s)", part)
        if not match:
            raise click.BadParameter("give limits like ttft_p95=500ms,itl_p95=50ms")
        limits[match.group(1)] = float(match.group(2)) * (1000 if match.group(3) == "s" else 1)
    return limits["ttft_p95"], limits["itl_p95"]


def parse_share(text: str) -> float:
    """A share such as ``10%`` or ``0.1``, between 0 and 1."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(%?)\s*", text)
    if not match:
        raise click.BadParameter("give a share like 10%")
    share = float(match.group(1)) / (100 if match.group(2) else 1)
    if not 0 <= share < 1:
        raise click.BadParameter("give a share from 0% up to, but not including, 100%")
    return share


@main.command()
@config_option
@click.option("--replica", "replica_name", required=True, help="The replica to drain and test.")
@click.option(
    "--configs",
    "sweep_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The sweep file listing the configs to try.",
)
@click.option("--window", default="1h", show_default=True, help="How far back the recorded traffic is taken from: 90m, 24h, 7d.")
@click.option("--until", default=None, help="Where the window ends, as an ISO time (UTC unless it says otherwise). Default: now.")
@click.option("--slo", default="", help="Latency limits a config must hold, like ttft_p95=500ms,itl_p95=50ms. Default: those in the config file.")
@click.option("--budget", default="45m", show_default=True, help="Time the whole run may take, restore included.")
@click.option(
    "--min-saving",
    default="10%",
    show_default=True,
    help="How much cheaper than the current config a config must measure to be recommended.",
)
@click.option("--single-replica", is_flag=True, help="Accept downtime: go ahead although this is the only replica taking traffic.")
@click.option("--yes", "assume_yes", is_flag=True, help="Do not wait for a yes after the plan is printed.")
@click.option("--replica-url", default=None, help="Where this machine reaches the engine, if not at the url in the config file.")
@click.option("--sample-requests", default=300, show_default=True, help="How many consecutive recorded requests the replay is built from.")
@click.option("--health-timeout", default="15m", show_default=True, help="Longest an engine may take to come up after a restart.")
@click.option("--step-seconds", default=15.0, show_default=True, help="How long the load stays at each rate of the ramp.")
@click.option("--min-step-requests", default=40, show_default=True, help="Fewest requests a rate is judged on, however low the rate.")
def replay(
    config_path: Path, replica_name: str, sweep_path: Path, window: str, until: str | None, slo: str, budget: str,
    min_saving: str, single_replica: bool, assume_yes: bool, replica_url: str | None, sample_requests: int,
    health_timeout: str, step_seconds: float, min_step_requests: int,
) -> None:  # fmt: skip
    """Test engine configs on recorded traffic and name the cheapest that holds the latency limits.

    Run it on the GPU host of the replica: the start and stop commands of its
    launcher run here. The replica is drained for the length of the run and
    restored at the end, whatever happens in between.
    """
    import statistics

    from ico.analyzer import measure
    from ico.collector import store
    from ico.replay import launchers, results, roofline, run, sweep, tester, watchdog, workload
    from ico.replay.engine import Engine, is_healthy
    from ico.replay.gateway import GatewayAdmin, GatewayError
    from ico.replay.load import Limits
    from ico.replay.prompts import Synthesizer
    from ico.replay.ramp import RampSettings

    cfg = load_config(config_path)
    replica = next((r for r in cfg.replicas if r.name == replica_name), None)
    if replica is None:
        raise click.ClickException(f"{config_path} has no replica named {replica_name}")
    if replica.gpu_model not in cfg.gpu_price_per_hour:
        raise click.ClickException(f"{replica.gpu_model!r} has no price in gpu_price_per_hour, so configs cannot be costed")
    token = os.environ.get("ICO_ADMIN_TOKEN", "")
    if not token:
        raise click.ClickException("set ICO_ADMIN_TOKEN: draining a replica goes through the admin API of the gateway")

    now = datetime.now(timezone.utc)
    end = parse_moment(until) if until else now
    span = measure.Window(end - parse_window(window), end)
    limits = Limits(*parse_slo(slo, cfg.slo))
    saving = parse_share(min_saving)
    settings = RampSettings(
        step_seconds=step_seconds,
        min_requests=min_step_requests,
        # Both scale with the step, so a short step is not followed by a long wait.
        grace_seconds=max(1.0, step_seconds * 4 / 3),
        settle_seconds=min(2.0, step_seconds / 7.5),
    )
    engine_url = (replica_url or replica.url).rstrip("/")
    say = click.echo

    # 1. The workload: a sample of the traffic in the window, as prompts of the same shape.
    client = store.connect(cfg.clickhouse)
    recorded = workload.fetch(client, replica.name, span)
    if len(recorded) < settings.min_requests:
        raise click.ClickException(
            f"only {len(recorded)} requests went through {replica.name} in that window; "
            f"a replay needs at least {settings.min_requests}. Try a longer --window."
        )
    chosen = workload.sample(recorded, sample_requests, cfg.gateway.prefix_block_chars)

    async def learn_tokenizer() -> tuple[str, list[int]]:
        async with Engine(engine_url) as engine:
            model = await engine.model()
            return model, await engine.token_pool(model)

    try:
        model_name, pool = asyncio.run(learn_tokenizer())
    except Exception as e:  # noqa: BLE001 - whatever went wrong, the engine is not usable from this machine
        raise click.ClickException(
            f"could not reach the engine of {replica.name} at {engine_url} ({e or type(e).__name__}). "
            "If this machine reaches it under another address, pass --replica-url."
        ) from None
    synth = Synthesizer(pool, chosen.requests, cfg.gateway.prefix_block_chars)
    gaps = [b.at - a.at for a, b in zip(chosen.requests, chosen.requests[1:])]
    if gaps:
        gaps.append(statistics.fmean(gaps))  # the wait before the sample starts over
    tokens_per_request = statistics.fmean(r.prompt_tokens + r.completion_tokens for r in chosen.requests)

    def median(values: tuple[int, ...]) -> int:
        return values[len(values) // 2]

    say(f"{replica.name} serves {model_name}. {len(recorded):,} requests in the window; the replay uses a run of {len(chosen.requests):,}.")
    say(
        f"  sample against window: median prompt {median(chosen.shape.prompt_tokens):,} vs {median(chosen.window_shape.prompt_tokens):,} tokens, "
        f"median completion {median(chosen.shape.completion_tokens):,} vs {median(chosen.window_shape.completion_tokens):,}, "
        f"shared prefix {chosen.shape.shared_prefix:.0%} vs {chosen.window_shape.shared_prefix:.0%} of prompt text"
    )
    say(f"  synthetic prompts repeat the start of an earlier prompt for {synth.shared_tokens():.0%} of their tokens")
    seen_ttft, streamed = workload.recorded_ttft_p95(client, replica.name, span)
    if streamed:
        say(f"  as recorded, time to first token was {seen_ttft:,.0f} ms at p95 over {streamed:,} streamed requests")
        if seen_ttft > limits.ttft_p95_ms:
            say(f"  note: that is already over the limit of {limits.ttft_p95_ms:g} ms, so expect the current config to hold at no rate")
        elif seen_ttft > 0.7 * limits.ttft_p95_ms:
            say(
                f"  note: that is close to the limit of {limits.ttft_p95_ms:g} ms. With so little room, whether a rate holds "
                "comes down to a few requests, and the rate found will vary from run to run"
            )

    # 2. The configs, less those that cannot work on paper. Each is the flags
    # running now with some changed; a launcher that can see them knows best.
    def ask(message: str) -> None:
        click.prompt(message, default="", show_default=False, prompt_suffix=" ")

    try:
        launcher = launchers.build(replica.launcher, replica.name, ask)
    except launchers.LauncherError as e:
        raise click.ClickException(str(e)) from None
    original = launcher.original if launcher.original is not None else dict(replica.launcher.flags)
    try:
        candidates = sweep.load(sweep_path, original)
    except sweep.SweepError as e:
        raise click.ClickException(str(e)) from None
    gpu_memory = roofline.gpu_memory(client, replica.host)
    ruled_out, to_test = [], []
    for c in candidates:
        reason = roofline.rule_out(c.flags, cfg.models.get(model_name), gpu_memory, replica.gpu_count)
        if reason:
            ruled_out.append(run.ConfigResult(c, run.RULED_OUT, reason))
            say(f"  ruled out before any restart: {c.id} ({reason})")
        else:
            to_test.append(c)
    say(f"{len(to_test)} configs to test within {budget}; limits: ttft p95 {limits.ttft_p95_ms:g} ms, itl p95 {limits.itl_p95_ms:g} ms")

    # 3. The run. From here on the replica is ours to put back.
    if not launcher.unattended:
        say("  note: with the manual launcher nothing can restart the engine for you if this command dies mid-run")
    budget_s = parse_window(budget).total_seconds()
    health_timeout_s = parse_window(health_timeout).total_seconds()
    dog = watchdog.spawn(
        replica=replica.name, engine_url=engine_url, admin_url=cfg.gateway.admin_url, admin_token=token,
        launcher=replica.launcher, original=original, health_timeout_s=health_timeout_s,
    )  # fmt: skip
    replay_run = run.Replay(
        settings=run.Settings(
            replica=replica.name,
            budget_s=budget_s,
            replica_price_per_hour=cfg.gpu_price_per_hour[replica.gpu_model] * replica.gpu_count,
            tokens_per_request=tokens_per_request,
            # Low enough that a step is not hours long, however quiet the window was.
            start_rps=max(chosen.shape.rate_rps, settings.min_requests / (4 * settings.step_seconds)),
            ramp_growth=settings.growth,
            single_replica=single_replica,
            assume_yes=assume_yes,
            health_timeout_s=health_timeout_s,
            min_saving=saving,
        ),
        original=original,
        admin=GatewayAdmin(cfg.gateway.admin_url, token),
        launcher=launcher,
        test=tester.make(engine_url, synth, gaps, chosen.shape.rate_rps, limits, settings, say),
        full_refinements=settings.refinements,
        healthy=lambda: is_healthy(engine_url),
        say=say,
        confirm=lambda question: click.confirm(question, default=False),
        watchdog=dog,
    )
    record = dict(
        trigger="cli", replica=replica.name, window=span, limits=limits, budget_s=budget_s, started_at=now, min_saving=saving
    )
    results.record_start(client, replay_run.run_id, **record)
    try:
        outcome = replay_run.run(to_test, ruled_out)
    except (run.ReplayError, GatewayError) as e:
        failed = run.RunResult(replay_run.run_id, run.FAILED, (), str(e), min_saving=saving)
        results.record_finish(client, failed, **record, finished_at=datetime.now(timezone.utc))
        raise click.ClickException(str(e)) from None
    results.record_finish(client, outcome, **record, finished_at=datetime.now(timezone.utc))

    say("")
    say(results.summary(outcome))
    if outcome.status != run.FINISHED:
        say(f"\nThe run ended early: {outcome.status}" + (f" ({outcome.note})" if outcome.note else "") + ".")
        raise SystemExit(130 if outcome.status == run.ABORTED else 1)
