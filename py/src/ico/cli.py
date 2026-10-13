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


_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


def parse_window(text: str) -> timedelta:
    match = re.fullmatch(r"(\d+)([mhd])", text.strip())
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
    from ico.analyzer import cost, measure
    from ico.collector import store

    cfg = load_config(config_path)
    now = datetime.now(timezone.utc)
    end = parse_moment(until) if until else now
    span = measure.Window(end - parse_window(window), end)
    try:
        result = cost.analyze(cfg, span, measure.fetch(store.connect(cfg.clickhouse), span), generated_at=now)
    except config_module.ConfigError as e:
        raise click.ClickException(f"{config_path}: {e}") from None
    out.write(report_module.render(result))

    # Standard error, so that `--out -` stays a clean HTML stream.
    summary = (
        f"{report_module.money(result.cost_per_mtok)} per million tokens, {report_module.percent(result.idle_share)} of GPU cost idle"
        if result.cost_per_mtok is not None
        else "no tokens served in the window"
    )
    click.echo(f"wrote {out.name}: {summary}", err=True)
