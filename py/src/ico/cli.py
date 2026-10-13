"""The ``ico`` command line."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import click

from ico import config as config_module


@click.group()
def main() -> None:
    """Self-hosted inference cost optimizer."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every scrape at INFO


@main.command()
@click.option(
    "--config",
    "config_path",
    default=lambda: os.environ.get("ICO_CONFIG", "/etc/ico/config.yaml"),
    show_default="$ICO_CONFIG or /etc/ico/config.yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML config file.",
)
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

    try:
        cfg = config_module.load(config_path)
    except config_module.ConfigError as e:
        raise click.ClickException(f"{config_path}: {e}") from None
    server.migrate_when_ready(cfg, schema_dir)
    asyncio.run(server.serve(cfg))
