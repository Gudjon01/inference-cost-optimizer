"""Puts a replica back if the replay lab dies while it holds one.

The replay lab starts this as a separate process before it drains anything
and keeps a pipe to it open. Down the pipe it reports how far it has got:
``drained``, ``stopped``, ``restored`` and finally ``done``. If the pipe
closes without ``done`` the replay lab is gone, whatever killed it, and the
watchdog finishes the job from the last step reported: restart the engine
with its original flags if it was stopped, then let traffic back in.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import IO

from ico.config import LauncherConfig
from ico.replay.engine import is_healthy
from ico.replay.gateway import GatewayAdmin, GatewayError
from ico.replay import launchers
from ico.replay.launchers import LauncherError
from ico.replay.sweep import Flags

DRAINED, STOPPED, RESTORED, DONE = "drained", "stopped", "restored", "done"

log = logging.getLogger("ico.watchdog")


class Watchdog:
    """The replay lab's end of the pipe."""

    def __init__(self, process: subprocess.Popen, log_path: Path) -> None:
        self._process = process
        self.log_path = log_path

    def report(self, state: str) -> None:
        try:
            self._process.stdin.write(state + "\n")
            self._process.stdin.flush()
        except (OSError, ValueError):
            pass  # a dead watchdog must not take the replay down with it

    def done(self) -> None:
        self.report(DONE)
        try:
            self._process.stdin.close()
        except OSError:
            pass


def spawn(
    *, replica: str, engine_url: str, admin_url: str, admin_token: str, launcher: LauncherConfig, original: Flags, health_timeout_s: float
) -> Watchdog:
    directory = Path(tempfile.mkdtemp(prefix="ico-replay-"))
    state_path = directory / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "replica": replica,
                "engine_url": engine_url,
                "admin_url": admin_url,
                "launcher": asdict(launcher),
                "original": original,
                "health_timeout_s": health_timeout_s,
            }
        ),
        encoding="utf-8",
    )
    log_path = directory / "watchdog.log"
    # Its own session or process group, so the Ctrl-C or kill that takes the
    # replay lab down does not take the watchdog with it.
    detached = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "ico.replay.watchdog", str(state_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        # The token travels in the environment, never in the state file.
        env={**os.environ, "ICO_ADMIN_TOKEN": admin_token},
        **detached,
    )
    return Watchdog(process, log_path)


def last_report(pipe: IO[str]) -> str:
    """Read until the pipe closes and return the last state reported."""
    last = ""
    for line in pipe:
        last = line.strip() or last
    return last


def recover(state: dict, last: str) -> None:
    """Finish what the replay lab left undone. ``last`` is the last step it reported."""
    replica = state["replica"]
    if last == STOPPED:
        launcher_cfg = LauncherConfig(**state["launcher"])
        if launcher_cfg.kind == "manual":
            log.error(
                "replica %s was left stopped or on a test config and only a person can restart it. "
                "It stays drained: restart it with its original flags, then undrain it.",
                replica,
            )
            return
        log.info("restarting %s with its original flags", replica)
        try:
            launcher = launchers.build(launcher_cfg, replica)
            launcher.stop()
            launcher.start(state["original"])
        except LauncherError as e:
            log.error("could not restart %s; it stays drained: %s", replica, e)
            return
        deadline = time.monotonic() + state["health_timeout_s"]
        while not is_healthy(state["engine_url"]):
            if time.monotonic() > deadline:
                log.error("%s did not come back healthy; it stays drained", replica)
                return
            time.sleep(2)
    try:
        GatewayAdmin(state["admin_url"], os.environ.get("ICO_ADMIN_TOKEN", "")).undrain(replica)
    except GatewayError as e:
        log.error("%s is healthy but could not be undrained: %s", replica, e)
        return
    log.info("%s is back in service", replica)


def main(argv: list[str]) -> None:
    state_path = Path(argv[1])
    logging.basicConfig(
        filename=state_path.with_name("watchdog.log"), level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    last = last_report(sys.stdin)
    if last == DONE:
        return
    log.warning("the replay lab went away after reporting %r", last or "nothing")
    if last:  # nothing reported means nothing was touched yet
        recover(state, last)


if __name__ == "__main__":
    main(sys.argv)
