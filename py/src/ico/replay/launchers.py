"""How a replica's engine is restarted under different flags.

Every launcher does the same two things, stop the engine and start it with
given flags; they differ in who does the work. Health checking and restoring
the original are the same for all of them and live in run.py.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from typing import Callable

from ico.config import LauncherConfig
from ico.replay import sweep
from ico.replay.sweep import Flags, render


class LauncherError(RuntimeError):
    pass


class Launcher(ABC):
    # Whether start and stop work with nobody at the keyboard. A watchdog can
    # only put the original back through a launcher that does.
    unattended: bool

    @property
    def original(self) -> Flags | None:
        """The engine flags in production, if the launcher can see them. None: take them from the config file."""
        return None

    @abstractmethod
    def stop(self) -> None:
        """Stop the engine so its GPU memory is free. Draining alone does not do that."""

    @abstractmethod
    def start(self, flags: Flags) -> None:
        """Start the engine with these flags. Returns before it is ready to serve."""


class CommandLauncher(Launcher):
    """Runs the customer's own start and stop commands, on the machine the CLI runs on.

    ``{flags}`` in the start command, and in the values of ``env``, is
    replaced by the engine flags.
    """

    unattended = True

    def __init__(self, cfg: LauncherConfig, *, timeout_s: float = 600.0) -> None:
        if not cfg.start or not cfg.stop:
            raise LauncherError("a command launcher needs both a start and a stop command")
        self._cfg = cfg
        self._timeout = timeout_s

    def stop(self) -> None:
        self._run(self._cfg.stop, "")

    def start(self, flags: Flags) -> None:
        self._run(self._cfg.start, render(flags))

    def _run(self, template: str, flags: str) -> None:
        command = template.replace("{flags}", flags)
        env = {**os.environ, **{k: str(v).replace("{flags}", flags) for k, v in self._cfg.env.items()}}
        try:
            done = subprocess.run(
                command, shell=True, cwd=self._cfg.cwd or None, env=env, timeout=self._timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            raise LauncherError(f"`{command}` did not finish within {self._timeout:.0f}s") from None
        if done.returncode != 0:
            tail = "\n".join(done.stdout.strip().splitlines()[-8:])
            raise LauncherError(f"`{command}` exited with {done.returncode}:\n{tail}")


class ManualLauncher(Launcher):
    """Prints the flags and waits for a person to restart the engine."""

    unattended = False

    def __init__(self, replica: str, ask: Callable[[str], None]) -> None:
        self._replica = replica
        self._ask = ask

    def stop(self) -> None:
        # The person stops and starts in one go when asked to start.
        pass

    def start(self, flags: Flags) -> None:
        self._ask(
            f"Restart the engine of replica {self._replica} with these flags:\n\n    {render(flags) or '(no flags)'}\n\n"
            "Press Enter once it has been restarted"
        )


class DockerLauncher(Launcher):
    """Runs each config in a copy of the engine's own container, with only the engine flags changed.

    The original container is stopped while a copy runs, never changed or
    removed, so putting it back is starting it again: whatever it was started
    with, it comes back with. The copy gets the same image, GPUs, volumes,
    environment, ports and networks, and none of the labels Compose uses to
    claim containers as its own.
    """

    unattended = True
    SUFFIX = "-ico-replay"
    LABEL = "ico.replay.copy-of"

    def __init__(self, container: str, *, client=None) -> None:
        import docker  # only this launcher needs it

        try:
            self._client = client or docker.from_env()
            self._attrs = self._client.api.inspect_container(container)
        except docker.errors.DockerException as e:
            raise LauncherError(f"cannot read container {container} from Docker: {e}") from None
        self._docker = docker
        self._name = self._attrs["Name"].lstrip("/")
        try:
            self._leading, self._original = sweep.parse_args(list(self._attrs["Config"]["Cmd"] or []))
        except sweep.SweepError as e:
            raise LauncherError(f"container {self._name}: {e}") from None

    @property
    def original(self) -> Flags:
        return dict(self._original)

    @property
    def copy_name(self) -> str:
        return self._name + self.SUFFIX

    def stop(self) -> None:
        self._remove_copy()
        self._call(self._client.api.stop, self._name)

    def start(self, flags: Flags) -> None:
        self._remove_copy()
        if flags == self._original:
            self._call(self._client.api.start, self._name)
            return

        config = self._attrs["Config"]
        host = dict(self._attrs["HostConfig"])
        # A config that crashes must stay down, not be restarted in a loop.
        host["RestartPolicy"] = {"Name": "no"}
        networks = self._attrs["NetworkSettings"]["Networks"] or {}
        aliases = {name: [a for a in (net.get("Aliases") or []) if a != self._name] for name, net in networks.items()}
        primary = host.get("NetworkMode") if host.get("NetworkMode") in aliases else next(iter(aliases), None)
        api = self._client.api
        # Compose finds its containers by label. Those set on the container
        # are left off the copy; those that come with an image Compose built
        # cannot be left off, so they are blanked.
        from_image = self._call(api.inspect_image, config["Image"])["Config"].get("Labels") or {}
        labels = {
            k: ("" if k.startswith("com.docker.compose.") else v)
            for k, v in (config.get("Labels") or {}).items()
            if not k.startswith("com.docker.compose.") or k in from_image
        }
        labels[self.LABEL] = self._name
        created = self._call(
            api.create_container,
            image=config["Image"],
            command=self._leading + sweep.as_args(flags),
            entrypoint=config.get("Entrypoint"),
            environment=config.get("Env"),
            working_dir=config.get("WorkingDir") or None,
            user=config.get("User") or None,
            labels=labels,
            name=self.copy_name,
            host_config=host,
            networking_config={"EndpointsConfig": {primary: {"Aliases": aliases[primary]}}} if primary else None,
        )
        # Docker attaches one network at creation; the rest are joined before the start.
        for name, names in aliases.items():
            if name != primary:
                self._call(api.connect_container_to_network, created["Id"], name, aliases=names)
        self._call(api.start, created["Id"])

    def _remove_copy(self) -> None:
        try:
            self._client.api.remove_container(self.copy_name, force=True)
        except self._docker.errors.NotFound:
            pass
        except self._docker.errors.DockerException as e:
            raise LauncherError(f"cannot remove {self.copy_name}: {e}") from None

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except self._docker.errors.DockerException as e:
            raise LauncherError(f"Docker refused to {fn.__name__.replace('_', ' ')}: {e}") from None


KINDS = ("manual", "command", "docker")


def build(cfg: LauncherConfig, replica: str, ask: Callable[[str], None] | None = None) -> Launcher:
    where = f"replica {replica}, launcher"
    if cfg.kind == "command":
        return CommandLauncher(cfg)
    if cfg.kind == "docker":
        if not cfg.container:
            raise LauncherError(f"{where}: a docker launcher needs the name of the container")
        return DockerLauncher(cfg.container)
    if cfg.kind != "manual":
        raise LauncherError(f"{where}: unknown kind {cfg.kind!r}, expected one of {', '.join(KINDS)}")
    if ask is None:
        raise LauncherError("the manual launcher needs someone to ask")
    return ManualLauncher(replica, ask)
