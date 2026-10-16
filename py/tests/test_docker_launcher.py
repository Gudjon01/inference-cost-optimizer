"""The Docker launcher against a real Docker engine, with a throwaway container of the fake engine."""

import socket
import time
import uuid
from importlib.metadata import version

import httpx
import pytest

from ico.replay import sweep
from ico.replay.launchers import DockerLauncher

# The fake engine ships in the gateway image, as Compose names it for this version.
IMAGE = f"ghcr.io/gudjon01/ico-gateway:{version('ico')}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def healthy(port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    return False


@pytest.fixture
def engine_container():
    """A stand-in for a Compose-managed engine: restart policy, a published port, a network alias, Compose labels."""
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.images.get(IMAGE)
    except Exception:  # noqa: BLE001 - no Docker, or no image: nothing to test against
        pytest.skip(f"needs a running Docker engine and the {IMAGE} image")

    api = client.api
    suffix = uuid.uuid4().hex[:8]
    network, name, port = f"ico-test-{suffix}", f"ico-test-engine-{suffix}", free_port()
    api.create_network(network)
    created = api.create_container(
        IMAGE,
        entrypoint=["/usr/local/bin/mockengine"],
        command=["--model", "mock-model", "--max-num-seqs", "2", "--listen", ":8000"],
        environment={"ENGINE_MODE": "production"},
        labels={"com.docker.compose.project": "someone-elses", "team": "inference"},
        name=name,
        host_config=api.create_host_config(
            port_bindings={8000: ("127.0.0.1", port)}, restart_policy={"Name": "unless-stopped"}, network_mode=network
        ),
        networking_config=api.create_networking_config({network: api.create_endpoint_config(aliases=["engine"])}),
    )
    api.start(created["Id"])
    assert healthy(port), "the original container did not come up"
    try:
        yield client, name, port, network
    finally:
        for container in (name + DockerLauncher.SUFFIX, name):
            try:
                api.remove_container(container, force=True)
            except docker.errors.NotFound:
                pass
        api.remove_network(network)


def test_docker_launcher_runs_a_copy_with_other_flags_and_restores_the_original(engine_container):
    client, name, port, network = engine_container
    api = client.api
    launcher = DockerLauncher(name, client=client)

    # The flags in production are read from the container itself.
    assert launcher.original == {"model": "mock-model", "max-num-seqs": 2, "listen": ":8000"}

    launcher.stop()
    assert api.inspect_container(name)["State"]["Running"] is False

    launcher.start({**launcher.original, "max-num-seqs": 6})
    assert healthy(port), "the copy should serve on the same published port"
    copy = api.inspect_container(launcher.copy_name)
    assert copy["Config"]["Cmd"] == ["--model", "mock-model", "--max-num-seqs", "6", "--listen", ":8000"]
    assert copy["Config"]["Entrypoint"] == ["/usr/local/bin/mockengine"]
    assert "ENGINE_MODE=production" in copy["Config"]["Env"]
    assert copy["HostConfig"]["PortBindings"] == {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]}
    assert copy["HostConfig"]["RestartPolicy"]["Name"] == "no", "a config that crashes must not be restarted in a loop"
    assert "engine" in copy["NetworkSettings"]["Networks"][network]["Aliases"], "reachable under the same name"
    labels = copy["Config"]["Labels"]
    assert labels["team"] == "inference" and labels[DockerLauncher.LABEL] == name
    assert not any(k.startswith("com.docker.compose.") and v for k, v in labels.items()), "Compose must not take the copy for its own"
    for project in ("someone-elses", "ico"):  # set on the container, and inherited from the image Compose built
        claimed = client.containers.list(all=True, filters={"label": f"com.docker.compose.project={project}"})
        assert launcher.copy_name not in [c.name for c in claimed]
    assert api.inspect_container(name)["State"]["Running"] is False, "the original stays stopped while the copy runs"

    # Another config replaces the copy.
    launcher.stop()
    launcher.start({**launcher.original, "max-num-seqs": 9})
    assert healthy(port)
    assert "9" in api.inspect_container(launcher.copy_name)["Config"]["Cmd"]

    # Putting the original back is starting the untouched original container.
    launcher.stop()
    launcher.start(launcher.original)
    assert healthy(port)
    assert api.inspect_container(name)["State"]["Running"] is True
    assert not client.containers.list(all=True, filters={"name": launcher.copy_name}), "no copy is left behind"
    assert api.inspect_container(name)["HostConfig"]["RestartPolicy"]["Name"] == "unless-stopped"


def test_a_copy_that_crashes_stays_down(engine_container):
    client, name, port, _ = engine_container
    launcher = DockerLauncher(name, client=client)
    launcher.stop()
    launcher.start({**launcher.original, "no-such-flag": 1})  # the fake engine exits on an unknown flag
    time.sleep(2)
    assert not healthy(port, timeout_s=2)
    state = client.api.inspect_container(launcher.copy_name)["State"]
    assert state["Running"] is False and state["ExitCode"] != 0
    launcher.stop()
    launcher.start(launcher.original)
    assert healthy(port)


def test_engine_command_lines_split_into_arguments_and_flags():
    assert sweep.parse_args(["serve", "org/model", "--max-num-seqs", "64", "--enforce-eager", "--gpu-memory-utilization=0.9"]) == (
        ["serve", "org/model"],
        {"max-num-seqs": 64, "enforce-eager": True, "gpu-memory-utilization": 0.9},
    )
    assert sweep.parse_args(["--no-enable-prefix-caching", "--host", "0.0.0.0"]) == ([], {"enable-prefix-caching": False, "host": "0.0.0.0"})
    with pytest.raises(sweep.SweepError):
        sweep.parse_args(["--port", "8000", "stray"])
    # And back: what was read is what gets written.
    flags = {"max-num-seqs": 64, "enforce-eager": True, "enable-prefix-caching": False}
    assert sweep.parse_args(sweep.as_args(flags)) == ([], flags)
