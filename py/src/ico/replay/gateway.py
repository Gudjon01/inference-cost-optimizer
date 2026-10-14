"""The gateway's admin API, as the replay lab uses it: drain a replica, put it back."""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx


class GatewayError(RuntimeError):
    pass


class GatewayAdmin:
    def __init__(self, base_url: str, token: str, *, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            transport=transport,
            timeout=10.0,
        )

    def replicas(self) -> list[dict[str, Any]]:
        return self._call("GET", "/admin/replicas")["replicas"]

    def drain(self, name: str) -> dict[str, Any]:
        """Stop new requests to a replica. ``was_drained`` in the answer says whether it already was."""
        return self._call("POST", f"/admin/replicas/{name}/drain")

    def undrain(self, name: str) -> dict[str, Any]:
        return self._call("POST", f"/admin/replicas/{name}/undrain")

    def wait_idle(self, name: str, timeout_s: float, *, sleep: Callable[[float], None] = time.sleep) -> None:
        """Wait for the requests a drained replica was still answering to finish."""
        deadline = time.monotonic() + timeout_s
        while True:
            inflight = next(r["inflight"] for r in self.replicas() if r["name"] == name)
            if inflight == 0:
                return
            if time.monotonic() > deadline:
                raise GatewayError(f"replica {name} still had {inflight} requests in flight {timeout_s:.0f}s after draining")
            sleep(1.0)

    def _call(self, method: str, path: str) -> dict[str, Any]:
        try:
            response = self._client.request(method, path)
        except httpx.HTTPError as e:
            raise GatewayError(f"the gateway admin API at {self._client.base_url} is unreachable: {e}") from None
        if response.status_code == 401:
            raise GatewayError("the gateway refused the admin token; check ICO_ADMIN_TOKEN")
        if response.status_code != 200:
            raise GatewayError(f"gateway {method} {path} answered {response.status_code}: {response.text.strip()[:200]}")
        return response.json()
