"""The replica under test, reached directly and not through the gateway.

The gateway sends a drained replica nothing, so replay traffic goes straight
to the engine. That also keeps it out of the request log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from ico.replay.prompts import POOL_TEXT, Prompt


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Outcome:
    """How one replayed request went. Times are seconds from the moment it was sent."""

    ok: bool
    ttft_s: float = 0.0
    # Average gap between tokens after the first. Counting tokens, not chunks,
    # keeps it the same number however the engine batches its stream.
    itl_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Engine:
    def __init__(self, base_url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Engine:
        # Connections are kept for the whole test, as a gateway in front of the
        # engine keeps its own: a replay is not there to measure reconnecting.
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None, keepalive_expiry=None)
        # No read timeout on its own: a queued request legitimately waits. The
        # load generator decides how long is too long.
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
        self._client = httpx.AsyncClient(base_url=self.base_url, transport=self._transport, limits=limits, timeout=timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def model(self) -> str:
        """The name the engine serves its model under."""
        response = await self._client.get("/v1/models", timeout=10.0)
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models:
            raise EngineError(f"{self.base_url} serves no model")
        return models[0]["id"]

    async def token_pool(self, model: str) -> list[int]:
        """Ids of ordinary tokens in this model's vocabulary, from its own tokenizer."""
        response = await self._client.post(
            "/tokenize", json={"model": model, "prompt": POOL_TEXT, "add_special_tokens": False}, timeout=30.0
        )
        if response.status_code != 200:
            raise EngineError(f"{self.base_url}/tokenize answered {response.status_code}: {response.text[:200]}")
        return list(response.json()["tokens"])

    async def complete(self, model: str, prompt: Prompt) -> Outcome:
        """Send one prompt and time the stream that comes back."""
        body = {
            "model": model,
            "prompt": list(prompt.tokens),
            "max_tokens": prompt.max_tokens,
            # Random tokens make the model want to stop at once; the recorded
            # request generated this many tokens, so the replay must too.
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        sent = time.perf_counter()
        first = last = 0.0
        prompt_tokens = completion_tokens = 0

        async def trace(event: str, info: dict) -> None:
            # The clock starts when the request goes out on a connection that
            # is ready. Opening one is a cost of the load generator, not of the
            # engine, and on some hosts it is hundreds of milliseconds.
            nonlocal sent
            if event.endswith("send_request_headers.started"):
                sent = time.perf_counter()

        try:
            async with self._client.stream("POST", "/v1/completions", json=body, extensions={"trace": trace}) as response:
                if response.status_code != 200:
                    return Outcome(ok=False)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if usage := chunk.get("usage"):
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                    if any(choice.get("text") for choice in chunk.get("choices") or ()):
                        last = time.perf_counter()
                        first = first or last
        except (httpx.HTTPError, json.JSONDecodeError):
            return Outcome(ok=False)
        if not first or not completion_tokens:
            return Outcome(ok=False)
        gaps = completion_tokens - 1
        return Outcome(
            ok=True,
            ttft_s=first - sent,
            itl_s=(last - first) / gaps if gaps > 0 else 0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def is_healthy(base_url: str, timeout_s: float = 3.0) -> bool:
    """Whether the engine answers its health check. vLLM does once the model is loaded."""
    try:
        return httpx.get(base_url.rstrip("/") + "/health", timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False
