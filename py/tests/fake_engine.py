"""An in-process stand-in for a vLLM replica, with a capacity that tests can set.

It serves ``slots`` requests at a time and makes the rest wait their turn, so
pushing the arrival rate past ``slots / service time`` shows up as growing
time to first token, the way an overloaded engine does.
"""

from __future__ import annotations

import asyncio
import json

import httpx


class FakeEngine:
    def __init__(self, *, slots: int, ttft_s: float = 0.03, token_s: float = 0.02, model: str = "fake-model") -> None:
        self.slots = slots
        self.ttft_s = ttft_s
        self.token_s = token_s
        self.model = model
        self.prompts: list[list[int]] = []
        self.fail_every = 0
        self._free: asyncio.Semaphore | None = None

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200)
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": self.model}]})
        if path == "/tokenize":
            words = json.loads(request.content)["prompt"].split()
            return httpx.Response(200, json={"count": len(words), "tokens": [1000 + hash(w) % 5000 for w in words]})
        if path == "/v1/completions":
            body = json.loads(request.content)
            self.prompts.append(body["prompt"])
            if self.fail_every and len(self.prompts) % self.fail_every == 0:
                return httpx.Response(500, json={"error": {"message": "boom"}})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Stream(self, body))
        return httpx.Response(404)


class _Stream(httpx.AsyncByteStream):
    def __init__(self, engine: FakeEngine, body: dict) -> None:
        self._engine = engine
        self._body = body

    async def __aiter__(self):
        engine = self._engine
        if engine._free is None:
            engine._free = asyncio.Semaphore(engine.slots)
        tokens = self._body["max_tokens"]
        async with engine._free:
            await asyncio.sleep(engine.ttft_s)
            for i in range(tokens):
                if i:
                    await asyncio.sleep(engine.token_s)
                yield b'data: {"choices":[{"index":0,"text":" tok","finish_reason":null}]}\n\n'
        usage = {"prompt_tokens": len(self._body["prompt"]), "completion_tokens": tokens}
        yield b'data: {"choices":[],"usage":' + json.dumps(usage).encode() + b"}\n\n"
        yield b"data: [DONE]\n\n"
