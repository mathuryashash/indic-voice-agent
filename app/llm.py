"""Streaming OpenAI-compatible chat client with tool calling and provider fallback.

Primary: Groq (fast). Fallback: NVIDIA NIM. Keys come from environment variables
(GROQ_API_KEY / NVIDIA_API_KEY) which `app.config.load_keys()` fills from local
.env files without ever printing them.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    key_env: str

    @property
    def key(self) -> str | None:
        return os.environ.get(self.key_env)


def default_providers() -> list[Provider]:
    return [
        Provider("groq", "https://api.groq.com/openai/v1",
                 os.environ.get("GROQ_MODEL_VOICE", "openai/gpt-oss-120b"), "GROQ_API_KEY"),
        Provider("nvidia", os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                 os.environ.get("NVIDIA_MODEL_VOICE", "openai/gpt-oss-20b"), "NVIDIA_API_KEY"),
    ]


@dataclass
class StreamEvent:
    """kind: 'text' (delta), 'tool_calls' (final list), 'done', 'meta'."""
    kind: str
    text: str = ""
    tool_calls: list = field(default_factory=list)
    provider: str = ""


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, providers: list[Provider] | None = None, timeout: float = 30.0, max_retries: int = 2,
                 cooldown_s: float = 30.0, clock=time.monotonic, transport=None):
        self.providers = [p for p in (providers or default_providers()) if p.key]
        if not self.providers:
            raise LLMError("No LLM API key configured (GROQ_API_KEY / NVIDIA_API_KEY).")
        self.timeout = timeout
        self.max_retries = max_retries
        self.cooldown_s = cooldown_s
        self.transport = transport  # httpx transport override (tests)
        self.clock = clock
        self.cooling_until: dict[str, float] = {}  # circuit breaker: provider -> time it may be retried
        self.last_retry_wait_s = 0.0
        self.last_attempts = 0

    def _order(self) -> list[Provider]:
        """Healthy providers first (config order), cooling-down ones last as a last resort."""
        now = self.clock()
        return sorted(self.providers, key=lambda p: self.cooling_until.get(p.name, 0.0) > now)

    async def stream(self, messages: list[dict], tools: list[dict] | None = None,
                     temperature: float = 0.2) -> AsyncIterator[StreamEvent]:
        last_err: Exception | None = None
        self.last_retry_wait_s = 0.0
        self.last_attempts = 0
        order = self._order()
        for pi, prov in enumerate(order):
            is_last = pi == len(order) - 1
            for attempt in range(self.max_retries + 1):
                emitted = False
                self.last_attempts += 1
                try:
                    async for ev in self._stream_one(prov, messages, tools, temperature):
                        if ev.kind == "text":
                            emitted = True  # only spoken tokens make a retry unsafe
                        yield ev
                    self.cooling_until.pop(prov.name, None)
                    return
                except (httpx.HTTPError, LLMError) as exc:
                    last_err = exc
                    if emitted:  # can't transparently retry once tokens reached the user
                        raise
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    retryable = status == 429 or status is None or status >= 500
                    if retryable and not is_last:
                        # rate-limited / down: fail over NOW instead of sleeping, and skip this
                        # provider for cooldown_s on later turns (voice latency budget is tiny)
                        self.cooling_until[prov.name] = self.clock() + self.cooldown_s
                        break
                    if retryable and attempt < self.max_retries:
                        wait = min(8.0, 1.5 * (2 ** attempt))
                        self.last_retry_wait_s += wait
                        await asyncio.sleep(wait)
                        continue
                    break  # non-retryable 4xx or retries exhausted: next provider
        raise LLMError(f"All LLM providers failed: {type(last_err).__name__}: {str(last_err)[:200]}")

    async def _stream_one(self, prov: Provider, messages, tools, temperature):
        body = {"model": prov.model, "messages": messages, "stream": True,
                "temperature": temperature, "max_tokens": 1024}
        if "gpt-oss" in prov.model:
            # reasoning model: keep thinking short so time-to-first-spoken-token stays low
            body["reasoning_effort"] = os.environ.get("VOICE_REASONING_EFFORT", "low")
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {prov.key}", "Content-Type": "application/json"}
        calls: dict[int, dict] = {}
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            async with client.stream("POST", f"{prov.base_url}/chat/completions", json=body,
                                     headers=headers) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise httpx.HTTPStatusError(f"{prov.name} HTTP {resp.status_code}: {resp.text[:200]}",
                                                request=resp.request, response=resp)
                yield StreamEvent("meta", provider=prov.name)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise LLMError(str(chunk["error"])[:200])
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            yield StreamEvent("text", text=delta["content"], provider=prov.name)
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
        if calls:
            yield StreamEvent("tool_calls", tool_calls=[calls[i] for i in sorted(calls)], provider=prov.name)
        yield StreamEvent("done", provider=prov.name)


def now() -> float:
    return time.perf_counter()
