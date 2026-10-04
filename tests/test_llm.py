"""ChatClient: SSE parsing, tool-call delta assembly, provider failover + circuit breaker (no network)."""
import asyncio
import json

import httpx

from app.llm import ChatClient, Provider


def sse(*chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def providers(monkeypatch):
    monkeypatch.setenv("K1", "x")
    monkeypatch.setenv("K2", "y")
    return [Provider("groq", "https://a.test/v1", "m1", "K1"), Provider("nvidia", "https://b.test/v1", "m2", "K2")]


def collect(client, **kw):
    async def go():
        return [e async for e in client.stream([{"role": "user", "content": "hi"}], **kw)]
    return asyncio.run(go())


def test_parses_text_and_assembles_tool_call_deltas(monkeypatch):
    body = sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_exam", "arguments": '{"sub'}}]}}]},
               {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "_schedule", "arguments": 'ject": "CN"}'}}]}}]},
               {"choices": [{"delta": {"content": "ok"}}]})
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, text=body)
    c = ChatClient(providers(monkeypatch), transport=httpx.MockTransport(handler))
    evs = collect(c, tools=[{"type": "function"}])
    tc = next(e for e in evs if e.kind == "tool_calls").tool_calls
    assert tc == [{"id": "c1", "name": "get_exam_schedule", "arguments": '{"subject": "CN"}'}]
    assert [e.text for e in evs if e.kind == "text"] == ["ok"]
    assert seen[0]["stream"] is True and seen[0]["tool_choice"] == "auto"


def test_429_fails_over_immediately_and_cools_down(monkeypatch):
    hits = []

    def handler(req):
        hits.append(req.url.host)
        if req.url.host == "a.test":
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, text=sse({"choices": [{"delta": {"content": "from nvidia"}}]}))
    t = [100.0]
    c = ChatClient(providers(monkeypatch), transport=httpx.MockTransport(handler), clock=lambda: t[0])
    evs = collect(c)
    assert hits == ["a.test", "b.test"] and c.last_retry_wait_s == 0.0   # no sleeping on 429
    assert next(e for e in evs if e.kind == "meta").provider == "nvidia"
    hits.clear()
    collect(c)                       # groq is cooling down -> nvidia tried first
    assert hits == ["b.test"]
    t[0] += 31
    hits.clear()
    collect(c)                       # cooldown expired -> groq tried again
    assert hits[0] == "a.test"


def test_all_providers_failing_raises(monkeypatch):
    import pytest

    from app.llm import LLMError
    c = ChatClient(providers(monkeypatch), max_retries=0,
                   transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")))
    with pytest.raises(LLMError):
        collect(c)


def test_midstream_error_before_any_text_still_fails_over(monkeypatch):
    def handler(req):
        if req.url.host == "a.test":   # HTTP 200, then an error chunk (seen from NIM: 'list index out of range')
            return httpx.Response(200, text='data: {"error": {"message": "boom", "code": 500}}\n\n')
        return httpx.Response(200, text=sse({"choices": [{"delta": {"content": "ok"}}]}))
    c = ChatClient(providers(monkeypatch), transport=httpx.MockTransport(handler))
    evs = collect(c)
    assert [e.text for e in evs if e.kind == "text"] == ["ok"]
