"""Server-level test: FastAPI WebSocket protocol with mocked LLM/TTS (no network, no GPU)."""
import base64

from fastapi.testclient import TestClient

from app import pipeline, server
from tests.test_pipeline import ScriptedLLM, fake_tts


def test_ws_text_turn(monkeypatch):
    monkeypatch.setenv("VOICE_NO_STT", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy")
    # inject the fake TTS into every turn the server runs
    orig = pipeline.run_turn

    async def run_turn(**kw):
        kw.setdefault("tts_fn", fake_tts)
        return await orig(**kw)
    monkeypatch.setattr(server, "run_turn", run_turn)

    with TestClient(server.app) as client:
        server.STATE["llm"] = ScriptedLLM()
        assert client.get("/health").json()["ok"]
        assert "VTU Saathi" in client.get("/").text
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and hello["needs_key"] is False
            ws.send_json({"type": "text", "text": "When is the CN exam?"})
            events = []
            while True:
                ev = ws.receive_json()
                events.append(ev)
                if ev["type"] in ("done", "error"):
                    break
    kinds = [e["type"] for e in events]
    assert kinds[0] == "stt" and "tool_call" in kinds and kinds[-1] == "done"
    audio = [e for e in events if e["type"] == "audio"]
    assert audio and base64.b64decode(audio[0]["mp3"]).startswith(b"[en-IN-NeerjaNeural]")


def test_ws_barge_in_cancels_running_turn(monkeypatch):
    import asyncio as aio

    from app.llm import StreamEvent

    class Endless:
        async def stream(self, messages, tools=None, temperature=0.2):
            yield StreamEvent("meta", provider="mock")
            while True:
                await aio.sleep(0.01)
                yield StreamEvent("text", text="More words keep coming here. ")

    monkeypatch.setenv("VOICE_NO_STT", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy")
    orig = pipeline.run_turn

    async def run_turn(**kw):
        kw.setdefault("tts_fn", fake_tts)
        return await orig(**kw)
    monkeypatch.setattr(server, "run_turn", run_turn)
    with TestClient(server.app) as client:
        server.STATE["llm"] = Endless()
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "text", "text": "tell me everything"})
            kinds = []
            while "audio" not in kinds:          # reply is streaming and already speaking
                kinds.append(ws.receive_json()["type"])
            ws.send_json({"type": "cancel"})      # user starts talking -> barge-in
            while True:
                k = ws.receive_json()["type"]
                if k == "cancelled":
                    break
                assert k in ("token", "audio")
    assert "done" not in kinds


def test_shared_quota_then_need_key_and_bad_key(monkeypatch):
    monkeypatch.setenv("VOICE_NO_STT", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-dummy")
    monkeypatch.setattr(server, "SHARED_PER_IP", 1)
    server._usage.clear()
    orig = pipeline.run_turn

    async def run_turn(**kw):
        kw.setdefault("tts_fn", fake_tts)
        return await orig(**kw)
    monkeypatch.setattr(server, "run_turn", run_turn)

    async def never_valid(key):
        return False
    monkeypatch.setattr(server, "validate_groq_key", never_valid)
    with TestClient(server.app) as client:
        server.STATE["llm"] = ScriptedLLM()
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["free_turns_left"] == 1
            ws.send_json({"type": "text", "text": "When is the CN exam?"})
            while ws.receive_json()["type"] not in ("done", "error"):
                pass
            ws.send_json({"type": "text", "text": "again"})
            assert ws.receive_json()["type"] == "need_key"
            ws.send_json({"type": "auth", "key": "gsk_notARealKey123456"})
            assert ws.receive_json()["type"] == "auth_error"
    server._usage.clear()


def test_redact_hides_keys():
    assert "gsk_abcdefgh12345" not in server.redact("HTTP 401 for key gsk_abcdefgh12345 xyz")


def test_tts_failure_falls_back_to_browser_voice():
    import asyncio

    from tests.test_pipeline import ScriptedLLM as L2

    async def broken_tts(text, voice):
        raise RuntimeError("403 Invalid response status")
        yield b""  # pragma: no cover

    events = []

    async def emit(ev):
        events.append(ev)

    asyncio.run(pipeline.run_turn(emit=emit, llm=L2(), text="When is the CN exam?", tts_fn=broken_tts))
    kinds = [e["type"] for e in events]
    assert "speak" in kinds and "audio" not in kinds and kinds[-1] == "done"
