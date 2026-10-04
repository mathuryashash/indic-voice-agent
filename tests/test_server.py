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
