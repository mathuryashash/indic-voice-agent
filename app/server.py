"""FastAPI + WebSocket server.

Protocol on /ws (one connection = one conversation):
  client -> {"type":"audio","data":<base64 webm/wav>, "language": optional}
  client -> {"type":"text","text":"..."}         (typed fallback)
  client -> {"type":"cancel"}                     (barge-in: stop the running turn)
  audio/text messages may carry "voice": "female" | "male" (TTS voice gender, default female)
  client -> {"type":"auth","key":"gsk_..."}       (bring-your-own Groq key, kept in memory for this socket only)
  server -> {"type":"hello", needs_key, free_turns_left, stt}   on connect
  server -> {"type":"auth_ok"} | {"type":"auth_error", message} | {"type":"need_key", message}
  server -> JSON events from app.pipeline; audio events carry base64 mp3.

Cloud mode (STT_BACKEND=groq): speech recognition runs on Groq, so the server needs no GPU.
Visitors get SHARED_TURNS_PER_IP free turns per day on the server's key (if one is set,
capped by SHARED_TURNS_PER_DAY overall), then must paste their own free Groq key.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import load_keys
from .llm import ChatClient, LLMError, groq_provider
from .pipeline import run_turn

log = logging.getLogger("voice")
STATIC = Path(__file__).parent / "static"
STATE: dict = {}
KEY_RE = re.compile(r"gsk_[A-Za-z0-9]{8,}")
STT_BACKEND = os.environ.get("STT_BACKEND", "local")  # local (faster-whisper) | groq
SHARED_PER_IP = int(os.environ.get("SHARED_TURNS_PER_IP", "5"))
SHARED_PER_DAY = int(os.environ.get("SHARED_TURNS_PER_DAY", "150"))
_usage: dict[str, int] = {}  # "YYYY-MM-DD|ip" -> turns on the shared key (in memory; resets on restart)


def redact(text: str) -> str:
    return KEY_RE.sub("gsk_***", text)


def _day() -> str:
    return time.strftime("%Y-%m-%d")


def shared_left(ip: str) -> int:
    if STATE.get("llm") is None:
        return 0
    day = _day()
    total = sum(v for k, v in _usage.items() if k.startswith(day + "|"))
    return max(0, min(SHARED_PER_IP - _usage.get(f"{day}|{ip}", 0), SHARED_PER_DAY - total))


def client_ip(ws: WebSocket) -> str:
    fwd = ws.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (ws.client.host if ws.client else "?")


async def validate_groq_key(key: str) -> bool:
    if not key.startswith("gsk_") or len(key) > 200:
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"})
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def make_stt(api_key: str | None = None):
    if STT_BACKEND == "groq":
        from .stt_groq import GroqSTT
        return GroqSTT(api_key=api_key)
    return STATE.get("stt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_keys()
    try:
        STATE["llm"] = ChatClient()
    except LLMError:
        STATE["llm"] = None  # no server key: every visitor brings their own
        log.warning("No server LLM key: bring-your-own-key only")
    if STT_BACKEND == "groq":
        if os.environ.get("GROQ_API_KEY"):
            STATE["stt"] = make_stt()
    elif os.environ.get("VOICE_NO_STT") != "1":
        from .stt import WhisperSTT
        STATE["stt"] = await asyncio.to_thread(WhisperSTT)
        log.warning("STT ready: %s on %s", STATE["stt"].model_size, STATE["stt"].device)
    yield
    STATE.clear()


app = FastAPI(title="Vidya Saathi voice agent (formerly VTU Saathi)", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    stt = STATE.get("stt")
    return {"ok": True, "stt_backend": STT_BACKEND, "stt": getattr(stt, "model_size", None),
            "device": getattr(stt, "device", None), "server_key": STATE.get("llm") is not None,
            "providers": [p.name for p in getattr(STATE.get("llm"), "providers", [])]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    history: list[dict] = []
    current: asyncio.Task | None = None
    send_lock = asyncio.Lock()
    ip = client_ip(ws)
    own: dict = {}  # this visitor's own key -> their llm/stt clients (memory only, dropped on disconnect)

    async def emit(ev: dict) -> None:
        if ev.get("type") == "audio":
            ev = {**ev, "mp3": base64.b64encode(ev["mp3"]).decode()}
        async with send_lock:
            await ws.send_json(ev)

    def clients() -> tuple:
        """(llm, stt) for this turn: the visitor's own key, else the shared key while quota lasts."""
        if own:
            return own["llm"], own["stt"]
        if shared_left(ip) > 0:
            key = f"{_day()}|{ip}"
            _usage[key] = _usage.get(key, 0) + 1
            return STATE["llm"], STATE.get("stt")
        return None, None

    async def turn(msg: dict) -> None:
        tmp = None
        vidx = 1 if str(msg.get("voice", "")).lower() == "male" else 0
        llm, stt = clients()
        if llm is None:
            await emit({"type": "need_key", "message": "Free demo turns used up. Paste your own free Groq key "
                        "to keep going (console.groq.com/keys, no card needed)."})
            return
        try:
            if msg["type"] == "audio":
                raw = base64.b64decode(msg["data"])
                if len(raw) > 10 * 1024 * 1024:
                    await emit({"type": "error", "message": "Audio too large (max 10 MB)."})
                    return
                fd, tmp = tempfile.mkstemp(suffix=".audio")
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                await run_turn(emit=emit, llm=llm, stt=stt, audio=tmp,
                               language=msg.get("language"), history=history, voice_index=vidx)
            else:
                await run_turn(emit=emit, llm=llm, text=msg["text"][:2000],
                               language=msg.get("language"), history=history, voice_index=vidx)
        except asyncio.CancelledError:
            await emit({"type": "cancelled"})
            raise
        except Exception as exc:  # noqa: BLE001 - report to the client instead of killing the socket
            log.exception("turn failed")
            await emit({"type": "error", "message": redact(f"{type(exc).__name__}: {str(exc)[:200]}")})
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)

    await emit({"type": "hello", "needs_key": shared_left(ip) == 0, "free_turns_left": shared_left(ip),
                "stt": STT_BACKEND})
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "auth":
                key = str(msg.get("key") or "").strip()
                if await validate_groq_key(key):
                    own["llm"] = ChatClient(providers=[groq_provider(key)])
                    own["stt"] = make_stt(key) if STT_BACKEND == "groq" else STATE.get("stt")
                    await emit({"type": "auth_ok"})
                else:
                    await emit({"type": "auth_error", "message": "That Groq key did not work. Check it and try again."})
                continue
            if kind in ("cancel", "audio", "text") and current and not current.done():
                current.cancel()  # barge-in: a new utterance or explicit cancel stops the old reply
                try:
                    await current
                except (asyncio.CancelledError, Exception):
                    pass
            if kind in ("audio", "text"):
                current = asyncio.create_task(turn(msg))
    except WebSocketDisconnect:
        if current and not current.done():
            current.cancel()
