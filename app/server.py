"""FastAPI + WebSocket server.

Protocol on /ws (one connection = one conversation):
  client -> {"type":"audio","data":<base64 webm/wav>, "language": optional}
  client -> {"type":"text","text":"..."}         (typed fallback)
  client -> {"type":"cancel"}                     (barge-in: stop the running turn)
  server -> JSON events from app.pipeline; audio events carry base64 mp3.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import load_keys
from .llm import ChatClient
from .pipeline import run_turn

log = logging.getLogger("voice")
STATIC = Path(__file__).parent / "static"
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_keys()
    STATE["llm"] = ChatClient()
    if os.environ.get("VOICE_NO_STT") != "1":
        from .stt import WhisperSTT
        STATE["stt"] = await asyncio.to_thread(WhisperSTT)
        log.warning("STT ready: %s on %s", STATE["stt"].model_size, STATE["stt"].device)
    yield
    STATE.clear()


app = FastAPI(title="VTU Saathi voice agent", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    stt = STATE.get("stt")
    return {"ok": True, "stt": getattr(stt, "model_size", None), "device": getattr(stt, "device", None),
            "providers": [p.name for p in getattr(STATE.get("llm"), "providers", [])]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    history: list[dict] = []
    current: asyncio.Task | None = None
    send_lock = asyncio.Lock()

    async def emit(ev: dict) -> None:
        if ev.get("type") == "audio":
            ev = {**ev, "mp3": base64.b64encode(ev["mp3"]).decode()}
        async with send_lock:
            await ws.send_json(ev)

    async def turn(msg: dict) -> None:
        tmp = None
        try:
            if msg["type"] == "audio":
                raw = base64.b64decode(msg["data"])
                if len(raw) > 10 * 1024 * 1024:
                    await emit({"type": "error", "message": "Audio too large (max 10 MB)."})
                    return
                fd, tmp = tempfile.mkstemp(suffix=".audio")
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                await run_turn(emit=emit, llm=STATE["llm"], stt=STATE.get("stt"), audio=tmp,
                               language=msg.get("language"), history=history)
            else:
                await run_turn(emit=emit, llm=STATE["llm"], text=msg["text"][:2000],
                               language=msg.get("language"), history=history)
        except asyncio.CancelledError:
            await emit({"type": "cancelled"})
            raise
        except Exception as exc:  # noqa: BLE001 - report to the client instead of killing the socket
            log.exception("turn failed")
            await emit({"type": "error", "message": f"{type(exc).__name__}: {str(exc)[:200]}"})
        finally:
            if tmp:
                Path(tmp).unlink(missing_ok=True)

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
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
