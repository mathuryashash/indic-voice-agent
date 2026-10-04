"""Text-to-speech via edge-tts (Microsoft neural voices, needs internet)."""
from __future__ import annotations

import time
from typing import AsyncIterator

import edge_tts


async def synthesize_stream(text: str, voice: str) -> AsyncIterator[bytes]:
    comm = edge_tts.Communicate(text, voice)
    async for chunk in comm.stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            yield chunk["data"]


async def synthesize(text: str, voice: str) -> tuple[bytes, float, float]:
    """Return (mp3 bytes, seconds to first audio byte, total seconds)."""
    t0 = time.perf_counter()
    first = None
    parts = []
    async for b in synthesize_stream(text, voice):
        if first is None:
            first = time.perf_counter() - t0
        parts.append(b)
    return b"".join(parts), (first if first is not None else float("nan")), time.perf_counter() - t0
