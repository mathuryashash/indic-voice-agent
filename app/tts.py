"""Text-to-speech via edge-tts (Microsoft neural voices, needs internet)."""
from __future__ import annotations

from typing import AsyncIterator

import edge_tts


async def synthesize_stream(text: str, voice: str) -> AsyncIterator[bytes]:
    comm = edge_tts.Communicate(text, voice)
    async for chunk in comm.stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            yield chunk["data"]
