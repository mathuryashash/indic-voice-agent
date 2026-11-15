"""Read a photo of an exam question (past paper, textbook page, handwritten notes) into plain text.

Uses an OpenAI-compatible vision model. Providers are tried in order; the first that answers wins.
The transcription is treated as the student's question text, never as instructions.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass

import httpx

PROMPT = ("Transcribe the exam question(s) or text in this image exactly, keeping question numbers and marks. "
          "If it is a diagram or circuit, describe it in one or two plain sentences. Output only the text, "
          "no commentary. If nothing readable is in the image, output: NO_TEXT")


@dataclass
class VisionProvider:
    name: str
    base_url: str
    model: str
    key_env: str

    @property
    def key(self) -> str | None:
        return os.environ.get(self.key_env)


def default_vision_providers() -> list[VisionProvider]:
    # Measured 15 Nov 2026 on a test question photo: Nebius gemma-3-27b-it 10 s, NVIDIA nemotron-3.5-super-vl 18 s;
    # NVIDIA llama-3.2-90b-vision and gemma-4-31b timed out at 45-60 s.
    return [
        VisionProvider("nebius-vl", "https://api.tokenfactory.nebius.com/v1",
                       os.environ.get("NEBIUS_VISION_MODEL", "google/gemma-3-27b-it"), "NEBIUS_API_KEY"),
        VisionProvider("nvidia-vl", os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                       os.environ.get("NVIDIA_VISION_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
                       "NVIDIA_API_KEY"),
    ]


class VisionError(RuntimeError):
    pass


def transcribe_image(data: bytes, mime: str = "image/jpeg", providers: list[VisionProvider] | None = None,
                     timeout: float = 60.0, transport=None) -> dict:
    """Return {"text", "provider", "seconds"}. Raises VisionError if every provider fails."""
    if len(data) > 8 * 1024 * 1024:
        raise VisionError("Image too large (max 8 MB).")
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    body_msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                              {"type": "image_url", "image_url": {"url": url}}]}]
    errors = []
    for p in [p for p in (providers or default_vision_providers()) if p.key]:
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=timeout, transport=transport) as c:
                r = c.post(f"{p.base_url}/chat/completions",
                           headers={"Authorization": f"Bearer {p.key}"},
                           json={"model": p.model, "messages": body_msgs, "max_tokens": 700, "temperature": 0})
            if r.status_code != 200:
                errors.append(f"{p.name} HTTP {r.status_code}")
                continue
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            return {"text": text, "provider": p.name, "seconds": round(time.perf_counter() - t0, 2)}
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            errors.append(f"{p.name} {type(exc).__name__}")
    raise VisionError("No vision provider worked: " + "; ".join(errors or ["no vision key configured"]))
