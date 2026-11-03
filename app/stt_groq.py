"""Cloud speech-to-text via Groq's hosted Whisper large-v3 (no GPU needed on the server).

Same interface as WhisperSTT.transcribe(). Groq has no restricted language-ID API, so when the
user has not pinned a language we use Groq's detected language and fall back to the script of
the transcript (Devanagari -> hi, Kannada -> kn) when Groq reports something outside hi/kn/en.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from . import lang as L
from .stt import STTResult, SUPPORTED

URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_NAMES = {"hindi": "hi", "kannada": "kn", "english": "en", "urdu": "hi"}


class GroqSTT:
    device = "groq-cloud"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.model_size = model or os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")
        self.timeout = timeout

    def transcribe(self, path_or_bytes, language: str | None = None) -> STTResult:
        t0 = time.perf_counter()
        data = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, (str, Path)) else bytes(path_or_bytes)
        form = {"model": self.model_size, "response_format": "verbose_json", "temperature": "0"}
        if language in SUPPORTED:
            form["language"] = language
        r = httpx.post(URL, headers={"Authorization": f"Bearer {self.api_key}"}, data=form,
                       files={"file": ("speech.webm", data, "audio/webm")}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Groq STT HTTP {r.status_code}: {r.text[:160]}")
        j = r.json()
        text = (j.get("text") or "").strip()
        raw = str(j.get("language") or "").lower()
        code = language if language in SUPPORTED else _NAMES.get(raw, raw if raw in SUPPORTED else "")
        script = L.detect_text_language(text)
        if code not in SUPPORTED:
            code = script or "en"
        prob = 1.0 if language in SUPPORTED else (0.9 if code == script or script is None else 0.5)
        return STTResult(text, code, prob, time.perf_counter() - t0, {})
