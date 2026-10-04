"""Language handling: script-based detection, voices, and sentence splitting for streaming TTS."""
from __future__ import annotations

import re

LANG_NAMES = {"hi": "Hindi", "kn": "Kannada", "en": "English"}
SCRIPTS = {"hi": "Devanagari", "kn": "Kannada", "en": "Latin"}

VOICES = {
    "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "kn": ["kn-IN-SapnaNeural", "kn-IN-GaganNeural"],
    "en": ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
}

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_KANNADA = re.compile(r"[\u0C80-\u0CFF]")
_LATIN = re.compile(r"[A-Za-z]")


def script_counts(text: str) -> dict:
    return {"hi": len(_DEVANAGARI.findall(text)), "kn": len(_KANNADA.findall(text)),
            "en": len(_LATIN.findall(text))}


def detect_text_language(text: str) -> str | None:
    """Language by dominant script. Indic scripts win over Latin as long as they
    make up >=20% of letters, because code-mixed speech ("Computer Networks ಪರೀಕ್ಷೆ")
    has many English technical words. Returns None when there are no letters."""
    c = script_counts(text)
    total = sum(c.values())
    if total == 0:
        return None
    indic = max(("hi", "kn"), key=lambda k: c[k])
    if c[indic] / total >= 0.2:
        return indic
    return "en"


def choose_language(stt_lang: str, stt_prob: float, transcript: str) -> str:
    """Combine Whisper LID with the transcript's script.

    Whisper sometimes writes Kannada speech in Devanagari or Latin when it is
    unsure; we trust the script of the transcript unless LID is confident.
    """
    script_lang = detect_text_language(transcript)
    if script_lang is None:
        return stt_lang if stt_lang in LANG_NAMES else "en"
    if stt_lang in LANG_NAMES and stt_prob >= 0.8 and stt_lang != "en":
        return stt_lang
    return script_lang


def voice_for(lang: str, index: int = 0) -> str:
    vs = VOICES.get(lang, VOICES["en"])
    return vs[index % len(vs)]


_SENT_END = re.compile(r"(.+?[.!?।॥\n]+)(\s+)", re.S)


class SentenceSplitter:
    """Accumulates streamed tokens; yields complete sentences as soon as they end."""

    def __init__(self, min_chars: int = 12):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            m = _SENT_END.match(self.buf)
            if not m:
                break
            sent = m.group(1).strip()
            # avoid splitting on "Dr." / "2." etc.: require a minimum length unless it's the end
            if len(sent) < self.min_chars and m.end() >= len(self.buf):
                break
            if len(sent) < self.min_chars:
                # merge short fragment with the next sentence
                nxt = _SENT_END.match(self.buf, m.end())
                if not nxt:
                    break
                sent = self.buf[: nxt.end(1)].strip()
                self.buf = self.buf[nxt.end():]
            else:
                self.buf = self.buf[m.end():]
            if sent:
                out.append(sent)
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


def clean_for_tts(text: str) -> str:
    """Strip markdown symbols the LLM sometimes emits so TTS doesn't read them out."""
    text = re.sub(r"[*_#`>|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
