"""Speech-to-text with faster-whisper (CTranslate2).

* GPU (float16) when CUDA libs are available, otherwise CPU int8.
* Language identification is restricted to the three supported languages
  (hi / kn / en): we take Whisper's language probabilities and pick the argmax
  among those three, then transcribe with that language forced. Unrestricted
  Whisper LID often confuses Kannada with Telugu/Tamil or Hindi with Urdu.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = ("hi", "kn", "en")


def _add_cuda_dll_dirs() -> None:
    """pip-installed nvidia-cublas/cudnn wheels ship DLLs that Windows won't find on its own."""
    if sys.platform != "win32":
        return
    for base in {Path(sys.prefix), Path(sys.base_prefix)}:
        nv = base / "Lib" / "site-packages" / "nvidia"
        if nv.is_dir():
            for sub in nv.iterdir():
                b = sub / "bin"
                if b.is_dir():
                    os.add_dll_directory(str(b))
                    os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_dirs()
os.environ.setdefault("HF_HOME", "D:/hf-cache")


@dataclass
class STTResult:
    text: str
    language: str
    language_prob: float
    seconds: float
    lid_probs: dict


class WhisperSTT:
    def __init__(self, model_size: str | None = None, device: str | None = None):
        from faster_whisper import WhisperModel

        self.model_size = model_size or os.environ.get("WHISPER_MODEL", "large-v3")
        root = os.environ.get("WHISPER_CACHE", "D:/hf-cache/whisper")
        self.device = device or os.environ.get("WHISPER_DEVICE", "cuda")
        try:
            if self.device != "cuda":
                raise RuntimeError("cpu requested")
            self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16", download_root=root)
            self._warmup()
        except Exception:  # noqa: BLE001 - any CUDA failure -> CPU fallback
            self.device = "cpu"
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8", download_root=root)
        self._lock = threading.Lock()

    def _warmup(self) -> None:
        import numpy as np
        segs, _ = self.model.transcribe(np.zeros(16000, dtype="float32"), language="en", beam_size=1)
        list(segs)

    def detect_language(self, audio) -> tuple[str, float, dict]:
        _, _, all_probs = self.model.detect_language(audio)
        probs = {l: p for l, p in all_probs if l in SUPPORTED}
        # Whisper frequently labels Hindi speech as Urdu (same spoken language) -> fold it into hi
        probs["hi"] = probs.get("hi", 0.0) + dict(all_probs).get("ur", 0.0)
        lang = max(probs, key=probs.get)
        total = sum(probs.values()) or 1.0
        return lang, probs[lang] / total, {k: round(v, 4) for k, v in probs.items()}

    def transcribe(self, path_or_array, language: str | None = None) -> STTResult:
        from faster_whisper.audio import decode_audio

        t0 = time.perf_counter()
        audio = decode_audio(path_or_array) if isinstance(path_or_array, (str, Path)) else path_or_array
        with self._lock:
            if language in SUPPORTED:
                lang, prob, probs = language, 1.0, {}
            else:
                lang, prob, probs = self.detect_language(audio)
            segs, _ = self.model.transcribe(audio, language=lang, beam_size=5, vad_filter=True,
                                            condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segs).strip()
        return STTResult(text, lang, round(prob, 4), time.perf_counter() - t0, probs)
