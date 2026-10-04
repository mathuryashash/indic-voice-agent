"""Generate eval audio with edge-tts (different voices) and add white noise to every 3rd clip.

Usage: python eval/make_audio.py   -> eval/audio/<id>.wav (16 kHz mono)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import edge_tts  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402

from eval.cases import CASES  # noqa: E402

OUT = ROOT / "eval" / "audio"


def add_noise(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(x ** 2)) or 1e-9
    noise = rng.normal(0, np.sqrt(p_sig / (10 ** (snr_db / 10))), size=x.shape)
    return np.clip(x + noise, -1, 1).astype("float32")


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(CASES):
        wav = OUT / f"{c['id']}.wav"
        if wav.exists():
            continue
        mp3 = OUT / f"{c['id']}.mp3"
        await edge_tts.Communicate(c["text"], c["voice"]).save(str(mp3))
        x = decode_audio(str(mp3), sampling_rate=16000)
        if c["noise_snr_db"]:
            x = add_noise(x, c["noise_snr_db"], seed=i)
        sf.write(str(wav), x, 16000, subtype="PCM_16")
        mp3.unlink()
        print(c["id"], c["voice"], "noise" if c["noise_snr_db"] else "", f"{len(x)/16000:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
