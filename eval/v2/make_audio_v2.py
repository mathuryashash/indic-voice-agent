"""Render the v2 clips -> eval/v2/audio/<clip>.wav (16 kHz mono). Skips clips that already exist."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import edge_tts  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402
from scipy.signal import butter, sosfilt, resample_poly  # noqa: E402

from eval.v2.cases_v2 import ALL  # noqa: E402

OUT = HERE / "audio"
SEM = asyncio.Semaphore(6)


def white(x, snr, seed):
    rng = np.random.default_rng(seed)
    p = float(np.mean(x ** 2)) or 1e-9
    return np.clip(x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), x.shape), -1, 1).astype("float32")


def phone(x):
    sos = butter(6, [300, 3400], btype="band", fs=16000, output="sos")
    y = sosfilt(sos, x)
    y = resample_poly(resample_poly(y, 1, 2), 2, 1)  # through 8 kHz and back
    return np.clip(y * 1.5, -1, 1).astype("float32")


def condition(x, cond, seed):
    if cond == "white15":
        return white(x, 15, seed)
    if cond == "white5":
        return white(x, 5, seed)
    if cond == "phone":
        return phone(x)
    return x.astype("float32")


async def render(i, c, log):
    wav = OUT / f"{c['clip']}.wav"
    if wav.exists():
        return
    async with SEM:
        mp3 = Path(tempfile.mkdtemp()) / "a.mp3"
        for attempt in range(4):
            try:
                if c["engine"] == "edge":
                    await edge_tts.Communicate(c["text"], c["voice"], rate=c["rate"], pitch=c["pitch"]).save(str(mp3))
                else:
                    from gtts import gTTS
                    await asyncio.to_thread(lambda: gTTS(c["text"], lang=c["lang"], tld=c["voice"],
                                                         slow=c["engine"] == "gtts_slow").save(str(mp3)))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    log.append({"clip": c["clip"], "error": f"{type(e).__name__}: {str(e)[:150]}"})
                    print("FAIL", c["clip"], e, flush=True)
                    return
                await asyncio.sleep(2 + attempt * 3)
        x = decode_audio(str(mp3), sampling_rate=16000)
        if len(x) < 1600:
            log.append({"clip": c["clip"], "error": "empty audio"})
            print("EMPTY", c["clip"], flush=True)
            return
        x = condition(x, c["cond"], seed=i)
        sf.write(str(wav), x, 16000, subtype="PCM_16")
        print(c["clip"], c["cond"], f"{len(x) / 16000:.1f}s", flush=True)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    await asyncio.gather(*(render(i, c, log) for i, c in enumerate(ALL)))
    (HERE / "make_audio_errors.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done; failures:", len(log))


if __name__ == "__main__":
    asyncio.run(main())
