"""STT-only benchmark: WER/CER per language + language-ID accuracy for one or more Whisper models.

Usage: python eval/stt_bench.py small large-v3-turbo large-v3
Writes eval/results/stt_bench.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jiwer  # noqa: E402

from app.lang import choose_language  # noqa: E402
from app.stt import WhisperSTT  # noqa: E402
from eval.cases import CASES  # noqa: E402

_PUNCT = re.compile(r"[\u0964\u0965.,!?;:'\"()\[\]{}\-–—‑]+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = s.replace("\u200c", "").replace("\u200d", "")  # ZWNJ/ZWJ appear inconsistently in Kannada
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


def wer_cer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    r = [normalize(x) for x in refs]
    h = [normalize(x) or "<empty>" for x in hyps]
    return jiwer.wer(r, h), jiwer.cer(r, h)


def bench(model: str, forced: bool = False) -> dict:
    t = time.perf_counter()
    stt = WhisperSTT(model)
    load_s = time.perf_counter() - t
    rows = []
    for c in CASES:
        res = stt.transcribe(str(ROOT / "eval" / "audio" / f"{c['id']}.wav"), c["lang"] if forced else None)
        final = choose_language(res.language, res.language_prob, res.text)
        rows.append({"id": c["id"], "lang": c["lang"], "ref": c["text"], "hyp": res.text,
                     "whisper_lang": res.language, "final_lang": final, "seconds": round(res.seconds, 3)})
    out = {"model": model + ("+forced_lang" if forced else ""), "forced_language": forced, "device": stt.device, "load_s": round(load_s, 1), "per_lang": {}}
    for lang in ("hi", "kn", "en"):
        rs = [r for r in rows if r["lang"] == lang]
        w, cr = wer_cer([r["ref"] for r in rs], [r["hyp"] for r in rs])
        out["per_lang"][lang] = {
            "n": len(rs), "wer": round(w, 4), "cer": round(cr, 4),
            "lid_acc_whisper": round(sum(r["whisper_lang"] == lang for r in rs) / len(rs), 4),
            "lid_acc_final": round(sum(r["final_lang"] == lang for r in rs) / len(rs), 4),
            "stt_p50_s": round(statistics.median(r["seconds"] for r in rs), 3)}
    out["rows"] = rows
    del stt
    return out


if __name__ == "__main__":
    args = sys.argv[1:] or ["large-v3-turbo"]
    forced = "--forced" in args
    results = [bench(m, forced) for m in args if not m.startswith("--")]
    (ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
    path = ROOT / "eval" / "results" / "stt_bench.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update({r["model"]: r for r in results})
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results:
        print(r["model"], r["device"], json.dumps(r["per_lang"]))
