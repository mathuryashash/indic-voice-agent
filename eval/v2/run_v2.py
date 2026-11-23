"""v2 eval runner (23 Nov 2026).

  python eval/v2/run_v2.py stt  --model large-v3 [--model large-v3-turbo]   # STT only, all v2 clips + original 42
  python eval/v2/run_v2.py e2e  [--scope in|oos|orig] [--sleep 1]           # full pipeline (STT -> LLM+tools -> TTS)

Results: eval/v2/results/*.json (one row per clip). Summaries are produced by eval/v2/report_v2.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("VOICE_AGENT_DATA_DIR", str(ROOT / "eval" / "fixtures" / "demo_data"))

from app import lang as L  # noqa: E402
from eval.cases import CASES  # noqa: E402
from eval.v2.cases_v2 import ALL  # noqa: E402

RES = HERE / "results"
TODAY = datetime(2026, 10, 4, 10, 0)


def orig_rows():
    return [dict(c, clip=c["id"], base_id=c["id"], variant="orig:" + c["voice"], engine="edge",
                 cond="white15" if c["noise_snr_db"] else "clean", scope="orig") for c in CASES]


def wav_for(c):
    return (ROOT / "eval" / "audio" / f"{c['id']}.wav") if c["scope"] == "orig" else (HERE / "audio" / f"{c['clip']}.wav")


def pick(scope):
    if scope == "orig":
        return orig_rows()
    return [c for c in ALL if c["scope"] == scope]


def run_stt(models, forced):
    from app.lang import choose_language
    from app.stt import WhisperSTT
    clips = orig_rows() + ALL
    RES.mkdir(parents=True, exist_ok=True)
    for m in models:
        t = time.perf_counter()
        stt = WhisperSTT(m)
        load = time.perf_counter() - t
        print("STT", m, stt.device, f"load {load:.1f}s", flush=True)
        rows = []
        for c in clips:
            w = wav_for(c)
            if not w.exists():
                continue
            lang_arg = c["lang"] if (forced and c["lang"] in ("hi", "kn", "en")) else None
            r = stt.transcribe(str(w), lang_arg)
            from faster_whisper.audio import decode_audio
            # raw Whisper LID over ALL languages (does it notice an unsupported language?)
            raw = None
            if c["scope"] == "oos":
                _, _, allp = stt.model.detect_language(decode_audio(str(w)))
                raw = allp[0]
            rows.append({"clip": c["clip"], "scope": c["scope"], "lang": c["lang"], "variant": c["variant"],
                         "cond": c["cond"], "ref": c["text"], "hyp": r.text, "whisper_lang": r.language,
                         "whisper_prob": r.language_prob, "final_lang": choose_language(r.language, r.language_prob, r.text),
                         "raw_top_lang": raw[0] if raw else None, "raw_top_prob": round(raw[1], 4) if raw else None,
                         "seconds": round(r.seconds, 3)})
            if len(rows) % 50 == 0:
                print(m, len(rows), flush=True)
        name = f"stt_{m}{'_forced' if forced else ''}.json"
        (RES / name).write_text(json.dumps({"model": m, "device": stt.device, "forced": forced, "load_s": round(load, 1),
                                            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", name, len(rows), flush=True)
        del stt


async def run_e2e(scope, sleep_s, tag):
    from app.config import load_keys
    from app.llm import ChatClient
    from app.pipeline import run_turn
    from app.stt import WhisperSTT
    from eval.run_eval import check_case
    load_keys()
    llm = ChatClient()
    print("LLM providers:", [p.name for p in llm.providers], flush=True)
    stt = WhisperSTT()
    print("STT", stt.model_size, stt.device, flush=True)
    db = Path(tempfile.mkdtemp()) / "r.sqlite"
    RES.mkdir(parents=True, exist_ok=True)
    out_path = RES / f"e2e_{scope}{'_' + tag if tag else ''}.json"
    done = {}
    if out_path.exists():
        done = {r["clip"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))["rows"] if not r.get("error")}
    rows = list(done.values())
    for c in pick(scope):
        if c["clip"] in done or not wav_for(c).exists():
            continue
        events = []

        async def emit(ev):
            events.append(ev)
        row = {k: c[k] for k in ("clip", "scope", "lang", "variant", "cond", "base_id", "expected_tool")}
        row["ref"] = c["text"]
        try:
            out = await run_turn(emit=emit, llm=llm, db_path=db, today=TODAY, voice_index=0, stt=stt,
                                 audio=str(wav_for(c)), language=None)
            row.update(transcript=out["transcript"], detected_lang=out["language"], reply=out["reply"],
                       tools=[{"name": t["name"], "arguments": t["arguments"]} for t in out["tools"]],
                       timings=out["timings"])
            if c["scope"] == "oos":
                names = [t["name"] for t in out["tools"]]
                exp = c["expected_tool"]
                row["tool_ok"] = (not names) if exp is None else (exp in names)
                row["reply_lang"] = L.detect_text_language(L.normalize_text(out["reply"]))
                row["answer_ok"] = row["tool_ok"] and (c["base_id"] != "sgpa" or "9.09" in out["reply"]) and \
                    (c["base_id"] != "exam" or "14" in out["reply"])
            else:
                row.update(check_case(c, out))
        except Exception as exc:  # noqa: BLE001
            row.update(error=f"{type(exc).__name__}: {str(exc)[:200]}", tool_ok=False, answer_ok=False, timings={})
        rows.append(row)
        print(f"{row['clip']:28} det={row.get('detected_lang')} tool={'Y' if row.get('tool_ok') else 'N'} "
              f"ans={'Y' if row.get('answer_ok') else 'N'} e2e={row.get('timings', {}).get('e2e_first_audio_s')} "
              f"{row.get('timings', {}).get('provider', '')} {row.get('error', '')}", flush=True)
        if len(rows) % 10 == 0:
            out_path.write_text(json.dumps({"scope": scope, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        await asyncio.sleep(sleep_s)
    out_path.write_text(json.dumps({"scope": scope, "stt_model": stt.model_size, "stt_device": stt.device,
                                    "run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_path.name, len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stt", "e2e"])
    ap.add_argument("--model", action="append")
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--scope", default="in", choices=["in", "oos", "orig"])
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.what == "stt":
        run_stt(a.model or ["large-v3"], a.forced)
    else:
        asyncio.run(run_e2e(a.scope, a.sleep, a.tag))
