"""End-to-end eval: audio file -> STT -> LLM (+tools) -> streamed TTS, for all 42 cases.

Usage:
  python eval/run_eval.py --mode audio          # full voice pipeline, auto language detection
  python eval/run_eval.py --mode audio --pin    # language pinned (as when the user picks it in the UI)
  python eval/run_eval.py --mode text           # reference transcript fed to the LLM (isolates STT errors)
Writes eval/results/eval_<mode>[_pinned].json and appends to eval/results/RESULTS.md via report.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import lang as L  # noqa: E402
from app.config import load_keys  # noqa: E402
from app.llm import ChatClient  # noqa: E402
from app.pipeline import run_turn  # noqa: E402
from eval.cases import CASES  # noqa: E402
from eval.stt_bench import wer_cer  # noqa: E402

TODAY = datetime(2026, 10, 4, 10, 0)
SLEEP_S = 1.0
TAG = ""


def check_case(case: dict, out: dict) -> dict:
    tools = out["tools"]
    names = [t["name"] for t in tools]
    exp = case["expected_tool"]
    tool_ok = (not names) if exp is None else (exp in names)
    chk = case["check"]
    reasons = []
    if "exam_code" in chk:
        codes = [t["result"].get("exam", {}).get("code") for t in tools if t["name"] == "get_exam_schedule"]
        if chk["exam_code"] not in codes:
            reasons.append(f"exam_code {codes}")
    if "syllabus_code" in chk:
        tops = [t["result"]["results"][0]["code"] for t in tools
                if t["name"] == "search_syllabus" and t["result"].get("results")]
        if chk["syllabus_code"] not in tops:
            reasons.append(f"syllabus_code {tops}")
    if "reminder_time" in chk:
        times = [t["result"].get("due_at") for t in tools if t["name"] == "set_reminder"]
        if chk["reminder_time"] not in times:
            reasons.append(f"reminder_time {times}")
    reply = L.normalize_text(out["reply"])
    for group in chk.get("facts", []):
        if not any(f.lower() in reply.lower() for f in group):
            reasons.append(f"missing fact {group[:3]}")
    reply_lang = L.detect_text_language(reply)
    lang_ok = reply_lang == case["lang"]
    answer_ok = tool_ok and not reasons
    return {"tool_ok": tool_ok, "answer_ok": answer_ok, "reply_lang": reply_lang, "reply_lang_ok": lang_ok,
            "reasons": reasons}


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def summarize(rows: list[dict], mode: str) -> dict:
    s = {"mode": mode, "n": len(rows), "per_lang": {}, "overall": {}}
    groups = {"hi": [], "kn": [], "en": []}
    for r in rows:
        groups[r["lang"]].append(r)
    groups["all"] = rows
    for g, rs in groups.items():
        d = {"n": len(rs),
             "tool_acc": round(sum(r["tool_ok"] for r in rs) / len(rs), 4),
             "answer_acc": round(sum(r["answer_ok"] for r in rs) / len(rs), 4),
             "reply_lang_acc": round(sum(r["reply_lang_ok"] for r in rs) / len(rs), 4),
             "detected_lang_acc": round(sum(r["detected_lang"] == r["lang"] for r in rs) / len(rs), 4),
             "errors": sum(1 for r in rs if r.get("error"))}
        if mode != "text":
            w, c = wer_cer([r["ref"] for r in rs], [r["transcript"] for r in rs])
            d["wer"], d["cer"] = round(w, 4), round(c, 4)
        for key in ("stt_s", "llm_ttft_s", "tts_first_audio_s", "e2e_first_audio_s", "total_s"):
            vals = [r["timings"].get(key) for r in rs if r.get("timings")]
            d[key] = {"p50": pct(vals, 0.5), "p95": pct(vals, 0.95)}
        (s["overall"] if g == "all" else s["per_lang"]).update({g: d} if g != "all" else d)
    return s


async def main(mode: str, pin: bool, only: list[str] | None) -> None:
    load_keys()
    llm = ChatClient()
    stt = None
    if mode == "audio":
        from app.stt import WhisperSTT
        stt = WhisperSTT()
        print("STT", stt.model_size, stt.device, flush=True)
    db = Path(tempfile.mkdtemp()) / "eval_reminders.sqlite"
    rows = []
    for case in CASES:
        if only and case["id"] not in only:
            continue
        events = []

        async def emit(ev):
            events.append(ev)
        kwargs = dict(emit=emit, llm=llm, db_path=db, today=TODAY, voice_index=0)
        if mode == "audio":
            kwargs.update(stt=stt, audio=str(ROOT / "eval" / "audio" / f"{case['id']}.wav"),
                          language=case["lang"] if pin else None)
        else:
            kwargs.update(text=case["text"], language=None)
        row = {"id": case["id"], "lang": case["lang"], "ref": case["text"], "expected_tool": case["expected_tool"],
               "voice": case["voice"], "noisy": bool(case["noise_snr_db"])}
        try:
            out = await run_turn(**kwargs)
            row.update(transcript=out["transcript"], detected_lang=out["language"], reply=out["reply"],
                       tools=[{"name": t["name"], "arguments": t["arguments"]} for t in out["tools"]],
                       timings=out["timings"],
                       audio_bytes=sum(len(e["mp3"]) for e in events if e["type"] == "audio"))
            row.update(check_case(case, out))
        except Exception as exc:  # noqa: BLE001 - record and continue
            row.update(error=f"{type(exc).__name__}: {str(exc)[:200]}", transcript="", detected_lang=None,
                       reply="", tools=[], timings={}, tool_ok=False, answer_ok=False, reply_lang_ok=False,
                       reasons=["exception"])
        rows.append(row)
        print(f"{row['id']} tool={'Y' if row['tool_ok'] else 'N'} ans={'Y' if row['answer_ok'] else 'N'} "
              f"e2e={row['timings'].get('e2e_first_audio_s')} {row.get('reasons') or ''} {row.get('error') or ''}",
              flush=True)
        await asyncio.sleep(SLEEP_S)  # Groq free tier: 8000 tokens/min; ~2-4k tokens per turn
    name = mode + ("_pinned" if pin else "") + (f"_{TAG}" if TAG else "")
    summary = summarize(rows, name)
    summary["stt_model"] = getattr(stt, "model_size", None)
    summary["stt_device"] = getattr(stt, "device", None)
    summary["llm_providers_used"] = sorted({r["timings"].get("provider") for r in rows if r.get("timings")} - {None})
    summary["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                                                          indent=1), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=1))
    print({k: (v["tool_acc"], v["answer_acc"], v.get("wer")) for k, v in summary["per_lang"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audio", "text"], default="audio")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between cases (rate limits)")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    a = ap.parse_args()
    SLEEP_S = a.sleep
    TAG = a.tag
    asyncio.run(main(a.mode, a.pin, a.only))
