"""Summaries for the v2 eval + comparison with the 4 Oct baseline. Writes eval/v2/results/RESULTS_v2.md and prints it."""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from eval.stt_bench import wer_cer  # noqa: E402

RES = HERE / "results"
OLD = ROOT / "eval" / "results"


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def f(x, p=1):
    return "–" if x is None else f"{x * 100:.{p}f}%"


def ms(x):
    return "–" if x is None else f"{x * 1000:.0f}"


def stt_tables(lines):
    for path in sorted(RES.glob("stt_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rows = d["rows"]
        lines.append(f"\n## STT: {d['model']} on {d['device']} ({'forced language' if d['forced'] else 'auto-detect'}), "
                     f"{len(rows)} clips, run {d['run_at']}\n")
        lines.append("| set | lang | n | WER | CER | LID acc (final) | STT p50 ms |")
        lines.append("|---|---|---|---|---|---|---|")
        for scope in ("orig", "in"):
            for lang in ("hi", "kn", "en"):
                rs = [r for r in rows if r["scope"] == scope and r["lang"] == lang]
                if not rs:
                    continue
                w, c = wer_cer([r["ref"] for r in rs], [r["hyp"] for r in rs])
                lid = sum(r["final_lang"] == lang for r in rs) / len(rs)
                lines.append(f"| {'original 42' if scope == 'orig' else 'new voices'} | {lang} | {len(rs)} | {f(w)} | {f(c)} | "
                             f"{f(lid)} | {ms(statistics.median(r['seconds'] for r in rs))} |")
        # breakdown by condition and by variant on the new set
        lines.append("\nNew-voice set, by acoustic condition (WER, all 3 languages / kn only):\n")
        lines.append("| condition | n | WER all | WER hi | WER kn | WER en | LID acc |")
        lines.append("|---|---|---|---|---|---|---|")
        for cond in ("clean", "white15", "white5", "phone"):
            rs = [r for r in rows if r["scope"] == "in" and r["cond"] == cond]
            if not rs:
                continue
            cells = []
            for lang in (None, "hi", "kn", "en"):
                sub = [r for r in rs if lang is None or r["lang"] == lang]
                cells.append(f(wer_cer([r["ref"] for r in sub], [r["hyp"] for r in sub])[0]) if sub else "–")
            lid = sum(r["final_lang"] == r["lang"] for r in rs) / len(rs)
            lines.append(f"| {cond} | {len(rs)} | " + " | ".join(cells) + f" | {f(lid)} |")
        lines.append("\nNew-voice set, by speaker variant:\n")
        lines.append("| variant | lang | n | WER | CER | LID acc |")
        lines.append("|---|---|---|---|---|---|")
        by = defaultdict(list)
        for r in rows:
            if r["scope"] == "in":
                by[(r["lang"], r["variant"])].append(r)
        for (lang, var), rs in sorted(by.items()):
            w, c = wer_cer([r["ref"] for r in rs], [r["hyp"] for r in rs])
            lines.append(f"| {var} | {lang} | {len(rs)} | {f(w)} | {f(c)} | {f(sum(r['final_lang'] == lang for r in rs) / len(rs))} |")
        oos = [r for r in rows if r["scope"] == "oos"]
        if oos:
            lines.append("\nUnsupported languages (agent only does hi/kn/en). 'Whisper top-1' = Whisper's own language guess over "
                         "all ~100 languages; 'agent picked' = what the agent would answer in.\n")
            lines.append("| lang | n | Whisper top-1 correct | median top-1 prob | agent picked hi | kn | en |")
            lines.append("|---|---|---|---|---|---|---|")
            for lang in sorted({r["lang"] for r in oos}):
                rs = [r for r in oos if r["lang"] == lang]
                top_ok = sum(r["raw_top_lang"] == lang for r in rs) / len(rs)
                pk = {k: sum(r["final_lang"] == k for r in rs) for k in ("hi", "kn", "en")}
                lines.append(f"| {lang} | {len(rs)} | {f(top_ok, 0)} | {statistics.median(r['raw_top_prob'] for r in rs):.2f} | "
                             f"{pk['hi']} | {pk['kn']} | {pk['en']} |")


def e2e_tables(lines):
    for path in sorted(RES.glob("e2e_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rows = d["rows"]
        provs = sorted({r.get("timings", {}).get("provider") for r in rows} - {None})
        lines.append(f"\n## End-to-end: {path.stem}, n={len(rows)}, STT={d.get('stt_model')} ({d.get('stt_device')}), "
                     f"LLM={','.join(provs)}, run {d.get('run_at')}\n")
        lines.append("| lang | n | WER | detected-lang acc | tool acc | answer acc | reply-in-lang | errors | E2E first audio p50/p95 ms |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        langs = sorted({r["lang"] for r in rows}, key=lambda x: ("hi", "kn", "en").index(x) if x in ("hi", "kn", "en") else 9)
        for lang in langs + ["all"]:
            rs = [r for r in rows if lang == "all" or r["lang"] == lang]
            ok = [r for r in rs if not r.get("error")]
            w = wer_cer([r["ref"] for r in ok], [r["transcript"] for r in ok])[0] if ok and d["scope"] != "oos" else None
            det = sum(r.get("detected_lang") == r["lang"] for r in rs) / len(rs)
            rl = sum(bool(r.get("reply_lang_ok")) for r in rs) / len(rs) if d["scope"] != "oos" else None
            e2e = [r.get("timings", {}).get("e2e_first_audio_s") for r in rs]
            lines.append(f"| {'**all**' if lang == 'all' else lang} | {len(rs)} | {f(w)} | {f(det)} | "
                         f"{f(sum(r['tool_ok'] for r in rs) / len(rs))} | {f(sum(r['answer_ok'] for r in rs) / len(rs))} | {f(rl)} | "
                         f"{sum(1 for r in rs if r.get('error'))} | {ms(pct(e2e, .5))} / {ms(pct(e2e, .95))} |")
        if d["scope"] == "oos":
            lines.append("\nWhat the agent did with unsupported languages (detected language -> count):\n")
            by = defaultdict(lambda: defaultdict(int))
            for r in rows:
                by[r["lang"]][r.get("detected_lang")] += 1
            for lang, cnt in sorted(by.items()):
                lines.append(f"- {lang}: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))


def main():
    lines = ["# Vidya Saathi voice eval v2 (23 Nov 2026, RTX 4060 laptop)",
             "", "Generated by eval/v2/report_v2.py from eval/v2/results/*.json. Not hand-edited."]
    stt_tables(lines)
    e2e_tables(lines)
    out = "\n".join(lines)
    (RES / "RESULTS_v2.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
