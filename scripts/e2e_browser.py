"""Browser E2E: drive the real UI with Chromium fake-media (a WAV file as the microphone).

For each language: launch Chromium with --use-file-for-fake-audio-capture=<wav>, click the mic,
wait for the clip to be "spoken", click stop, then wait for transcript + tool call + audio replies.
Screenshots -> docs/screenshots/e2e_<lang>.png, results -> eval/results/e2e_browser.json

Usage (server must be running on :8765):  python scripts/e2e_browser.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import soundfile as sf
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8765/"
TURNS = [("hi", "hi01"), ("kn", "kn10"), ("en", "en06")]


def run_turn(p, lang: str, case_id: str) -> dict:
    wav = ROOT / "eval" / "audio" / f"{case_id}.wav"
    dur = sf.info(str(wav)).duration
    browser = p.chromium.launch(headless=True, args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={wav.as_posix()}", "--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page(viewport={"width": 1280, "height": 860})
    page.goto(URL)
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=20000)
    t0 = time.perf_counter()
    page.click("#mic")
    page.wait_for_timeout(int(dur * 1000) + 600)  # fake capture plays the file from the start
    page.click("#mic")
    page.wait_for_function("window.__voice.turnsDone >= 1 || window.__voice.events.some(e => e.type === 'error')",
                           timeout=90000)
    # let queued clips finish playing
    page.wait_for_function("window.__voice.clipsPlayed + window.__voice.playErrors >= window.__voice.clipsReceived",
                           timeout=60000)
    wall = time.perf_counter() - t0
    state = page.evaluate("window.__voice")
    shot = ROOT / "docs" / "screenshots" / f"e2e_{lang}.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(shot), full_page=True)
    browser.close()
    ev = state["events"]
    stt = next((e for e in ev if e["type"] == "stt"), {})
    tools = [e for e in ev if e["type"] == "tool_call"]
    timings = next((e for e in ev if e["type"] == "timings"), {})
    done = next((e for e in ev if e["type"] == "done"), {})
    audio = [e for e in ev if e["type"] == "audio"]
    return {"lang": lang, "case": case_id, "transcript": stt.get("text"), "detected_language": stt.get("language"),
            "tool_calls": [{"name": t["name"], "arguments": t["arguments"]} for t in tools],
            "reply": done.get("reply"), "audio_clips_received": state["clipsReceived"],
            "audio_clips_played": state["clipsPlayed"], "audio_play_errors": state["playErrors"],
            "audio_played_seconds": round(state["playedSeconds"], 2), "audio_b64_chars": sum(a["bytes"] for a in audio),
            "timings": timings, "errors": [e for e in ev if e["type"] == "error"],
            "wall_s_from_mic_click": round(wall, 2), "screenshot": str(shot.relative_to(ROOT))}


def main() -> int:
    results = []
    with sync_playwright() as p:
        for lang, cid in TURNS:
            r = run_turn(p, lang, cid)
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    out = ROOT / "eval" / "results" / "e2e_browser.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = all(r["audio_clips_received"] > 0 and r["audio_clips_played"] == r["audio_clips_received"] and r["tool_calls"] and not r["errors"] and r["detected_language"] == r["lang"]
             for r in results)
    print("E2E", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
