"""Cloud smoke test: send real recorded clips + a text turn over /ws to a running server.
Usage: python cloud_smoke.py ws://127.0.0.1:8767/ws
"""
import asyncio, base64, json, sys, time
import websockets

URL = sys.argv[1]
CLIPS = ["hi01", "kn10", "en06"]
ROOT = "D:/ai-build/indic-voice-agent/eval/audio/"


async def turn(ws, msg):
    t0 = time.perf_counter(); await ws.send(json.dumps(msg)); out = {"tools": [], "audio": 0, "speak": 0}
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 90))
        k = ev["type"]
        if k == "stt": out["heard"] = ev["text"]; out["lang"] = ev["language"]
        elif k == "tool_call": out["tools"].append(ev["name"])
        elif k == "audio": out["audio"] += 1; out.setdefault("first_audio_s", round(time.perf_counter() - t0, 2))
        elif k == "speak": out["speak"] += 1
        elif k == "timings": out["stt_s"] = ev.get("stt_s"); out["e2e_s"] = ev.get("e2e_first_audio_s")
        elif k in ("done", "error", "need_key"):
            out["end"] = k; out["reply"] = ev.get("reply") or ev.get("message"); return out


async def main():
    async with websockets.connect(URL, max_size=20_000_000, open_timeout=120) as ws:
        print("hello", json.loads(await asyncio.wait_for(ws.recv(), 120)))
        import os
        key = os.environ.get("BYO_KEY")
        if key:  # act like a visitor pasting their own key
            await ws.send(json.dumps({"type": "auth", "key": key}))
            print("key", json.loads(await asyncio.wait_for(ws.recv(), 60)))
        for c in CLIPS:
            b64 = base64.b64encode(open(ROOT + c + ".wav", "rb").read()).decode()
            print(c, json.dumps(await turn(ws, {"type": "audio", "data": b64}), ensure_ascii=False))
        print("text", json.dumps(await turn(ws, {"type": "text", "text": "How many credits is Big Data Analytics?"}), ensure_ascii=False))

asyncio.run(main())
