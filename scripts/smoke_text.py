"""Run text-only turns against the real LLM + edge-tts (no STT). Usage: python scripts/smoke_text.py"""
import asyncio, sys, json
sys.path.insert(0, ".")
from app.config import load_keys
from app.llm import ChatClient
from app.pipeline import run_turn

async def main(qs):
    load_keys(); llm = ChatClient()
    for q in qs:
        evs = []
        async def emit(e): evs.append(e)
        out = await run_turn(emit=emit, llm=llm, text=q, db_path="tmp_smoke.sqlite")
        print(json.dumps({"q": q, "lang": out["language"], "tools": [(t["name"], t["arguments"]) for t in out["tools"]],
                          "reply": out["reply"], "t": out["timings"],
                          "audio_bytes": sum(len(e["mp3"]) for e in evs if e["type"] == "audio")}, ensure_ascii=False))

asyncio.run(main(sys.argv[1:] or ["When is the CN exam?", "कंप्यूटर नेटवर्क्स की परीक्षा कब है?", "ಥಿಯರಿ ಆಫ್ ಕಂಪ್ಯೂಟೇಶನ್ ಪರೀಕ್ಷೆ ಯಾವಾಗ?"]))
