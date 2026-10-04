"""One conversational turn: audio -> STT -> LLM (+tools) -> streamed sentence TTS.

`run_turn` is transport-agnostic: it reports everything through an async `emit`
callback, so the WebSocket server, the eval harness and the tests share it.

Events emitted (dicts with a "type" key):
  stt        {text, language, language_prob, lid_probs}
  tool_call  {name, arguments}
  tool_result{name, result}
  token      {text}
  audio      {seq, sentence, mp3: bytes}
  timings    {stt_s, llm_ttft_s, tts_first_audio_s, e2e_first_audio_s, total_s, ...}
  done       {reply, language, tools}
  error      {message}
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Awaitable, Callable

from . import lang as L
from .tools import TOOL_SCHEMAS, run_tool
from .tts import synthesize_stream

Emit = Callable[[dict], Awaitable[None]]
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """You are "VTU Saathi", a voice study assistant for VTU (Visvesvaraya Technological University) \
5th-semester ISE engineering students. Today is {today}.
Rules:
- The user is speaking {lang_name}. Reply ONLY in {lang_name}, written in {script} script \
(technical terms like subject names may stay in English). Keep replies short: 1-3 spoken sentences, no markdown, no lists.
- Use the tools for facts: get_exam_schedule for exam dates/times/halls, search_syllabus for syllabus topics/modules/\
textbooks/credits, set_reminder to save reminders (convert relative times like "tomorrow 6 pm" to YYYY-MM-DD HH:MM \
using today's date), calculate_sgpa / calculate_cgpa for grade calculations (VTU: SGPA = sum(credits*grade point)/sum(credits)).
- Tool arguments must be in English even when the user speaks Hindi or Kannada.
- Never invent exam dates, syllabus content or numbers; if a tool returns an error, say so briefly.
- Say dates naturally, e.g. "14 December".
"""


def build_system_prompt(lang: str, today: datetime | None = None) -> str:
    today = today or datetime.now()
    return SYSTEM_PROMPT.format(today=today.strftime("%A, %Y-%m-%d %H:%M"),
                                lang_name=L.LANG_NAMES[lang], script=L.SCRIPTS[lang])


class TurnTimer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.perf_counter() - self.t0)

    def since(self, name: str) -> float:
        return time.perf_counter() - self.t0 - self.marks[name]


async def run_turn(*, emit: Emit, llm, stt=None, audio=None, text: str | None = None,
                   language: str | None = None, history: list[dict] | None = None,
                   voice_index: int = 0, db_path=None, today: datetime | None = None,
                   tts_fn=synthesize_stream) -> dict[str, Any]:
    timer = TurnTimer()
    history = history if history is not None else []
    timings: dict[str, Any] = {}

    # ---- 1. STT
    if text is None:
        if stt is None or audio is None:
            raise ValueError("need audio+stt or text")
        res = await asyncio.to_thread(stt.transcribe, audio, language)
        text = res.text
        timings["stt_s"] = round(res.seconds, 3)
        lang = L.choose_language(res.language, res.language_prob, text)
        await emit({"type": "stt", "text": text, "language": lang, "whisper_language": res.language,
                    "language_prob": res.language_prob, "lid_probs": res.lid_probs})
    else:
        timings["stt_s"] = 0.0
        lang = language or L.detect_text_language(text) or "en"
        await emit({"type": "stt", "text": text, "language": lang, "whisper_language": None,
                    "language_prob": 1.0, "lid_probs": {}})
    timer.mark("stt_done")

    if not text.strip():
        await emit({"type": "error", "message": "No speech detected."})
        return {"reply": "", "language": lang, "tools": [], "timings": timings, "transcript": ""}

    messages = [{"role": "system", "content": build_system_prompt(lang, today)},
                *history, {"role": "user", "content": text}]

    # ---- 2. LLM with tool rounds; 3. TTS per sentence, started while tokens still stream
    splitter = L.SentenceSplitter()
    tts_tasks: list[asyncio.Task] = []
    audio_q: asyncio.Queue = asyncio.Queue()
    voice = L.voice_for(lang, voice_index)
    tools_used: list[dict] = []
    reply_parts: list[str] = []
    provider = ""

    async def tts_sentence(seq: int, sentence: str) -> tuple[int, str, bytes, float, float]:
        t_start = time.perf_counter()
        first = None
        buf = bytearray()
        async for b in tts_fn(sentence, voice):
            if first is None:
                first = time.perf_counter() - t_start
            buf.extend(b)
        return seq, sentence, bytes(buf), (first or 0.0), time.perf_counter() - t_start

    async def emitter() -> None:
        """Emit audio strictly in sentence order as soon as each one is ready."""
        idx = 0
        while True:
            task = await audio_q.get()
            if task is None:
                return
            seq, sentence, mp3, first, clip_s = await task
            if idx == 0:
                timer.mark("first_audio")
                timings["tts_first_audio_s"] = round(first, 3)  # edge-tts first byte for sentence 1
                timings["tts_first_clip_s"] = round(clip_s, 3)   # whole sentence-1 clip synthesized
            idx += 1
            if mp3:
                await emit({"type": "audio", "seq": seq, "sentence": sentence, "mp3": mp3})

    emitter_task = asyncio.create_task(emitter())

    def schedule(sentence: str) -> None:
        sentence = L.clean_for_tts(sentence)
        if not sentence:
            return
        if not tts_tasks:
            timer.mark("first_sentence")
        t = asyncio.create_task(tts_sentence(len(tts_tasks), sentence))
        tts_tasks.append(t)
        audio_q.put_nowait(t)

    try:
        llm_start = time.perf_counter()
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            tool_calls: list[dict] = []
            round_text = []
            use_tools = TOOL_SCHEMAS if round_no < MAX_TOOL_ROUNDS else None
            async for ev in llm.stream(messages, tools=use_tools):
                if ev.kind == "meta":
                    provider = ev.provider
                elif ev.kind == "text":
                    if "llm_first_token" not in timer.marks:
                        timer.mark("llm_first_token")
                        timings["llm_ttft_s"] = round(time.perf_counter() - llm_start, 3)
                    round_text.append(ev.text)
                    reply_parts.append(ev.text)
                    await emit({"type": "token", "text": ev.text})
                    for s in splitter.feed(ev.text):
                        schedule(s)
                elif ev.kind == "tool_calls":
                    tool_calls = ev.tool_calls
            if not tool_calls:
                break
            if round_no == 0:
                timings["llm_tool_decision_s"] = round(time.perf_counter() - llm_start, 3)
            messages.append({"role": "assistant", "content": "".join(round_text) or None,
                             "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                                             "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                            for i, c in enumerate(tool_calls)]})
            for i, c in enumerate(tool_calls):
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = c["arguments"]
                await emit({"type": "tool_call", "name": c["name"], "arguments": args})
                t_tool = time.perf_counter()
                result = await asyncio.to_thread(run_tool, c["name"], c["arguments"] or "{}", db_path)
                tools_used.append({"name": c["name"], "arguments": args, "result": result,
                                   "seconds": round(time.perf_counter() - t_tool, 4)})
                await emit({"type": "tool_result", "name": c["name"], "result": result})
                messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                                 "content": json.dumps(result, ensure_ascii=False)[:4000]})
        for s in splitter.flush():
            schedule(s)
        audio_q.put_nowait(None)
        await emitter_task
    except BaseException:
        emitter_task.cancel()
        for t in tts_tasks:
            t.cancel()
        raise

    reply = "".join(reply_parts).strip()
    timings["e2e_first_audio_s"] = round(timer.marks["first_audio"], 3) if "first_audio" in timer.marks else None
    timings["total_s"] = round(time.perf_counter() - timer.t0, 3)
    timings["sentences"] = len(tts_tasks)
    timings["provider"] = provider
    history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
    await emit({"type": "timings", **timings})
    await emit({"type": "done", "reply": reply, "language": lang,
                "tools": [{"name": t["name"], "arguments": t["arguments"]} for t in tools_used]})
    return {"reply": reply, "language": lang, "tools": tools_used, "timings": timings, "transcript": text}
