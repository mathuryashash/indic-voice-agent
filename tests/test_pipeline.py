import asyncio
from datetime import datetime

import pytest

from app import lang as L
from app.llm import StreamEvent
from app.pipeline import build_system_prompt, run_turn


# ------------------------------------------------------------------ language
@pytest.mark.parametrize("text,expected", [
    ("डेटा स्ट्रक्चर्स की परीक्षा कब है?", "hi"),
    ("ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್ ಪರೀಕ್ಷೆ ಯಾವಾಗ?", "kn"),
    ("When is the TOC exam?", "en"),
    ("Computer Networks ಪರೀಕ್ಷೆ ಯಾವಾಗ?", "kn"),   # code-mixed: Kannada wins
    ("Computer Networks की परीक्षा कब है?", "hi"),
    ("1234 ?", None),
])
def test_detect_text_language(text, expected):
    assert L.detect_text_language(text) == expected


def test_choose_language_trusts_script_when_lid_unsure():
    # whisper unsure ("en" 0.58) but transcript is Kannada script -> kn
    assert L.choose_language("en", 0.58, "ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್ ಪರೀಕ್ಷೆ") == "kn"
    # confident Kannada LID even though transcript came out Latin
    assert L.choose_language("kn", 0.95, "Computer Networks parikshe yavaga") == "kn"
    assert L.choose_language("en", 0.99, "When is my exam") == "en"
    assert L.choose_language("hi", 0.4, "") == "hi"


def test_voices():
    assert L.voice_for("kn") == "kn-IN-SapnaNeural" and L.voice_for("kn", 1) == "kn-IN-GaganNeural"
    assert L.voice_for("xx").startswith("en-IN")


def test_sentence_splitter_streams_sentences():
    sp = L.SentenceSplitter()
    out = []
    for tok in ["Your Computer", " Networks exam is on", " 14 December. It", " starts at 9:30", " in Room 105."]:
        out += sp.feed(tok)
    assert out == ["Your Computer Networks exam is on 14 December."]
    assert sp.flush() == ["It starts at 9:30 in Room 105."]


def test_sentence_splitter_devanagari_danda_and_short_fragments():
    sp = L.SentenceSplitter()
    out = sp.feed("ठीक है। आपकी परीक्षा 14 दिसंबर को है। ")
    assert out == ["ठीक है। आपकी परीक्षा 14 दिसंबर को है।"]   # "ठीक है।" is too short alone -> merged


def test_clean_for_tts():
    assert L.clean_for_tts("**Module 4**: `TCP`") == "Module 4 : TCP"


def test_system_prompt_language():
    p = build_system_prompt("kn", datetime(2026, 10, 4, 10, 0))
    assert "Kannada" in p and "Kannada script" in p and "2026-10-04" in p


# ------------------------------------------------------------------ pipeline (mocked LLM + TTS)
class ScriptedLLM:
    """Round 1 asks for a tool, round 2 streams a two-sentence answer token by token."""

    def __init__(self, tool_name="get_exam_schedule", args='{"subject": "CN"}', answer=None):
        self.calls = []
        self.tool_name, self.args = tool_name, args
        self.answer = answer or ["Your Computer Networks", " exam is on 14 December.", " Best of luck!"]

    async def stream(self, messages, tools=None, temperature=0.2):
        self.calls.append({"messages": list(messages), "tools": tools})
        yield StreamEvent("meta", provider="mock")
        if len(self.calls) == 1 and self.tool_name:
            yield StreamEvent("tool_calls", tool_calls=[{"id": "c1", "name": self.tool_name, "arguments": self.args}])
        else:
            for t in self.answer:
                await asyncio.sleep(0)
                yield StreamEvent("text", text=t)
        yield StreamEvent("done")


async def fake_tts(text, voice):
    yield f"[{voice}]".encode()
    yield text.encode()


class FakeSTT:
    def __init__(self, text, lang, prob=0.99):
        self.r = (text, lang, prob)

    def transcribe(self, audio, language=None):
        from app.stt import STTResult
        return STTResult(self.r[0], self.r[1], self.r[2], 0.01, {self.r[1]: self.r[2]})


def collect():
    events = []

    async def emit(ev):
        events.append(ev)
    return events, emit


def test_pipeline_tool_round_and_streamed_audio(tmp_path):
    events, emit = collect()
    llm = ScriptedLLM()
    out = asyncio.run(run_turn(emit=emit, llm=llm, text="When is the CN exam?", tts_fn=fake_tts, db_path=tmp_path / "r.db"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "stt" and "tool_call" in kinds and "tool_result" in kinds
    assert kinds[-2:] == ["timings", "done"]
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["result"]["exam"]["date"] == "2026-12-14"
    # tool result was fed back to the LLM in round 2
    assert llm.calls[1]["messages"][-1]["role"] == "tool" and "2026-12-14" in llm.calls[1]["messages"][-1]["content"]
    audio = [e for e in events if e["type"] == "audio"]
    assert [a["seq"] for a in audio] == [0, 1]
    assert audio[0]["sentence"] == "Your Computer Networks exam is on 14 December."
    assert audio[0]["mp3"].startswith(b"[en-IN-NeerjaNeural]")
    assert out["tools"][0]["name"] == "get_exam_schedule"
    t = out["timings"]
    assert t["llm_ttft_s"] is not None and t["e2e_first_audio_s"] is not None and t["sentences"] == 2
    # first audio must be emitted before the last token arrives? at least before 'done'
    assert kinds.index("audio") < kinds.index("done")


def test_pipeline_first_audio_before_llm_finishes():
    """Streaming: TTS for sentence 1 starts while the LLM is still generating."""
    events, emit = collect()
    order = []

    class SlowLLM(ScriptedLLM):
        async def stream(self, messages, tools=None, temperature=0.2):
            yield StreamEvent("meta", provider="mock")
            yield StreamEvent("text", text="First sentence is here. ")
            await asyncio.sleep(0.2)
            order.append("llm_last_token")
            yield StreamEvent("text", text="Second sentence follows.")
            yield StreamEvent("done")

    async def tts(text, voice):
        order.append(f"tts:{text[:5]}")
        yield b"x"

    asyncio.run(run_turn(emit=emit, llm=SlowLLM(), text="hello there", tts_fn=tts))
    assert order.index("tts:First") < order.index("llm_last_token")


def test_pipeline_with_stt_kannada_uses_kannada_voice_and_prompt():
    events, emit = collect()
    llm = ScriptedLLM(tool_name=None, answer=["ನಿಮ್ಮ ಪರೀಕ್ಷೆ ಡಿಸೆಂಬರ್ 14 ರಂದು ಇದೆ."])
    out = asyncio.run(run_turn(emit=emit, llm=llm, stt=FakeSTT("ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್ ಪರೀಕ್ಷೆ ಯಾವಾಗ?", "kn"),
                               audio="x.wav", tts_fn=fake_tts))
    assert out["language"] == "kn"
    assert "Kannada" in llm.calls[0]["messages"][0]["content"]
    audio = [e for e in events if e["type"] == "audio"]
    assert audio and audio[0]["mp3"].startswith(b"[kn-IN-SapnaNeural]")


def test_pipeline_reminder_tool_writes_sqlite(tmp_path):
    from app.tools import list_reminders
    events, emit = collect()
    db = tmp_path / "r.db"
    llm = ScriptedLLM("set_reminder", '{"text": "revise TOC", "time": "2026-10-05 18:00"}', ["Reminder saved for tomorrow."])
    asyncio.run(run_turn(emit=emit, llm=llm, text="remind me", tts_fn=fake_tts, db_path=db))
    assert list_reminders(db)[0]["text"] == "revise TOC"


def test_pipeline_history_and_empty_transcript():
    events, emit = collect()
    hist = []
    llm = ScriptedLLM(tool_name=None, answer=["Hello there, how can I help?"])
    asyncio.run(run_turn(emit=emit, llm=llm, text="hi", history=hist, tts_fn=fake_tts))
    assert [m["role"] for m in hist] == ["user", "assistant"]
    events, emit = collect()
    out = asyncio.run(run_turn(emit=emit, llm=llm, stt=FakeSTT("", "en"), audio="x", tts_fn=fake_tts))
    assert out["reply"] == "" and events[-1]["type"] == "error"


def test_pipeline_cancellation_cancels_tts():
    events, emit = collect()

    class Endless(ScriptedLLM):
        async def stream(self, messages, tools=None, temperature=0.2):
            while True:
                await asyncio.sleep(0.01)
                yield StreamEvent("text", text="More words keep coming here. ")

    async def main():
        t = asyncio.create_task(run_turn(emit=emit, llm=Endless(), text="hi", tts_fn=fake_tts))
        await asyncio.sleep(0.1)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    asyncio.run(main())
    assert not any(e["type"] == "done" for e in events)
