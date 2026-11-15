"""Telegram layer: real aiogram dispatcher + handlers, fed real Update objects; only the HTTP session is fake."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest

pytest.importorskip("aiogram", reason="pip install -r requirements-bot.txt")

from aiogram import Bot  # noqa: E402
from aiogram.client.session.base import BaseSession
from aiogram.methods import (AnswerCallbackQuery, EditMessageReplyMarkup, EditMessageText, GetFile, GetMe, SendMessage,
                             SendPoll, SendVoice)
from aiogram.types import File, Message, Update, User

from app.bot_core import Caps, Store, VidyaBot
from app.telegram_bot import build
from tests.test_bot_core import FakeLLM, FakeSTT, fake_ogg, fake_tts

NOW = datetime(2026, 11, 16, 10, 0)
TOKEN = "42:TEST"
BOT_USER = {"id": 42, "is_bot": True, "first_name": "Vidya", "username": "VidyaSaathiBot"}


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls: list[Any] = []
        self.msg_id = 1000

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if isinstance(method, GetMe):
            return User(**BOT_USER)
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="u", file_path="voice/file_1.oga")
        if isinstance(method, (SendMessage, SendVoice, SendPoll)):
            self.msg_id += 1
            return Message.model_validate({"message_id": self.msg_id, "date": 0, "chat": {"id": 1, "type": "private"},
                                           "text": getattr(method, "text", None) or "x"})
        return True  # answerCallbackQuery, editMessage*, setMyCommands, sendChatAction

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        yield b"OggS-fake-voice"

    async def close(self):
        pass

    def sent(self, kind=SendMessage):
        return [c for c in self.calls if isinstance(c, kind)]


def chat(cid=111, kind="private"):
    return {"id": cid, "type": kind, **({"title": "CSE 7A"} if kind != "private" else {"first_name": "Stu"})}


def msg(text=None, cid=111, kind="private", mid=1, **extra):
    m = {"message_id": mid, "date": 0, "chat": chat(cid, kind), "from": {"id": cid, "is_bot": False,
                                                                        "first_name": "Stu"}}
    if text is not None:
        m["text"] = text
        if text.startswith("/"):
            m["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    m.update(extra)
    return Update.model_validate({"update_id": mid, "message": m})


def cb(data, cid=111, uid=None):
    return Update.model_validate({"update_id": 9000 + len(data), "callback_query": {
        "id": "cq1", "chat_instance": "ci", "data": data, "from": {"id": uid or cid, "is_bot": False,
                                                                  "first_name": "Stu"},
        "message": {"message_id": 5, "date": 0, "chat": chat(cid), "text": "x"}}})


def setup(tmp_path, llm=None, admins=frozenset()):
    session = FakeSession()
    bot = Bot(TOKEN, session=session)
    core = VidyaBot(llm=llm or FakeLLM(*["Deadlock: each process holds a resource and waits for another. " * 4] * 5),
                    stt=FakeSTT(), store=Store(tmp_path / "bot.sqlite"), reminders_db=tmp_path / "rem.sqlite",
                    caps=Caps(), tts_fn=fake_tts, to_ogg=fake_ogg, clock=lambda: NOW)
    dp = build(core, set(admins))
    return session, bot, core, dp


async def boot(dp, bot):
    """What on_startup does, minus the background reminder loop."""
    dp["vidya_me"]["username"] = (await bot.get_me()).username


def test_start_consent_then_answer_with_buttons(tmp_path):
    session, bot, core, dp = setup(tmp_path)

    async def go():
        await boot(dp, bot)
        await dp.feed_update(bot, msg("/start rvitm-cr"))
        first = session.sent()[-1]
        assert "Privacy" in first.text and first.reply_markup.inline_keyboard[0][0].callback_data == "consent"
        await dp.feed_update(bot, msg("what is deadlock", mid=2))
        assert "Start" in session.sent()[-1].text          # no answer before consent
        await dp.feed_update(bot, cb("consent"))
        await dp.feed_update(bot, msg("what is deadlock", mid=3))
        ans = session.sent()[-1]
        assert ans.text.startswith("Deadlock")
        buttons = [b.callback_data for b in ans.reply_markup.inline_keyboard[0]]
        assert buttons[0].startswith("r:") and buttons[-1] == "quiz"
        await dp.feed_update(bot, cb(buttons[1]))            # 👎
        assert core.stats()["thumbs_down"] == 1
        assert core.stats()["users_by_source"] == {"rvitm-cr": 1}
    asyncio.run(go())


def test_voice_note_downloads_and_replies_with_voice(tmp_path):
    session, bot, core, dp = setup(tmp_path)
    core.store.set("111", consent_at="x")

    async def go():
        await boot(dp, bot)
        await dp.feed_update(bot, msg(mid=4, voice={"file_id": "v1", "file_unique_id": "vu", "duration": 5}))
        texts = [m.text for m in session.sent()]
        assert texts[0].startswith("🎙 I heard: what is a deadlock")
        voice = session.sent(SendVoice)[-1]
        assert voice.reply_markup is not None
        await dp.feed_update(bot, msg(mid=5, voice={"file_id": "v2", "file_unique_id": "vu2",
                                                           "duration": 400}))
        assert "under 2 minutes" in session.sent()[-1].text
    asyncio.run(go())


def test_photo_question(tmp_path):
    session, bot, core, dp = setup(tmp_path)
    core.store.set("111", consent_at="x")
    core.vision_fn = lambda data, mime: {"text": "Q.5 Explain Dijkstra", "provider": "f", "seconds": 0}

    async def go():
        await boot(dp, bot)
        await dp.feed_update(bot, msg(mid=6, caption="help", photo=[
            {"file_id": "small", "file_unique_id": "s", "width": 90, "height": 90},
            {"file_id": "big", "file_unique_id": "b", "width": 1280, "height": 960}]))
        got = [c for c in session.calls if isinstance(c, GetFile)]
        assert got[-1].file_id == "big"                       # largest photo size is used
        assert session.sent()[0].text.startswith("📷 I read: Q.5 Explain Dijkstra")
    asyncio.run(go())


def test_group_only_when_mentioned(tmp_path):
    session, bot, core, dp = setup(tmp_path)
    core.store.set("-100", consent_at="x")

    async def go():
        await boot(dp, bot)
        await dp.feed_update(bot, msg("anyone has OS notes?", cid=-100, kind="supergroup", mid=7))
        assert session.sent() == []                           # ignored: not addressed
        await dp.feed_update(bot, msg("@VidyaSaathiBot what is paging", cid=-100, kind="supergroup", mid=8))
        assert session.sent()[-1].text.startswith("Deadlock")
        q = core.llm.calls[-1][-1]["content"]
        assert q == "what is paging"                           # mention stripped
    asyncio.run(go())


def test_quiz_sends_quiz_polls(tmp_path):
    quiz = json.dumps({"questions": [{"q": f"Q{i}?", "options": ["a", "b", "c", "d"], "answer": i % 4,
                                      "why": "because"} for i in range(3)]})
    long = "Paging splits memory into fixed-size pages so a process need not be contiguous. " * 4
    session, bot, core, dp = setup(tmp_path, llm=FakeLLM(long, quiz))
    core.store.set("111", consent_at="x")

    async def go():
        await boot(dp, bot)
        await dp.feed_update(bot, msg("what is paging", mid=9))
        await dp.feed_update(bot, cb("quiz"))
        polls = session.sent(SendPoll)
        assert len(polls) == 3 and polls[0].type == "quiz" and polls[1].correct_option_id == 1
        assert polls[0].is_anonymous is False
    asyncio.run(go())


def test_settings_delete_and_admin_stats(tmp_path):
    session, bot, core, dp = setup(tmp_path, admins={"777"})
    core.store.set("111", consent_at="x")

    async def go():
        await boot(dp, bot)
        await dp.feed_update(bot, cb("lang:kn"))
        await dp.feed_update(bot, cb("v:off"))
        u = core.store.user("111")
        assert (u["lang"], u["voice"]) == ("kn", "off")
        await dp.feed_update(bot, msg("/stats", mid=10))
        assert session.sent() == []                           # not an admin: silent
        await dp.feed_update(bot, msg("/stats", cid=777, mid=11))
        assert "users_started" in session.sent()[-1].text
        await dp.feed_update(bot, msg("/deletemydata", mid=12))
        await dp.feed_update(bot, cb("del:yes"))
        assert core.store.user("111")["consent_at"] is None
        assert any(isinstance(c, EditMessageReplyMarkup) for c in session.calls)
        assert any(isinstance(c, EditMessageText) for c in session.calls)
        assert any(isinstance(c, AnswerCallbackQuery) for c in session.calls)
    asyncio.run(go())
