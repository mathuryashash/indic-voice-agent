"""Vidya Saathi on Telegram (aiogram 3, long polling: no public URL or HTTPS needed).

Run:  python -m app.telegram_bot          (needs VIDYA_BOT_TOKEN, GROQ_API_KEY; NEBIUS/NVIDIA keys optional)
Admin: set BOT_ADMIN_IDS=<your chat id> to use /stats.

Students: text, voice notes (Hindi / Kannada / English) or a photo of an exam question. Every answer has
👍 / 👎 and "Quiz me" buttons. Works in groups when the bot is @mentioned or replied to (privacy mode stays on).
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (BotCommand, BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup,
                           Message)
from aiogram.utils.chat_action import ChatActionSender

from .bot_core import MAX_VOICE_S, PRIVACY, Reply, VidyaBot, now_ist, split_message, strip_mention
from .config import load_keys
from .llm import ChatClient, chat_providers

log = logging.getLogger("vidya")

WELCOME = ("Namaste! I'm Vidya Saathi, a free study helper for VTU students.\n\n"
           "Ask me in English, Hindi or Kannada:\n"
           "• type a question (\"explain Dijkstra's algorithm\", \"when is my 18CS51 exam?\")\n"
           "• send a voice note\n"
           "• send a photo of a past-paper question\n"
           "• \"remind me to revise DBMS tomorrow at 7 pm\"\n"
           "• SGPA: \"SGPA for 4 credits A+, 3 credits B, 1 credit O\"\n\n"
           "After an answer, tap Quiz me for 3 practice questions.\n\n")
HELP = ("Commands:\n/language - reply language (auto, English, Hindi, Kannada)\n/voice - spoken replies on/off\n"
        "/reminders - your upcoming reminders\n/privacy - what is stored\n/deletemydata - erase everything\n\n"
        "VTU dates and syllabus come only from VTU data I have; if I don't have it I'll say so. "
        "Always double-check exam dates on vtu.ac.in.")
LANGS = {"auto": "Auto (same as you)", "en": "English", "hi": "हिन्दी", "kn": "ಕನ್ನಡ"}


def kb(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=d) for t, d in r]
                                                 for r in rows])


def answer_kb(reply: Reply) -> InlineKeyboardMarkup | None:
    if reply.event_id is None:
        return None
    row = [("👍", f"r:{reply.event_id}:1"), ("👎", f"r:{reply.event_id}:-1")]
    if reply.can_quiz:
        row.append(("📝 Quiz me", "quiz"))
    return kb([row])


def build(bot_core: VidyaBot, admin_ids: set[str]) -> Dispatcher:
    dp = Dispatcher()
    me: dict[str, str | None] = {"username": None}
    dp["vidya_me"] = me  # filled from getMe at startup

    def addressed(m: Message) -> bool:
        """Private chats: always. Groups: only when @mentioned or when replying to the bot."""
        if m.chat.type == "private":
            return True
        text = m.text or m.caption or ""
        if me["username"] and f"@{me['username']}".lower() in text.lower():
            return True
        r = m.reply_to_message
        return bool(r and r.from_user and r.from_user.username == me["username"])

    async def send_reply(m: Message, reply: Reply) -> None:
        if reply.transcript:
            await m.answer(f"🎙 I heard: {reply.transcript[:500]}")
        parts = split_message(reply.text)
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            await m.answer(part, reply_markup=answer_kb(reply) if last and not reply.voice_ogg else None)
        if reply.voice_ogg:
            await m.answer_voice(BufferedInputFile(reply.voice_ogg, "answer.ogg"), reply_markup=answer_kb(reply))

    @dp.message(CommandStart())
    async def start(m: Message, command: CommandObject):
        chat = str(m.chat.id)
        user = bot_core.store.user(chat)
        src = re.sub(r"[^A-Za-z0-9_-]", "", command.args or "")[:32] or None
        if src and not user["consent_at"]:
            bot_core.store.set(chat, source=src)   # which poster / group invite brought them (no personal data)
        if user["consent_at"]:
            await m.answer(WELCOME + HELP)
            return
        await m.answer(WELCOME + PRIVACY, reply_markup=kb([[("✅ OK, start", "consent")]]))

    @dp.callback_query(F.data == "consent")
    async def consent(c: CallbackQuery):
        bot_core.store.set(str(c.message.chat.id), consent_at=now_ist().isoformat(timespec="seconds"))
        await c.message.edit_reply_markup(reply_markup=None)
        await c.message.answer("You're in. Ask me anything about your subjects. " + HELP.split("\n\n")[0])
        await c.answer()

    @dp.message(Command("help"))
    async def help_(m: Message):
        await m.answer(HELP)

    @dp.message(Command("privacy"))
    async def privacy(m: Message):
        await m.answer(PRIVACY)

    @dp.message(Command("deletemydata"))
    async def delete(m: Message):
        await m.answer("This erases your settings, reminders and usage history. Sure?",
                       reply_markup=kb([[("Yes, delete", "del:yes"), ("Cancel", "del:no")]]))

    @dp.callback_query(F.data.startswith("del:"))
    async def delete_cb(c: CallbackQuery):
        await c.message.edit_reply_markup(reply_markup=None)
        if c.data == "del:yes":
            bot_core.delete_user(c.message.chat.id)
            await c.message.answer("Done. Everything about you is deleted. Send /start to use me again.")
        else:
            await c.message.answer("Nothing deleted.")
        await c.answer()

    @dp.message(Command("language"))
    async def language(m: Message):
        await m.answer("Reply language:", reply_markup=kb([[(v, f"lang:{k}") for k, v in LANGS.items()]]))

    @dp.callback_query(F.data.startswith("lang:"))
    async def language_cb(c: CallbackQuery):
        code = c.data.split(":", 1)[1]
        if code in LANGS:
            bot_core.store.set(str(c.message.chat.id), lang=code)
            await c.message.edit_text(f"Reply language: {LANGS[code]}")
        await c.answer()

    @dp.message(Command("voice"))
    async def voice_setting(m: Message):
        await m.answer("Spoken replies:", reply_markup=kb([[("Only for voice notes", "v:auto"),
                                                            ("Always", "v:on"), ("Never", "v:off")]]))

    @dp.callback_query(F.data.startswith("v:"))
    async def voice_cb(c: CallbackQuery):
        mode = c.data.split(":", 1)[1]
        if mode in ("auto", "on", "off"):
            bot_core.store.set(str(c.message.chat.id), voice=mode)
            await c.message.edit_text({"auto": "Spoken replies: only for voice notes", "on": "Spoken replies: always",
                                       "off": "Spoken replies: never"}[mode])
        await c.answer()

    @dp.message(Command("reminders"))
    async def reminders(m: Message):
        items = bot_core.reminders(m.chat.id)
        await m.answer("\n".join(f"• {r['due_at']} - {r['text']}" for r in items) if items
                       else "No upcoming reminders. Try: \"remind me to revise OS on Friday at 6 pm\".")

    @dp.message(Command("stats"))
    async def stats(m: Message):
        if str(m.from_user.id) not in admin_ids:
            return
        s = bot_core.stats()
        await m.answer("\n".join(f"{k}: {v}" for k, v in s.items()))

    @dp.callback_query(F.data.startswith("r:"))
    async def rate(c: CallbackQuery):
        _, eid, val = c.data.split(":")
        ok = bot_core.rate(c.message.chat.id, int(eid), int(val))
        await c.answer("Thanks for the feedback!" if ok else "")

    @dp.callback_query(F.data == "quiz")
    async def quiz(c: CallbackQuery):
        await c.answer("Making a quiz...")
        async with ChatActionSender.typing(bot=c.bot, chat_id=c.message.chat.id):
            qs, msg = await bot_core.quiz(c.message.chat.id)
        if msg:
            await c.message.answer(msg)
            return
        for q in qs:
            await c.bot.send_poll(c.message.chat.id, question=q["q"], options=q["options"], type="quiz",
                                  correct_option_id=q["answer"], explanation=q["why"] or None,
                                  is_anonymous=False)

    @dp.message(F.voice | F.audio)
    async def voice(m: Message):
        if not addressed(m):
            return
        media = m.voice or m.audio
        if (media.duration or 0) > MAX_VOICE_S:
            await m.answer(f"Please keep voice notes under {MAX_VOICE_S // 60} minutes.")
            return
        async with ChatActionSender.record_voice(bot=m.bot, chat_id=m.chat.id):
            buf = io.BytesIO()
            await m.bot.download(media, destination=buf)
            reply = await bot_core.ask(m.chat.id, audio=buf.getvalue())
        await send_reply(m, reply)

    @dp.message(F.photo)
    async def photo(m: Message):
        if not addressed(m):
            return
        async with ChatActionSender.typing(bot=m.bot, chat_id=m.chat.id):
            buf = io.BytesIO()
            await m.bot.download(m.photo[-1], destination=buf)  # largest size
            reply = await bot_core.ask(m.chat.id, image=buf.getvalue(),
                                       caption=strip_mention(m.caption or "", me["username"]))
        if reply.image_text:
            await m.answer(f"📷 I read: {reply.image_text[:600]}")
        await send_reply(m, reply)

    @dp.message(F.text & ~F.text.startswith("/"))
    async def text(m: Message):
        if not addressed(m):
            return
        q = strip_mention(m.text, me["username"])
        if not q:
            return
        async with ChatActionSender.typing(bot=m.bot, chat_id=m.chat.id):
            reply = await bot_core.ask(m.chat.id, text=q)
        await send_reply(m, reply)

    @dp.message()
    async def other(m: Message):
        if m.chat.type == "private":
            await m.answer("I understand text, voice notes and photos of questions.")

    async def reminder_loop(bot: Bot):
        while True:
            for r in bot_core.due():
                try:
                    await bot.send_message(int(r["owner"]), f"⏰ Reminder: {r['text']}")
                except Exception as exc:  # noqa: BLE001 - user blocked the bot or chat is gone
                    log.warning("reminder %s not delivered: %s", r["id"], type(exc).__name__)
                bot_core.delivered(r["id"])
            await asyncio.sleep(30)

    @dp.startup()
    async def on_startup(bot: Bot):
        me["username"] = (await bot.get_me()).username
        await bot.set_my_commands([BotCommand(command=c, description=d) for c, d in [
            ("start", "Start / help"), ("language", "Reply language"), ("voice", "Spoken replies on/off"),
            ("reminders", "Your reminders"), ("privacy", "What is stored"), ("deletemydata", "Erase my data")]])
        asyncio.create_task(reminder_loop(bot))
        log.info("Vidya Saathi running as @%s", me["username"])

    return dp


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_keys()
    token = os.environ.get("VIDYA_BOT_TOKEN")
    if not token:
        raise SystemExit("Set VIDYA_BOT_TOKEN (from @BotFather). Do not reuse another bot's token.")
    stt = None
    if os.environ.get("GROQ_API_KEY"):
        from .stt_groq import GroqSTT
        stt = GroqSTT()
    core = VidyaBot(llm=ChatClient(chat_providers(), timeout=45.0), stt=stt)
    admins = {a.strip() for a in os.environ.get("BOT_ADMIN_IDS", "").split(",") if a.strip()}
    bot = Bot(token, default=DefaultBotProperties(link_preview_is_disabled=True))
    asyncio.run(build(core, admins).start_polling(bot, allowed_updates=["message", "callback_query"]))


if __name__ == "__main__":
    main()
