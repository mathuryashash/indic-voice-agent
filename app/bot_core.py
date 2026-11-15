"""Transport-free core of the Vidya Saathi Telegram bot.

`VidyaBot.ask()` takes a text question, a voice note or a photo and returns a `Reply`. Everything Telegram-specific
(downloading files, buttons, sending) lives in `app.telegram_bot`, so this module is unit-tested without Telegram.

Stored per student (SQLite, `BOT_DB`): Telegram chat id, settings, consent time, reminders, and per-message metadata
(kind, language, tools used, latency, provider, rating). Message text is NOT stored. `/deletemydata` removes all of it.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from . import lang as L
from .pipeline import run_turn
from .tools import DEFAULT_DB, delete_reminders, due_reminders, list_reminders, mark_reminder_delivered
from .vision import VisionError, transcribe_image

IST = timezone(timedelta(hours=5, minutes=30))
ROOT = Path(__file__).resolve().parent.parent
BOT_DB = Path(os.environ.get("BOT_DB") or ROOT / "bot.sqlite")
HISTORY_TURNS = 3          # remember the last 3 question/answer pairs per chat
HISTORY_IDLE_S = 30 * 60   # forget the conversation after 30 idle minutes
MAX_VOICE_S = 120
VOICE_MAX_CHARS = 450      # spoken part of a reply; measured: a 350 KB Hindi reply was ~87 s of audio
TG_LIMIT = 4000            # Telegram allows 4096 characters per message
LEAKED_TOKENS = re.compile(r"<\|[a-z_]{2,20}\|>")
NO_QUIZ_TOOLS = {"set_reminder", "calculate_sgpa", "calculate_cgpa", "get_exam_schedule", "get_academic_calendar"}

PRIVACY = ("Privacy: Vidya Saathi stores your Telegram chat id, your settings, reminders you ask for, and usage "
           "counts (what kind of message, how long the answer took, your 👍/👎). It does not store the text of your "
           "questions or answers, your voice notes or your photos; those are sent to the AI providers (Groq, Nebius, "
           "NVIDIA) only to answer you. Send /deletemydata at any time to erase everything.")


def now_ist() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


@dataclass
class Caps:
    """Daily limits. They protect the shared free-tier keys from one heavy user or abuse."""
    per_user: int = int(os.environ.get("BOT_CAP_PER_USER", "30"))
    voice: int = int(os.environ.get("BOT_CAP_VOICE", "15"))
    photo: int = int(os.environ.get("BOT_CAP_PHOTO", "8"))
    quiz: int = int(os.environ.get("BOT_CAP_QUIZ", "6"))
    global_day: int = int(os.environ.get("BOT_CAP_GLOBAL", "600"))


@dataclass
class Reply:
    text: str
    event_id: int | None = None
    voice_ogg: bytes | None = None
    transcript: str | None = None     # what we heard in a voice note
    image_text: str | None = None     # what we read in a photo
    language: str = "en"
    can_quiz: bool = False
    limited: bool = False
    error: bool = False
    tools: list[str] = field(default_factory=list)


async def mp3_to_ogg(mp3: bytes) -> bytes:
    """Telegram voice messages must be OGG/Opus; edge-tts gives MP3."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "32k",
        "-f", "ogg", "pipe:1", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate(mp3)
    if proc.returncode != 0 or not out:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')[:200]}")
    return out


async def no_tts(text: str, voice: str):
    """Text-only replies: skip speech synthesis entirely."""
    return
    yield b""  # pragma: no cover - makes this an async generator


def tidy(text: str) -> str:
    """Telegram is sent as plain text: drop markdown the model may still emit."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w*])\*(\S[^*\n]*?)\*(?![\w*])", r"\1", text)   # *emphasis*, not 4*3
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"`{1,3}", "", text)
    return L.normalize_text(text).strip()


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    parts, buf = [], ""
    for para in text.split("\n"):
        while len(para) > limit:  # one giant paragraph: hard-cut it
            if buf:
                parts.append(buf)
                buf = ""
            parts.append(para[:limit])
            para = para[limit:]
        if len(buf) + len(para) + 1 > limit:
            parts.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        parts.append(buf)
    return [p for p in parts if p.strip()] or [""]


def strip_mention(text: str, username: str | None) -> str:
    if not username:
        return text.strip()
    return re.sub(rf"@{re.escape(username)}\b", "", text, flags=re.I).strip()


QUIZ_PROMPT = """Write exactly 3 multiple-choice questions that test a VTU engineering student on the topic below.
Each question has 4 options and exactly one correct option. Write them in {lang_name} ({script} script; technical terms
may stay in English). Question at most 250 characters, each option at most 90 characters, explanation at most 180.
If the topic is not an engineering or study topic, return {{"questions": []}}.
Return ONLY this JSON, nothing else:
{{"questions": [{{"q": "...", "options": ["...", "...", "...", "..."], "answer": 0, "why": "..."}}]}}

Topic (the student's question and the answer they got):
{topic}"""


def parse_quiz(raw: str) -> list[dict]:
    """Validate the model's quiz JSON; return only well-formed questions (Telegram quiz poll limits)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for q in data.get("questions", [])[:3]:
        opts = [str(o).strip()[:100] for o in q.get("options", [])]
        ans = q.get("answer")
        question = str(q.get("q", "")).strip()
        if (question and len(opts) == 4 and all(opts) and len(set(opts)) == 4
                and isinstance(ans, int) and 0 <= ans < 4):
            out.append({"q": question[:300], "options": opts, "answer": ans,
                        "why": str(q.get("why", "")).strip()[:200]})
    return out


class Store:
    def __init__(self, path: Path | str = BOT_DB):
        self.path = str(path)
        with self._con() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS users (chat_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL,
                consent_at TEXT, lang TEXT NOT NULL DEFAULT 'auto', voice TEXT NOT NULL DEFAULT 'auto',
                source TEXT);
            CREATE TABLE IF NOT EXISTS usage (day TEXT NOT NULL, chat_id TEXT NOT NULL, kind TEXT NOT NULL,
                n INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (day, chat_id, kind));
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                day TEXT NOT NULL, chat_id TEXT NOT NULL, kind TEXT NOT NULL, language TEXT, tools TEXT,
                provider TEXT, seconds REAL, ok INTEGER NOT NULL, rating INTEGER);
            """)

    def _con(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def user(self, chat_id: str) -> dict:
        with self._con() as con:
            con.execute("INSERT OR IGNORE INTO users(chat_id, first_seen) VALUES (?,?)",
                        (chat_id, now_ist().isoformat(timespec="seconds")))
            row = con.execute("SELECT consent_at, lang, voice FROM users WHERE chat_id=?", (chat_id,)).fetchone()
        return {"consent_at": row[0], "lang": row[1], "voice": row[2]}

    def set(self, chat_id: str, **fields: Any) -> None:
        self.user(chat_id)
        for k, v in fields.items():
            if k not in ("consent_at", "lang", "voice", "source"):
                raise ValueError(k)
            with self._con() as con:
                con.execute(f"UPDATE users SET {k}=? WHERE chat_id=?", (v, chat_id))

    def used(self, day: str, chat_id: str | None = None, kind: str | None = None) -> int:
        q, args = "SELECT COALESCE(SUM(n),0) FROM usage WHERE day=?", [day]
        if chat_id is not None:
            q += " AND chat_id=?"
            args.append(chat_id)
        if kind is not None:
            q += " AND kind=?"
            args.append(kind)
        with self._con() as con:
            return con.execute(q, args).fetchone()[0]

    def count(self, day: str, chat_id: str, kind: str) -> None:
        with self._con() as con:
            con.execute("INSERT INTO usage(day, chat_id, kind, n) VALUES (?,?,?,1) "
                        "ON CONFLICT(day, chat_id, kind) DO UPDATE SET n=n+1", (day, chat_id, kind))

    def log(self, chat_id: str, kind: str, ok: bool, language: str | None = None, tools: list[str] | None = None,
            provider: str | None = None, seconds: float | None = None) -> int:
        t = now_ist()
        with self._con() as con:
            cur = con.execute("INSERT INTO events(ts, day, chat_id, kind, language, tools, provider, seconds, ok) "
                              "VALUES (?,?,?,?,?,?,?,?,?)",
                              (t.isoformat(timespec="seconds"), t.strftime("%Y-%m-%d"), chat_id, kind, language,
                               ",".join(tools or []), provider, seconds, int(ok)))
            return cur.lastrowid

    def rate(self, event_id: int, chat_id: str, rating: int) -> bool:
        with self._con() as con:
            return con.execute("UPDATE events SET rating=? WHERE id=? AND chat_id=?",
                               (1 if rating > 0 else -1, event_id, chat_id)).rowcount == 1

    def delete(self, chat_id: str) -> int:
        with self._con() as con:
            n = sum(con.execute(f"DELETE FROM {t} WHERE chat_id=?", (chat_id,)).rowcount
                    for t in ("events", "usage", "users"))
        return n

    def stats(self, today: str) -> dict:
        week_ago = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=6)).strftime("%Y-%m-%d")
        with self._con() as con:
            one = lambda q, *a: con.execute(q, a).fetchone()[0]  # noqa: E731
            kinds = dict(con.execute("SELECT kind, COUNT(*) FROM events WHERE day=? GROUP BY kind", (today,)))
            secs = sorted(r[0] for r in con.execute(
                "SELECT seconds FROM events WHERE day=? AND ok=1 AND seconds IS NOT NULL", (today,)))
            return {
                "users_started": one("SELECT COUNT(*) FROM users WHERE consent_at IS NOT NULL"),
                "users_by_source": dict(con.execute("SELECT COALESCE(source,'direct'), COUNT(*) FROM users "
                                                    "WHERE consent_at IS NOT NULL GROUP BY 1")),
                "active_today": one("SELECT COUNT(DISTINCT chat_id) FROM events WHERE day=?", today),
                "active_7d": one("SELECT COUNT(DISTINCT chat_id) FROM events WHERE day>=?", week_ago),
                "returning_users": one("SELECT COUNT(*) FROM (SELECT chat_id FROM events GROUP BY chat_id "
                                       "HAVING COUNT(DISTINCT day)>=2)"),
                "messages_today": sum(kinds.values()),
                "by_kind_today": kinds,
                "failed_today": one("SELECT COUNT(*) FROM events WHERE day=? AND ok=0", today),
                "thumbs_up": one("SELECT COUNT(*) FROM events WHERE rating=1"),
                "thumbs_down": one("SELECT COUNT(*) FROM events WHERE rating=-1"),
                "median_answer_s_today": secs[len(secs) // 2] if secs else None,
            }


class VidyaBot:
    def __init__(self, llm, stt=None, store: Store | None = None, reminders_db: Path | str = DEFAULT_DB,
                 caps: Caps | None = None, vision_fn: Callable = transcribe_image, tts_fn=None, to_ogg=mp3_to_ogg,
                 clock: Callable[[], datetime] = now_ist):
        from .tts import synthesize_stream
        self.llm, self.stt = llm, stt
        self.store = store or Store()
        self.reminders_db = reminders_db
        self.caps = caps or Caps()
        self.vision_fn = vision_fn
        self.tts_fn = tts_fn or synthesize_stream
        self.to_ogg = to_ogg
        self.clock = clock
        self.history: dict[str, list[dict]] = {}
        self.last_seen: dict[str, float] = {}
        self.last_topic: dict[str, tuple[str, str]] = {}  # chat -> (question, answer) for "Quiz me"
        self.locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------ limits
    def _limit_message(self, chat_id: str, kind: str) -> str | None:
        day = self.clock().strftime("%Y-%m-%d")
        if self.store.used(day) >= self.caps.global_day:
            return "Vidya Saathi has hit today's limit for everyone (it runs on free AI quotas). Please try tomorrow."
        if self.store.used(day, chat_id) >= self.caps.per_user:
            return f"You've used today's {self.caps.per_user} questions. The limit resets at midnight."
        cap = {"voice": self.caps.voice, "photo": self.caps.photo, "quiz": self.caps.quiz}.get(kind)
        if cap is not None and self.store.used(day, chat_id, kind) >= cap:
            return f"You've used today's {cap} {kind} messages. You can still type your question."
        return None

    def _history(self, chat_id: str) -> list[dict]:
        if time.monotonic() - self.last_seen.get(chat_id, 0) > HISTORY_IDLE_S:
            self.history[chat_id] = []
        self.last_seen[chat_id] = time.monotonic()
        return self.history.setdefault(chat_id, [])

    # ------------------------------------------------------------ main entry
    async def ask(self, chat_id: str | int, *, text: str | None = None, audio: bytes | None = None,
                  image: bytes | None = None, image_mime: str = "image/jpeg", caption: str | None = None) -> Reply:
        chat_id = str(chat_id)
        user = self.store.user(chat_id)
        if not user["consent_at"]:
            return Reply("Please tap Start (or send /start) and accept the short privacy note first.", limited=True)
        kind = "voice" if audio is not None else "photo" if image is not None else "text"
        if (msg := self._limit_message(chat_id, kind)) is not None:
            return Reply(msg, limited=True)
        lock = self.locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            return await self._ask(chat_id, user, kind, text, audio, image, image_mime, caption)

    async def _ask(self, chat_id, user, kind, text, audio, image, image_mime, caption) -> Reply:
        t0 = time.perf_counter()
        day = self.clock().strftime("%Y-%m-%d")
        self.store.count(day, chat_id, kind)
        pinned = None if user["lang"] == "auto" else user["lang"]
        image_text = None
        if kind == "photo":
            try:
                res = await asyncio.to_thread(self.vision_fn, image, image_mime)
            except VisionError:
                self.store.log(chat_id, kind, ok=False, seconds=time.perf_counter() - t0)
                return Reply("I couldn't read that photo right now. Please type the question instead.", error=True)
            image_text = res["text"].strip()
            if not image_text or image_text.upper().startswith("NO_TEXT"):
                self.store.log(chat_id, kind, ok=False, seconds=time.perf_counter() - t0)
                return Reply("I couldn't find any text in that photo. Try a sharper, well-lit photo of the question.",
                             error=True)
            text = (f"{caption.strip()}\n\n" if caption and caption.strip() else "") + \
                f"Help me answer this exam question from my photo:\n\"\"\"\n{image_text[:3000]}\n\"\"\""
        voice_out = user["voice"] == "on" or (user["voice"] == "auto" and kind == "voice")
        mp3_parts: list[bytes] = []
        spoken = [0]

        async def emit(ev: dict) -> None:
            # Speak only the opening sentences (about 30-40 s); the full answer is always sent as text.
            if ev.get("type") == "audio" and ev.get("mp3") and spoken[0] < VOICE_MAX_CHARS:
                mp3_parts.append(ev["mp3"])
                spoken[0] += len(ev.get("sentence") or "")

        history = self._history(chat_id)
        out = None
        for attempt in range(2):
            mp3_parts.clear()
            spoken[0] = 0
            turn_history = history if attempt == 0 else []
            try:
                out = await run_turn(
                    emit=emit, llm=self.llm, stt=self.stt if kind == "voice" else None,
                    audio=audio if kind == "voice" else None, text=None if kind == "voice" else (text or "")[:4000],
                    language=pinned if kind == "voice" else (pinned or L.detect_text_language(caption or text or "")),
                    history=turn_history, db_path=self.reminders_db, today=self.clock(),
                    tts_fn=self.tts_fn if voice_out else no_tts,
                    channel="chat_voice" if kind == "voice" else "chat", owner=chat_id)
            except Exception:  # noqa: BLE001 - every provider failed or STT broke: tell the student, keep it alive
                self.store.log(chat_id, kind, ok=False, seconds=time.perf_counter() - t0)
                return Reply("I'm overloaded right now. Please try again in a minute.", error=True)
            if not LEAKED_TOKENS.search(out["reply"]):
                break
            # gpt-oss sometimes emits raw chat-template tokens (seen once, 15 Nov 2026): drop that turn, retry once
            # without history, and never show or speak it.
            if turn_history and turn_history[-1].get("content") == out["reply"]:
                del turn_history[-2:]
            out = {**out, "reply": ""} if attempt == 1 else out
        if LEAKED_TOKENS.search(out["reply"]) or not out["reply"]:
            mp3_parts.clear()
        elif turn_history is not history:  # the clean retry ran without history: remember it now
            history.extend([{"role": "user", "content": out.get("transcript") or text or ""},
                            {"role": "assistant", "content": out["reply"]}])
        del history[: max(0, len(history) - 2 * HISTORY_TURNS)]
        reply = tidy(LEAKED_TOKENS.sub("", out["reply"]))
        if not reply:
            ok = bool(out.get("transcript", "").strip())
            self.store.log(chat_id, kind, ok=False, seconds=time.perf_counter() - t0)
            return Reply("I didn't catch that. Please try again." if not ok else "Sorry, I couldn't answer that.",
                         error=True)
        voice_ogg = None
        if voice_out and mp3_parts:
            try:
                voice_ogg = await self.to_ogg(b"".join(mp3_parts))
            except Exception:  # noqa: BLE001 - the text answer still goes out
                voice_ogg = None
        tools = [t["name"] for t in out["tools"]]
        eid = self.store.log(chat_id, kind, ok=True, language=out["language"], tools=tools,
                             provider=out["timings"].get("provider"), seconds=round(time.perf_counter() - t0, 2))
        # Offer a quiz only after a real explanation, not after a reminder, a calculation or a refusal.
        can_quiz = len(reply) >= 200 and not set(tools) & NO_QUIZ_TOOLS
        if can_quiz:
            self.last_topic[chat_id] = (out.get("transcript") or text or "", reply)
        return Reply(reply, event_id=eid, voice_ogg=voice_ogg,
                     transcript=out.get("transcript") if kind == "voice" else None, image_text=image_text,
                     language=out["language"], can_quiz=can_quiz, tools=tools)

    # ------------------------------------------------------------ extras
    async def quiz(self, chat_id: str | int) -> tuple[list[dict], str | None]:
        """3 quiz questions on the last topic. Returns (questions, message-if-none)."""
        chat_id = str(chat_id)
        if chat_id not in self.last_topic:
            return [], "Ask me a question first, then tap Quiz me."
        if (msg := self._limit_message(chat_id, "quiz")) is not None:
            return [], msg
        self.store.count(self.clock().strftime("%Y-%m-%d"), chat_id, "quiz")
        q, a = self.last_topic[chat_id]
        lang = L.detect_text_language(a) or "en"
        prompt = QUIZ_PROMPT.format(lang_name=L.LANG_NAMES[lang], script=L.SCRIPTS[lang],
                                    topic=f"Question: {q[:1500]}\nAnswer: {a[:2500]}")
        raw = []
        try:
            async for ev in self.llm.stream([{"role": "user", "content": prompt}], tools=None, temperature=0.4):
                if ev.kind == "text":
                    raw.append(ev.text)
        except Exception:  # noqa: BLE001
            self.store.log(chat_id, "quiz", ok=False)
            return [], "Couldn't make a quiz right now. Try again in a minute."
        qs = parse_quiz("".join(raw))
        self.store.log(chat_id, "quiz", ok=bool(qs), language=lang)
        return qs, (None if qs else "Couldn't make a good quiz on that one. Try another question.")

    def rate(self, chat_id: str | int, event_id: int, rating: int) -> bool:
        return self.store.rate(event_id, str(chat_id), rating)

    def reminders(self, chat_id: str | int) -> list[dict]:
        return list_reminders(self.reminders_db, owner=str(chat_id))

    def due(self) -> list[dict]:
        return due_reminders(self.clock().strftime("%Y-%m-%d %H:%M"), self.reminders_db)

    def delivered(self, rid: int) -> None:
        mark_reminder_delivered(rid, self.reminders_db)

    def delete_user(self, chat_id: str | int) -> int:
        chat_id = str(chat_id)
        for d in (self.history, self.last_seen, self.last_topic):
            d.pop(chat_id, None)
        return self.store.delete(chat_id) + delete_reminders(chat_id, self.reminders_db)

    def stats(self) -> dict:
        return self.store.stats(self.clock().strftime("%Y-%m-%d"))
