# Vidya Saathi on Telegram: setup and launch

The bot is the same Vidya Saathi agent (tools, syllabus search, SGPA/CGPA, reminders) behind Telegram instead of a
web page. Students don't install anything or bring an API key.

What a student can do:
- type a question in English, Hindi or Kannada
- send a voice note (answer comes back as text plus a short voice reply)
- send a photo of a past-paper question (the bot reads it, then explains it)
- tap "Quiz me" after an explanation to get 3 Telegram quiz polls
- "remind me to revise DBMS tomorrow at 7 pm" (the bot messages them at that time)
- /language, /voice, /reminders, /privacy, /deletemydata

## 1. Create the bot (5 minutes, on your phone)

1. In Telegram, open @BotFather (blue tick) and send `/newbot`.
2. Name: `Vidya Saathi`. Username: must end in `bot`, e.g. `VidyaSaathiVTUbot`.
3. BotFather replies with a token like `123456:ABC...`. That token controls the bot: never post it or commit it.
4. Optional polish in BotFather: `/setdescription` (shown before Start), `/setabouttext`, `/setuserpic`.
5. Leave group privacy ON (the default). In groups the bot only sees messages that @mention it or reply to it.

## 2. Run it

Needs: `VIDYA_BOT_TOKEN`, `GROQ_API_KEY`. Recommended: `NEBIUS_API_KEY` (photos + backup answers), `NVIDIA_API_KEY`.

Local (Windows, for testing):
```
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements-bot.txt
copy bot.env.example bot.env   (fill it in)
set VOICE_AGENT_ENV_FILES=bot.env
.venv\Scripts\python -m app.telegram_bot
```
ffmpeg must be on PATH for voice replies (text answers work without it).

Server (any Linux box with Docker):
```
docker build -f Dockerfile.bot -t vidya-bot .
docker run -d --name vidya --restart unless-stopped --env-file bot.env -v vidya-data:/data vidya-bot
docker logs -f vidya
```
The bot uses long polling: it only makes outgoing connections, so no domain, HTTPS or open port is needed.
Run ONE copy per token (two copies fight over updates).

Where to host (checked 15 Nov 2026):
- Oracle Cloud Always Free (Ampere A1 VM): free, always on. Needs a card at signup (identity check).
- Any small Linux VPS with Docker works the same way.
- Not Render free: free web services spin down after 15 minutes with no inbound traffic (render.com/docs/free), and
  a polling bot gets no inbound traffic; its local SQLite file is also wiped on restart.
- Fly.io: no free plan for new accounts any more.

## 3. Limits and cost

Free Groq quota (console.groq.com/docs/rate-limits, 15 Nov 2026), per model: 1,000 requests and 200,000 tokens a
day, 8,000 tokens a minute. Measured on gpt-oss-120b (15 Nov): an explanation is one call of ~1,700 tokens, an
exam-date lookup is two calls of ~3,000 tokens together. So one Groq model covers roughly 60-115 answers a day, and
only ~4 answers a minute before it hits the per-minute limit. That's why the bot fails over
(Groq gpt-oss-120b -> Groq gpt-oss-20b, which has its own quota -> Nebius -> NVIDIA) and has daily caps:
30 questions per student, 15 voice notes, 8 photos, 6 quizzes, and 600 messages for everyone.
Change them in bot.env. Nebius/NVIDIA calls are billed to those accounts (credits) once Groq runs out.

## 4. Privacy (what students are told at /start)

Stored: Telegram chat id, settings, reminders, usage counts (message kind, answer time, 👍/👎, which invite
link they came from). Not stored: question text, answers, voice notes, photos. Those go to Groq/Nebius/NVIDIA only
to produce the answer. `/deletemydata` erases everything for that chat.

## 5. Measuring real users

- Send `/stats` from an id listed in `BOT_ADMIN_IDS`: students started, active today / 7 days, returning users,
  messages by kind, failures, 👍/👎, median answer time, and users per invite source.
- Give each group its own start link so you know what worked: `https://t.me/<botname>?start=rvitm-cse7a`,
  `?start=vtu-updates`, `?start=insta-reel`. The label is saved once, at first start.

## 6. Launch plan (first 100 students)

1. Week 1: your own class and friends (`?start=rvitm-<section>`). Fix whatever they hit.
2. Class reps / placement coordinators: a 20-second screen recording of a voice note in Kannada and a photo of a
   past-paper question, posted in their WhatsApp group with the start link.
3. VTU Telegram channels (e.g. @VTU_Updates, ~16K subscribers): ask the admin for one post, with its own link.
4. Instagram/YouTube Shorts: the same recording.

## 7. Before you tell anyone the numbers

- Exam dates come only from the VTU timetable data in `app/data`. Update it when VTU publishes a new timetable.
- Concept answers are the model's own explanation, not the official syllabus. The bot says so.
- Run `python scripts/bot_live_smoke.py <scratch-dir>` after any change: text, exam-date abstain, reminder, SGPA,
  Hindi and Kannada voice notes, photo, a prompt-injection attempt, and a quiz, all against the live providers.
