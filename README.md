# VTU Saathi: a Hindi / Kannada / English voice study assistant

A local voice agent for VTU engineering students. You speak a question in Hindi, Kannada or English
(code-mixed is fine). The agent transcribes it, works out the language, calls real local tools
(exam timetable, syllabus search, reminders, SGPA/CGPA calculator) and speaks the answer back in
the same language. Tokens are streamed, and speech synthesis starts on the first sentence or clause.

```
mic (browser) ──webm──▶ faster-whisper large-v3 (RTX 4060, fp16)
                         └─ language ID restricted to hi/kn/en + script check
               ──text──▶ Groq gpt-oss-120b (streaming, OpenAI tool calling)
                         ├─ get_exam_schedule / search_syllabus (BM25) / set_reminder (SQLite)
                         ├─ calculate_sgpa / calculate_cgpa (VTU credit formula)
                         └─ falls back to NVIDIA NIM gpt-oss-20b on 429/5xx (30 s circuit breaker)
               ──first sentence/clause──▶ edge-tts neural voice (hi/kn/en-IN) ──mp3──▶ browser plays it
```

## Architecture

```mermaid
flowchart LR
  subgraph Browser["Browser UI (app/static/index.html)"]
    MIC[MediaRecorder mic] -->|base64 webm over WS| WS
    WS[WebSocket client] --> PLAY[Ordered mp3 playback queue]
    WS --> PANEL[Transcript, tool calls, per-stage timings]
    STOP[Stop / new utterance] -->|cancel = barge-in| WS
  end
  subgraph Server["FastAPI server (app/server.py)"]
    WSS[/ws endpoint/] --> TURN[run_turn pipeline]
    TURN --> STT[faster-whisper large-v3<br/>CUDA fp16, CPU int8 fallback]
    STT --> LID[hi/kn/en LID + script check]
    LID --> LLM[Streaming chat client<br/>Groq gpt-oss-120b → NVIDIA NIM]
    LLM -->|tool_calls| TOOLS[(Tools: exams.json, BM25 syllabus,<br/>SQLite reminders, SGPA/CGPA)]
    TOOLS -->|results| LLM
    LLM -->|token stream| SPLIT[Sentence / first-clause splitter]
    SPLIT -->|parallel per sentence| TTS[edge-tts neural voices]
    TTS -->|mp3 chunks, in order| WSS
  end
  WS <--> WSS
```

Why this stack (checked on this machine, Windows 11 + RTX 4060 8 GB):
- **STT: faster-whisper `large-v3` on CUDA.** The STT benchmark below compares `small`, `large-v3-turbo` and `large-v3`.
  Turbo and large-v3 score the same on Hindi, but turbo is much worse on Kannada (it sends almost every Kannada clip to Hindi),
  so the default is `large-v3`. CTranslate2 needed the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels, and their DLL
  folders get registered at import (`app/stt.py`). If CUDA fails, the code falls back to CPU int8.
- **LLM: Groq `openai/gpt-oss-120b`** with `reasoning_effort=low`. Groq retired the Llama-3.3 models (the `/models`
  listing on 2026-10-04 has only gpt-oss and qwen chat models), and NVIDIA NIM returned HTTP 410 for `meta/llama-3.3-70b-instruct`.
  The fallback is NVIDIA NIM `openai/gpt-oss-20b`.
- **TTS: edge-tts** with `hi-IN-SwaraNeural`, `kn-IN-SapnaNeural` and `en-IN-NeerjaNeural` (the Madhur, Gagan and Prabhat voices are used in the eval).
- **No Pipecat.** A plain FastAPI + WebSocket pipeline is easier to read and to time per stage on Windows. The brief allowed this.

## Setup

```bash
# Python 3.13 with CUDA torch at C:/Python313 (system-site-packages reuses faster-whisper, fastapi, etc.)
C:/Python313/python.exe -m venv --system-site-packages .venv
.venv/Scripts/python.exe -m pip install edge-tts jiwer python-multipart nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
export HF_HOME=D:/hf-cache          # whisper weights (~3 GB for large-v3) are downloaded to D:/hf-cache/whisper
# API keys: put GROQ_API_KEY / NVIDIA_API_KEY in ./.env (gitignored). app/config.py also reads them from the
# owner's other local .env files. Key values are never printed or logged.

.venv/Scripts/python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765, press the mic, ask e.g. "ಕಂಪ್ಯೂಟರ್ ನೆಟ್‌ವರ್ಕ್ಸ್ ಪರೀಕ್ಷೆ ಯಾವಾಗ?"
```

Environment knobs: `WHISPER_MODEL` (default `large-v3`), `WHISPER_DEVICE` (`cuda`/`cpu`), `GROQ_MODEL_VOICE`,
`NVIDIA_MODEL_VOICE`, `VOICE_REASONING_EFFORT`.

## Tests and eval

```bash
.venv/Scripts/python.exe -m pytest -q                       # unit + pipeline (mocked LLM/TTS) + WS server + LLM client
.venv/Scripts/python.exe eval/make_audio.py                 # 42 clips via edge-tts, 2 voices/lang, 1 in 3 with 15 dB noise
.venv/Scripts/python.exe eval/stt_bench.py small large-v3-turbo large-v3   # WER/CER/LID per language
.venv/Scripts/python.exe eval/stt_bench.py large-v3 large-v3-turbo --forced
.venv/Scripts/python.exe eval/run_eval.py --mode audio      # full voice pipeline, auto language detection
.venv/Scripts/python.exe eval/run_eval.py --mode audio --pin  # language chosen in the UI dropdown
.venv/Scripts/python.exe eval/run_eval.py --mode text       # reference transcript -> LLM (separates out STT errors)
.venv/Scripts/python.exe scripts/e2e_browser.py             # Playwright + Chromium fake mic, hi/kn/en (server must be running)
.venv/Scripts/python.exe eval/report.py                     # -> eval/results/RESULTS.md
```

Metrics:
- **WER/CER**: jiwer on NFC-normalised text, with punctuation and ZWJ/ZWNJ removed.
- **Tool accuracy**: the expected tool was called. For chit-chat cases, no tool was called.
- **Answer accuracy**: rule checks. The tool must resolve to the right course or exam, the reminder time must be correct,
  and the required facts (date, module number, SGPA value, ...) must appear in the reply.
- **Latency**: p50/p95 per stage. STT. LLM time-to-first-token, measured from LLM start and including the tool round trip.
  TTS first audio byte. End-to-end, measured from receiving the audio to the first synthesized audio being ready to send.

## Results

<!-- RESULTS:START -->
See [eval/results/RESULTS.md](eval/results/RESULTS.md) (generated from the JSON files and never edited by hand).
<!-- RESULTS:END -->

## Limitations (honest)

- **Kannada STT is the weak link.** Whisper large-v3 on synthetic Kannada speech still has high WER (see the table).
  Code-mixed technical words get transliterated phonetically, and with auto-detect some Kannada clips are labelled
  Hindi or English, so the reply comes back in the wrong language. Picking the language in the UI dropdown (`--pin` eval)
  fixes the language choice but not the WER. An Indic-specific ASR model (AI4Bharat IndicConformer, or a
  Kannada fine-tune of Whisper) would be the next step. It is not done here.
- **The eval audio is synthetic.** It is edge-tts speech plus white noise, not real students on laptop mics, so real-world
  WER will be worse. The TTS voices used for the test clips come from the same family as the reply voices.
- **Latency depends on free-tier cloud services.** Groq free tier rate limits (HTTP 429) are shared with other projects
  on the same key. When Groq is rate-limited the client fails over to NVIDIA NIM straight away and skips Groq for 30 s.
  edge-tts needs internet. End-to-end time to first audio is several seconds, not sub-second, mostly because of the
  two LLM round trips (tool decision, then answer) and large-v3 STT time.
- **Barge-in is button- or utterance-based, not VAD-based.** Pressing Stop or starting a new recording cancels the server
  task and stops local playback. The server-side cancel is covered by a test. The server does not listen to the mic while
  it is speaking (no full-duplex VAD).
- **The demo data is made up.** `app/data/exams.json` and the syllabus notes were written for this demo. They are not an
  official VTU timetable or the full syllabus.
- **Reminders are only stored in SQLite.** No notification is fired when a reminder comes due.
- **Not deployed and no Docker image.** The project runs locally only.
