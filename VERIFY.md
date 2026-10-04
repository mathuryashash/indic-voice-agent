# VERIFY: indic-voice-agent (branch `main`, local only, never pushed)

All numbers below were copied from command output or from `eval/results/*.json`, which are committed.
Machine: Windows 11, RTX 4060 Laptop 8 GB, Python 3.13 venv (`.venv`, uses system site-packages from C:/Python313).

## 1. Unit / integration tests
```
$ .venv/Scripts/python.exe -m pytest -q
......................................................                   [100%]
54 passed in 3.39s
```
What is covered:
- **tools (27):** exam lookup by alias/code/fuzzy name, BM25 top-1 for 6 queries plus the overview chunk, SQLite reminder
  round trip and bad input, VTU grade table, SGPA/CGPA formula and errors, dispatcher.
- **pipeline (19):** script-based language detection including code-mixed text, LID+script fusion, sentence and first-clause
  splitter, Unicode normalisation, the tool round trip with a mocked LLM, first TTS starting before the LLM finishes,
  Kannada voice and prompt, the reminder written to SQLite, cancellation.
- **LLM client (4):** SSE parsing, assembling tool-call deltas, 429 → immediate failover and 30 s circuit breaker,
  a mid-stream error failing over before any text, all providers failing.
- **WebSocket server (2):** full text turn over `/ws`, and barge-in (`cancel` while audio is streaming → `cancelled`, no `done`).

## 2. STT benchmark (faster-whisper on CUDA fp16, 42 edge-tts clips, 1 in 3 with white noise at 15 dB SNR)
```
$ .venv/Scripts/python.exe eval/stt_bench.py small large-v3-turbo large-v3 ; ... large-v3 large-v3-turbo --forced
```
| model | lang mode | hi WER | kn WER | kn CER | en WER | kn LID acc | STT p50 ms hi/kn/en |
|---|---|---|---|---|---|---|---|
| small | auto | 55.2% | 107.9% | 99.5% | 4.6% | 57.1% | 1043/1532/455 |
| large-v3-turbo | auto | 24.0% | 106.1% | 88.3% | 3.9% | 0.0% | 815/828/722 |
| **large-v3 (chosen)** | auto | 24.0% | 83.3% | 46.7% | 2.6% | 64.3% | 2114/2635/1165 |
| large-v3 | forced | 24.0% | 69.3% | 27.6% | 2.6% | 100% | 1732/2551/798 |

Finding: turbo labels every Kannada clip as Hindi (0% Kannada LID), so `large-v3` is the default.
Groq's hosted whisper-large-v3 also tagged Kannada clips as Telugu or Punjabi when no language was given (manual 3-clip probe).

## 3. End-to-end eval (audio → STT → LLM+tools → TTS), 42 utterances (14 hi / 14 kn / 14 en)
```
$ .venv/Scripts/python.exe eval/run_eval.py --mode audio --sleep 15   # auto language detect
$ .venv/Scripts/python.exe eval/run_eval.py --mode audio --pin        # language chosen in the UI
$ .venv/Scripts/python.exe eval/run_eval.py --mode text               # reference transcript (no STT)
$ .venv/Scripts/python.exe eval/report.py
```
| run | LLM actually used | detected-lang acc | tool acc | answer acc | reply in user's language | E2E→first audio p50 / p95 |
|---|---|---|---|---|---|---|
| audio, auto-detect | NVIDIA NIM gpt-oss-20b (all 42) | 88.1% | 92.9% | 76.2% | 85.7% | 8544 / 12330 ms |
| audio, language pinned | NIM gpt-oss-20b (41) + Groq gpt-oss-120b (1) | 100% | 95.2% | 81.0% | 95.2% | 8032 / 14006 ms |
| text (no STT) | NVIDIA NIM gpt-oss-20b | 97.6% | 95.2% | 88.1% | 85.7% | 6342 / 10161 ms |

Per-language results for the auto-detect audio run:
- **hi:** tool accuracy 92.9%, answer accuracy 78.6%.
- **kn:** tool accuracy 92.9%, answer accuracy 57.1%.
- **en:** tool accuracy 92.9%, answer accuracy 92.9%.

Per-stage latency for the auto-detect audio run (all 42), p50 / p95:
- STT: 1949 / 4028 ms
- LLM time to first token (including the tool round trip): 3617 / 5706 ms
- edge-tts first audio byte: 1407 / 2544 ms
- end to end, until the first clip is ready: 8544 / 12330 ms

Why the LLM column differs between runs: Groq gpt-oss-120b hit its free-tier **200k tokens/day** limit during the runs
(I confirmed it by hand: HTTP 429 "tokens per day (TPD): Limit 200000, Used 199050"). The client then failed over to NVIDIA NIM,
as designed. Earlier in the session, Groq rate limits cost 21 s of backoff per turn. That led to the change to immediate
failover plus a 30 s circuit breaker, covered by `test_429_fails_over_immediately_and_cools_down`.
The text run's 1 error (kn10) was a NIM mid-stream HTTP 500. It led to the mid-stream failover fix
(`test_midstream_error_before_any_text_still_fails_over`). The text eval was not re-run after that fix.

## 4. Browser E2E: Playwright + Chromium fake microphone, real server, real models
```
$ .venv/Scripts/python.exe -m uvicorn app.server:app --port 8765     (health: {"ok":true,"stt":"large-v3","device":"cuda","providers":["groq","nvidia"]})
$ .venv/Scripts/python.exe scripts/e2e_browser.py
E2E PASS
```
| lang | WAV fed as mic | transcript (Whisper) | detected | tool called | clips received/played (errors) | audio played |
|---|---|---|---|---|---|---|
| hi | hi01 | मेरा कम्प्यूटर नेटवर्क्स का एकजाम कब है? | hi | get_exam_schedule | 2/2 (0) | 13.13 s |
| kn | kn10 | ನಾನೆ ಸಂಜೆ ಆರು ಗಂಟಗೆ ಕಮ್ಪ್ಯೂಟರ್ ನೇಟ್ವರ್ಕ್ಸ್ ಓದಲು ನಿನಪಿಸು. | kn | set_reminder (2026-10-05 18:00) | 1/1 (0) | 5.98 s |
| en | en06 | Which module of computer networks covers TCP congestion control? | en | search_syllabus | 1/1 (0) | 4.06 s |

Screenshots: `docs/screenshots/e2e_{hi,kn,en}.png`. Each one shows the transcript, the tool call and its result, the reply,
the per-stage timing panel and the language-ID probabilities. Raw output is in `eval/results/e2e_browser.json`.
Demo reply audio: `docs/demo_reply_kn.mp3`, 10.25 s by ffprobe, from a full audio→audio turn on kn01. Reply:
"ಕಂಪ್ಯೂಟರ್ ನೆಟ್ವರ್ಕ್ಸ್ ಪರೀಕ್ಷೆ 14 ಡಿಸೆಂಬರ್ ಸೋಮವಾರ, 09:30‑12:30, ಬ್ಲಾಕ್ A, ರೂಮ್ 105."

## 5. Hygiene
- `git grep -E "gsk_…|nvapi-…"` finds nothing (exit 1). `.env`, `*.sqlite`, `eval/audio/` and `.venv/` are gitignored
  (checked with `git check-ignore`). Keys are read by `app/config.py` and only key *names* are ever returned.
- The uvicorn server was killed after E2E. `netstat` shows nothing on :8765.

## NOT done / limitations
- Kannada STT is weak: 83.3% WER with auto-detect, 69.3% WER / 27.6% CER with the language pinned. That is the main
  reason Kannada answer accuracy is 57.1%. No Indic-specific ASR (for example IndicConformer) was tried.
- With auto-detect, 5 of 14 Kannada clips are routed to Hindi or English, so the reply comes back in the wrong language.
  Pinning the language fixes the routing.
- The eval audio is synthetic (edge-tts plus white noise). Real student speech was not tested.
- End-to-end time to first audio is about 8 s at p50, not real-time. Most of it is two LLM round trips on free-tier
  cloud APIs plus large-v3 STT time. "First audio" means the first clip is fully synthesized; TTS bytes are not streamed
  into the browser.
- Barge-in works through the Stop button or a new utterance (server-side task cancellation is tested). There is no
  full-duplex VAD.
- Pipecat was not used; the pipeline is plain FastAPI + WebSocket. No Docker image, no deployment.
- The exam dates and syllabus notes are demo data written for this project, not official VTU data.
