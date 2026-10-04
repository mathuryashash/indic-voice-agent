# Resume lines (every number is from VERIFY.md)

- Built a Hindi/Kannada/English voice agent (faster-whisper on CUDA → streaming LLM with tool calling → edge-tts). On 42 scripted utterances: 92.9% tool-selection accuracy, 8.5 s p50 time to first audio.
- Benchmarked Whisper small/turbo/large-v3 on Indic speech. Found turbo labels every Kannada clip as Hindi, and chose large-v3: 2.6% English and 24.0% Hindi WER. Added a 30 s LLM failover circuit breaker; 54 tests and Playwright fake-mic E2E pass.

Keywords proven: voice agents, speech-to-text (faster-whisper/CTranslate2, CUDA), text-to-speech, streaming LLM, OpenAI-style tool/function calling, multilingual NLP (Hindi, Kannada, code-mixed), language identification, BM25 retrieval, SQLite, FastAPI WebSockets, barge-in/cancellation, LLM provider failover and circuit breaker, evals (WER/CER with jiwer, tool accuracy, rule-based answer checks, p50/p95 latency), Playwright E2E, pytest.
