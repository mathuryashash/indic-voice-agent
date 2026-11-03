# Cloud image: speech recognition on Groq (no GPU, ~150 MB). Runs on Render / Cloud Run / HF Spaces.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STT_BACKEND=groq PORT=7860 REMINDERS_DB=/tmp/reminders.sqlite
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements-cloud.txt .
RUN pip install -r requirements-cloud.txt
COPY --chown=user app ./app
USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
