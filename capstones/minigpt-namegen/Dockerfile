# --- MiniGPT Name Generator: production image ---
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Train the tiny model at build time so the container starts instantly.
RUN python -m minigpt.train

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --timeout 120"]
