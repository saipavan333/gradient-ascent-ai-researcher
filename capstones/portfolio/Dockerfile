FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# train the tiny name-generator at build so the container starts instantly
RUN python -c "import app"
ENV PORT=8000
EXPOSE 8000
# one worker: the gridworld agent holds state in memory
CMD ["sh","-c","gunicorn app:app --bind 0.0.0.0:${PORT} --timeout 120 --workers 1"]
