FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY backend ./backend
COPY models ./models
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
# bind 0.0.0.0 and honour $PORT (Render/Railway/Fly inject it)
CMD ["sh", "-c", "uvicorn app:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
