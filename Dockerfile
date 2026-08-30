FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY backend/ backend/
COPY model/tokenizer.py model/__init__.py model/
COPY model/checkpoints/dockerfile-lm-v1.onnx model/checkpoints/dockerfile-lm-v1.onnx

ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache
RUN mkdir -p /app/.tiktoken_cache && \
    python -c "import tiktoken; tiktoken.get_encoding('gpt2')"

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn backend.api.app:app --host 0.0.0.0 --port ${PORT}"]
