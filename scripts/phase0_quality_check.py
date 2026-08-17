"""Quality gate: exercises the exact production path (ModelService: ONNX model,
greedy decoding, EOT/blank-line stop conditions) against realistic Dockerfile prefixes,
for manual review of completion quality and CPU latency.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import load_settings
from backend.inference.model_service import ModelService
from model.tokenizer import encode

PROMPTS = [
    "FROM python:3.11-slim\n\nWORKDIR /app\nCOPY requirements.txt ",
    "FROM node:20-alpine\n\nWORKDIR /usr/src/app\nCOPY package*.json ./\nRUN ",
    "FROM ubuntu:22.04\n\nRUN apt-",
    "FROM golang:1.22 AS builder\nWORKDIR /src\nCOPY . .\nRUN go build -o app .\n\nFROM ",
    "FROM python:3.11-slim\n\nENV PYTHONDONTWRITEBYTECODE=1\nENV PATH=",
    "FROM nginx:alpine\n\nCOPY ./dist /usr/share/nginx/html\nEXPOSE ",
    "FROM python:3.11-slim\n\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\n\nCMD [",
    "FROM postgres:16\n\nENV POSTGRES_USER=admin\nENV POSTGRES_PASSWORD=",
    "FROM ",
    "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 8000\nENTRYPOINT [",
    "FROM alpine:3.19\n\nRUN addgroup -S app && adduser -S app -G app\nUSER ",
    "FROM python:3.11-slim as base\n\nFROM base as dev\nRUN pip install pytest\n\nFROM base as prod\nCOPY . .\nCMD ",
    "FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build\nWORKDIR /src\nCOPY *.csproj .\nRUN dotnet restore\nCOPY . .\nRUN dotnet publish -c Release -o /app\n\nFROM mcr.microsoft.com/dotnet/aspnet:8.0\nWORKDIR /app\nCOPY --from=build ",
    "FROM python:3.11-slim\n\nLABEL maintainer=",
    "FROM redis:7-alpine\n\nHEALTHCHECK --interval=30s --timeout=3s ",
]


def main() -> None:
    settings = load_settings()
    service = ModelService(
        onnx_path=settings.onnx_path,
        block_size=settings.block_size,
        max_new_tokens=settings.max_new_tokens,
        eot_token_id=settings.eot_token_id,
    )

    total_time = 0.0
    total_tokens_generated = 0

    for i, prompt in enumerate(PROMPTS):
        start = time.perf_counter()
        completion = service.generate(encode(prompt))
        elapsed = time.perf_counter() - start

        tokens_generated = len(encode(completion))
        total_time += elapsed
        total_tokens_generated += tokens_generated

        print("=" * 70)
        print(f"[{i}] prompt: {prompt!r}")
        print("-" * 70)
        print(prompt + completion)
        print(f"({elapsed * 1000:.0f}ms for {tokens_generated} tokens)")

    print("=" * 70)
    print(f"TOTAL: {total_tokens_generated} tokens in {total_time:.2f}s "
          f"-> avg {total_time / total_tokens_generated * 1000:.1f} ms/token on CPU (ONNX)")


if __name__ == "__main__":
    main()
