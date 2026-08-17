"""Confirms the ONNX graph matches PyTorch's greedy-decoding output and measures the
ms/token improvement, which sets Phase 1's max_new_tokens budget.
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture import ModernTransformer
from model.tokenizer import encode, decode

CHECKPOINT = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "dockerfile-lm-v1.pt"
ONNX_MODEL = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "dockerfile-lm-v1.onnx"
BLOCK_SIZE = 256
PROMPT = "FROM python:3.11-slim\n\nWORKDIR /app\nCOPY requirements.txt "
NUM_TOKENS = 30


def load_pytorch_model() -> ModernTransformer:
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model = ModernTransformer(device="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def greedy_decode_pytorch(model: ModernTransformer, tokens: list[int], n: int) -> list[int]:
    idx = torch.tensor([tokens], dtype=torch.long)
    for _ in range(n):
        idx_cond = idx[:, -BLOCK_SIZE:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        next_id = int(torch.argmax(logits[0, -1]))
        idx = torch.cat([idx, torch.tensor([[next_id]])], dim=1)
    return idx[0].tolist()


def greedy_decode_onnx(session: ort.InferenceSession, tokens: list[int], n: int) -> list[int]:
    ids = list(tokens)
    for _ in range(n):
        window = ids[-BLOCK_SIZE:]
        idx = np.array([window], dtype=np.int64)
        logits = session.run(["logits"], {"idx": idx})[0]
        next_id = int(np.argmax(logits[0, -1]))
        ids.append(next_id)
    return ids


def main() -> None:
    tokens = encode(PROMPT)

    pytorch_model = load_pytorch_model()
    start = time.perf_counter()
    pytorch_ids = greedy_decode_pytorch(pytorch_model, tokens, NUM_TOKENS)
    pytorch_time = time.perf_counter() - start

    session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    onnx_ids = greedy_decode_onnx(session, tokens, NUM_TOKENS)
    onnx_time = time.perf_counter() - start

    match = pytorch_ids == onnx_ids
    print(f"outputs match (greedy): {match}")
    if not match:
        print("pytorch:", decode(pytorch_ids))
        print("onnx:   ", decode(onnx_ids))

    print(f"pytorch: {pytorch_time:.2f}s total, {pytorch_time / NUM_TOKENS * 1000:.1f} ms/token")
    print(f"onnx:    {onnx_time:.2f}s total, {onnx_time / NUM_TOKENS * 1000:.1f} ms/token")
    print(f"speedup: {pytorch_time / onnx_time:.2f}x")


if __name__ == "__main__":
    main()
