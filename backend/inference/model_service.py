from pathlib import Path

import numpy as np
import onnxruntime as ort

from model.tokenizer import decode


class ModelService:
    def __init__(self, onnx_path: Path, block_size: int, max_new_tokens: int, eot_token_id: int):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._block_size = block_size
        self._max_new_tokens = max_new_tokens
        self._eot_token_id = eot_token_id

    def generate(self, prompt_tokens: list[int]) -> str:
        ids = list(prompt_tokens)
        prompt_len = len(ids)

        for _ in range(self._max_new_tokens):
            next_id = self._predict_next(ids[-self._block_size:])
            if next_id == self._eot_token_id:
                break
            ids.append(next_id)
            if self._ends_with_blank_line(ids[prompt_len:]):
                break

        return decode(ids[prompt_len:])

    def _predict_next(self, window: list[int]) -> int:
        idx = np.array([window], dtype=np.int64)
        logits = self._session.run(["logits"], {"idx": idx})[0]
        return int(np.argmax(logits[0, -1]))

    @staticmethod
    def _ends_with_blank_line(generated_tokens: list[int]) -> bool:
        if len(generated_tokens) < 2:
            return False
        return decode(generated_tokens).endswith("\n\n")
