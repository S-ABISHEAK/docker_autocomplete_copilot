"""Tokenizer wrapper matching training exactly: stock tiktoken GPT-2 BPE, no custom vocab,
no FIM special tokens (see docker-model-scratch.ipynb cell 6)."""
import tiktoken

_enc = tiktoken.get_encoding("gpt2")


def encode(text: str) -> list[int]:
    return _enc.encode(text, disallowed_special=())


def decode(tokens: list[int]) -> str:
    return _enc.decode(tokens)
