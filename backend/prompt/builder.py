from model.tokenizer import encode


def build_prompt(prefix: str, context_window_tokens: int) -> list[int]:
    normalized = prefix.replace("\r\n", "\n")
    tokens = encode(normalized)
    return tokens[-context_window_tokens:]
