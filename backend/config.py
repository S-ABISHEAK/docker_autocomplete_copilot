import os
import sys
from dataclasses import dataclass
from pathlib import Path

if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ONNX_PATH = PROJECT_ROOT / "model" / "checkpoints" / "dockerfile-lm-v1.onnx"


@dataclass(frozen=True)
class Settings:
    onnx_path: Path
    block_size: int
    context_window_tokens: int
    max_new_tokens: int
    eot_token_id: int
    request_timeout_seconds: float


def load_settings() -> Settings:
    return Settings(
        onnx_path=Path(os.environ.get("MODEL_ONNX_PATH", DEFAULT_ONNX_PATH)),
        block_size=int(os.environ.get("BLOCK_SIZE", 256)),
        # The model has no KV cache (recomputes full attention over the whole window every
        # generated token), so latency scales worse than linearly with this value: measured
        # ~550ms at 64 tokens vs ~1.7-3.9s at 200 tokens on the dev CPU. 64 trades context
        # length for responsiveness until KV caching is implemented.
        context_window_tokens=int(os.environ.get("CONTEXT_WINDOW_TOKENS", 64)),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", 20)),
        eot_token_id=int(os.environ.get("EOT_TOKEN_ID", 50256)),
        request_timeout_seconds=float(os.environ.get("REQUEST_TIMEOUT_SECONDS", 5.0)),
    )
