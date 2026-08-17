"""Entry point for the standalone bundled executable (PyInstaller). Sets the tiktoken
cache directory to the bundled copy before anything imports the tokenizer, so the
executable never needs network access at runtime.
"""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
else:
    bundle_root = Path(__file__).resolve().parent.parent

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(bundle_root / ".tiktoken_cache"))

sys.path.insert(0, str(bundle_root))

import uvicorn

from backend.api.app import app


def main() -> None:
    port = int(os.environ.get("PORT", 8123))
    print(f"Dockerfile Autocomplete server starting on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
