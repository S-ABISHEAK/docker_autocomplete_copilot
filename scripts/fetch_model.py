"""Pulls the released ONNX model from Hugging Face Hub into model/checkpoints/, so a
Docker build always bakes in a specific published version rather than whatever a local
export happens to be sitting on. Run this before `docker build` for a release.
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "S-ABISHEAK/dockerfile-lm"
DEFAULT_FILENAME = "dockerfile-lm-v1.onnx"
DEST_PATH = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "dockerfile-lm-v1.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--revision", default=None, help="Git revision/tag/commit on the HF repo (default: latest)")
    args = parser.parse_args()

    print(f"fetching {args.filename} from {args.repo_id} (revision={args.revision or 'latest'}) ...")
    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
    )

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEST_PATH.write_bytes(Path(downloaded_path).read_bytes())
    print(f"wrote {DEST_PATH} ({DEST_PATH.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
