"""One-time conversion: drop Adam optimizer state from a training checkpoint, keeping
only the model weights needed for inference. checkpoint_07.pt is ~971MB because it stores
optimizer momentum/variance buffers alongside the ~320MB of weights; nothing after this
point in the project needs the optimizer state.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture import ModernTransformer

SOURCE_CHECKPOINT = Path(
    "/home/abisheak/PROJECTS/LLM_FROM_SCRATCH/PIPELINE/docker_model/checkpoint_07.pt"
)
OUTPUT_CHECKPOINT = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "dockerfile-lm-v1.pt"


def main() -> None:
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"source checkpoint not found: {SOURCE_CHECKPOINT}")

    print(f"loading {SOURCE_CHECKPOINT} ...")
    ckpt = torch.load(SOURCE_CHECKPOINT, map_location="cpu")
    state_dict = ckpt["model_state_dict"]
    global_step = ckpt.get("global_step")
    round_idx = ckpt.get("round_idx")
    print(f"source checkpoint: global_step={global_step}, round_idx={round_idx}")

    print("verifying state dict loads into current architecture (strict=True) ...")
    model = ModernTransformer(device="cpu")
    model.load_state_dict(state_dict, strict=True)
    print("OK: architecture matches checkpoint.")

    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state_dict}, OUTPUT_CHECKPOINT)

    size_mb = OUTPUT_CHECKPOINT.stat().st_size / (1024 * 1024)
    print(f"wrote {OUTPUT_CHECKPOINT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
