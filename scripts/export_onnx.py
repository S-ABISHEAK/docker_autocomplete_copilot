"""Export the checkpoint to ONNX for lower-overhead CPU inference than eager PyTorch.

Exports only the forward pass (idx -> logits); the token-by-token sampling loop lives in
backend/inference/model_service.py and calls this graph once per generated token, matching
the uncached full-recompute behavior of ModernTransformer.generate().
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.architecture import ModernTransformer

CHECKPOINT = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "dockerfile-lm-v1.pt"
OUTPUT_ONNX = Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "dockerfile-lm-v1.onnx"


class LogitsOnly(nn.Module):
    def __init__(self, model: ModernTransformer):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits


def main() -> None:
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    model = ModernTransformer(device="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    export_model = LogitsOnly(model)
    dummy_input = torch.zeros((1, 8), dtype=torch.long)

    torch.onnx.export(
        export_model,
        (dummy_input,),
        str(OUTPUT_ONNX),
        input_names=["idx"],
        output_names=["logits"],
        dynamic_axes={"idx": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=18,
        dynamo=False,
    )

    size_mb = OUTPUT_ONNX.stat().st_size / (1024 * 1024)
    print(f"wrote {OUTPUT_ONNX} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
