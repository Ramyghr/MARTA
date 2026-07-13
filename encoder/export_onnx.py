"""Export TFTRegimeEncoder to ONNX for sub-second CPU inference via ONNXRuntime.

Usage:
    python encoder/export_onnx.py                         # exports to encoder/checkpoints/encoder.onnx
    python encoder/export_onnx.py --out path/to/enc.onnx  # custom output path

The exported model takes a single float32 tensor (B, T, 14) and returns the 128-dim
embedding (B, 128), matching TFTRegimeEncoder.encode().

Requires: onnx, onnxruntime  (pip install onnx onnxruntime)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

_CKPT = Path("encoder/checkpoints/best_encoder.pt")
_OUT  = Path("encoder/checkpoints/encoder.onnx")
_INPUT_DIM   = 14
_HIDDEN_DIM  = 128
_WINDOW_SIZE = 60


def export(ckpt_path: Path = _CKPT, out_path: Path = _OUT) -> None:
    from encoder.model import TFTRegimeEncoder

    print(f"Loading encoder from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    enc = TFTRegimeEncoder(input_dim=_INPUT_DIM, hidden_dim=_HIDDEN_DIM)
    enc.load_state_dict(ckpt["model_state"])
    enc.eval()

    # Wrap encode() in a thin Module so torch.onnx exports the right forward path.
    class _EncodeWrapper(torch.nn.Module):
        def __init__(self, base: TFTRegimeEncoder) -> None:
            super().__init__()
            self.base = base

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base.encode(x)

    wrapper = _EncodeWrapper(enc)
    dummy = torch.zeros(1, _WINDOW_SIZE, _INPUT_DIM, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["window"],
        output_names=["embedding"],
        dynamic_axes={"window": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"Exported to {out_path}")

    # Quick validation with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"window": dummy.numpy()})[0]
        assert out.shape == (1, _HIDDEN_DIM), f"Unexpected shape: {out.shape}"
        print(f"ONNXRuntime validation passed — output shape {out.shape}")
    except ImportError:
        print("onnxruntime not installed — skipping validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(_CKPT))
    parser.add_argument("--out",  default=str(_OUT))
    args = parser.parse_args()
    export(Path(args.ckpt), Path(args.out))
