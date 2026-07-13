"""ONNXRuntime-backed encoder inference — no PyTorch import required at runtime.

Drop-in replacement for TFTRegimeEncoder.encode() with ~3-5× lower latency on CPU
(no Python→C++ dispatch overhead, no autograd graph construction, single ONNX op fusion).

Usage:
    from encoder.encoder_onnx import OnnxEncoder
    enc = OnnxEncoder()                # auto-loads encoder/checkpoints/encoder.onnx
    emb = enc.encode(window_np)        # (60, 14) float32 array → (128,) float32 array
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DEFAULT_ONNX = Path("encoder/checkpoints/encoder.onnx")
_WINDOW_SIZE  = 60
_INPUT_DIM    = 14
_EMB_DIM      = 128


class OnnxEncoder:
    """Wraps an ONNX encoder session for single-window synchronous inference.

    Parameters
    ----------
    onnx_path:
        Path to the exported .onnx file.  Defaults to ``encoder/checkpoints/encoder.onnx``.
    """

    def __init__(self, onnx_path: Path = _DEFAULT_ONNX) -> None:
        import onnxruntime as ort

        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                "Run `python encoder/export_onnx.py` first."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # single-threaded for lowest per-call latency
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def encode(self, window: np.ndarray) -> np.ndarray:
        """Encode a single (T, 14) window → (128,) embedding.

        Parameters
        ----------
        window:
            Float32 array of shape ``(T, 14)`` where T ≤ _WINDOW_SIZE.
            Shorter windows are zero-padded on the left.
        """
        if window.ndim != 2 or window.shape[1] != _INPUT_DIM:
            raise ValueError(f"Expected (T, {_INPUT_DIM}) array, got {window.shape}")

        # Pad to WINDOW_SIZE on the left if needed
        t = window.shape[0]
        if t < _WINDOW_SIZE:
            pad = np.zeros((_WINDOW_SIZE - t, _INPUT_DIM), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        elif t > _WINDOW_SIZE:
            window = window[-_WINDOW_SIZE:]

        x = window.astype(np.float32)[np.newaxis]  # (1, 60, 14)
        out = self._session.run(None, {self._input_name: x})[0]  # (1, 128)
        return out[0]  # (128,)

    @property
    def is_available(self) -> bool:
        return self._session is not None


def load_onnx_encoder(onnx_path: Path = _DEFAULT_ONNX) -> OnnxEncoder | None:
    """Return an OnnxEncoder or None if onnxruntime / the model file is absent."""
    try:
        return OnnxEncoder(onnx_path)
    except (FileNotFoundError, ImportError):
        return None
