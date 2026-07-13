"""ONNX encoder export and inference tests.

These tests exercise encoder/export_onnx.py and encoder/encoder_onnx.py without
requiring a real checkpoint — they export a randomly-initialised encoder to a
temp file and verify round-trip correctness against the PyTorch path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from encoder.model import TFTRegimeEncoder
from encoder.encoder_onnx import OnnxEncoder, load_onnx_encoder

_INPUT_DIM   = 14
_HIDDEN_DIM  = 128
_WINDOW_SIZE = 60


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory) -> Path:
    """Export a fresh random-weight encoder to a temp ONNX file."""
    import onnx as _onnx  # noqa: F401 — ensures onnx is importable

    tmp = tmp_path_factory.mktemp("onnx")
    onnx_file = tmp / "encoder_test.onnx"

    enc = TFTRegimeEncoder(input_dim=_INPUT_DIM, hidden_dim=_HIDDEN_DIM)
    enc.eval()

    class _Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            return self.base.encode(x)

    dummy = torch.zeros(1, _WINDOW_SIZE, _INPUT_DIM, dtype=torch.float32)
    torch.onnx.export(
        _Wrapper(enc),
        dummy,
        str(onnx_file),
        input_names=["window"],
        output_names=["embedding"],
        dynamic_axes={"window": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=17,
    )
    return onnx_file


@pytest.fixture(scope="module")
def torch_enc() -> TFTRegimeEncoder:
    enc = TFTRegimeEncoder(input_dim=_INPUT_DIM, hidden_dim=_HIDDEN_DIM)
    enc.eval()
    return enc


# ---------------------------------------------------------------------------
# OnnxEncoder contract
# ---------------------------------------------------------------------------

def test_onnx_encoder_loads(onnx_path):
    enc = OnnxEncoder(onnx_path)
    assert enc.is_available


def test_onnx_encoder_output_shape(onnx_path):
    enc = OnnxEncoder(onnx_path)
    window = np.random.randn(_WINDOW_SIZE, _INPUT_DIM).astype(np.float32)
    out = enc.encode(window)
    assert out.shape == (_HIDDEN_DIM,)
    assert out.dtype == np.float32


def test_onnx_encoder_short_window_padded(onnx_path):
    """Windows shorter than 60 should be zero-padded silently."""
    enc = OnnxEncoder(onnx_path)
    short = np.random.randn(30, _INPUT_DIM).astype(np.float32)
    out = enc.encode(short)
    assert out.shape == (_HIDDEN_DIM,)


def test_onnx_encoder_long_window_truncated(onnx_path):
    """Windows longer than 60 should be truncated from the left."""
    enc = OnnxEncoder(onnx_path)
    long = np.random.randn(90, _INPUT_DIM).astype(np.float32)
    out = enc.encode(long)
    assert out.shape == (_HIDDEN_DIM,)


def test_onnx_wrong_input_shape_raises(onnx_path):
    enc = OnnxEncoder(onnx_path)
    with pytest.raises(ValueError):
        enc.encode(np.zeros((_WINDOW_SIZE, 7), dtype=np.float32))  # wrong feature dim


# ---------------------------------------------------------------------------
# Numerical agreement with PyTorch
# ---------------------------------------------------------------------------

def test_onnx_vs_pytorch_close(onnx_path, torch_enc):
    """ONNX and PyTorch should produce numerically identical outputs
    for the same random-weight model exported to the same ONNX file.

    We re-export from torch_enc (same weights) for comparison.
    """
    import tempfile

    class _Wrap(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
        def forward(self, x):
            return self.b.encode(x)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = Path(f.name)

    dummy = torch.zeros(1, _WINDOW_SIZE, _INPUT_DIM)
    torch.onnx.export(
        _Wrap(torch_enc), dummy, str(tmp),
        input_names=["window"], output_names=["embedding"],
        dynamic_axes={"window": {0: "b"}, "embedding": {0: "b"}},
        opset_version=17,
    )

    ort_enc = OnnxEncoder(tmp)
    rng = np.random.default_rng(0)
    window_np = rng.standard_normal((_WINDOW_SIZE, _INPUT_DIM)).astype(np.float32)

    ort_out = ort_enc.encode(window_np)

    with torch.no_grad():
        x = torch.from_numpy(window_np).unsqueeze(0)
        pt_out = torch_enc.encode(x).squeeze(0).numpy()

    np.testing.assert_allclose(ort_out, pt_out, atol=1e-5,
                               err_msg="ORT and PyTorch outputs diverge")
    tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load_onnx_encoder helper
# ---------------------------------------------------------------------------

def test_load_onnx_encoder_returns_none_when_missing():
    result = load_onnx_encoder(Path("/nonexistent/encoder.onnx"))
    assert result is None


def test_load_onnx_encoder_returns_instance(onnx_path):
    result = load_onnx_encoder(onnx_path)
    assert isinstance(result, OnnxEncoder)
