"""Tests for NanoGEMM bare-metal SIMD acceleration and NumPy fallback parity."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from yfinance_ta_patterns import (
    HAS_NANOGEMM,
    AIPatternScorer,
    cosine_similarity_matrix,
    fast_matmul,
    fast_score_batch,
    get_gemm_backend,
)
from yfinance_ta_patterns import gemm as gemm_mod


def test_gemm_backend_detection():
    backend = get_gemm_backend()
    assert isinstance(backend, str)
    if HAS_NANOGEMM:
        assert "NanoGEMM" in backend
    else:
        assert "NumPy" in backend or "Pure-Python" in backend


@pytest.mark.parametrize(
    "m, k, n",
    [
        (1, 16, 1),
        (1, 16, 4),
        (4, 16, 2),
        (10, 16, 5),
        (25, 16, 16),
        (50, 8, 4),
    ],
)
def test_fast_matmul_parity_with_numpy(m: int, k: int, n: int):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    # 1. Native / active backend (NanoGEMM if installed)
    res_active = fast_matmul(a, b)
    expected = np.matmul(a, b)
    assert np.allclose(res_active, expected, atol=1e-4)

    # 2. In-place buffer test
    out_buf = np.zeros((m, n), dtype=np.float32)
    fast_matmul(a, b, out=out_buf)
    assert np.allclose(out_buf, expected, atol=1e-4)

    # 3. Forced NumPy fallback
    with patch.object(gemm_mod, "HAS_NANOGEMM", False):
        res_numpy = fast_matmul(a, b)
        assert np.allclose(res_numpy, expected, atol=1e-5)


def test_fast_matmul_pure_python_fallback():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    expected = [[19.0, 22.0], [43.0, 50.0]]

    with patch.object(gemm_mod, "HAS_NANOGEMM", False), patch.object(gemm_mod, "HAS_NUMPY", False):
        res = fast_matmul(a, b)
        assert res == expected


def test_fast_matmul_dimension_mismatch():
    a = np.ones((4, 8), dtype=np.float32)
    b = np.ones((6, 2), dtype=np.float32)  # 8 != 6
    with pytest.raises(ValueError, match="Incompatible"):
        fast_matmul(a, b)


def test_fast_matmul_invalid_ndim():
    a = np.ones(8, dtype=np.float32)  # 1D
    b = np.ones((8, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="2D matrices"):
        fast_matmul(a, b)


@pytest.mark.parametrize("activation", ["sigmoid", "linear", "relu"])
def test_fast_score_batch_activations(activation: str):
    rng = np.random.default_rng(123)
    n, k = 20, 16
    features = rng.standard_normal((n, k)).astype(np.float32)
    weights = rng.standard_normal((k,)).astype(np.float32)
    bias = 0.25

    # Active engine (NanoGEMM if present)
    scores = fast_score_batch(features, weights, bias=bias, activation=activation)
    assert len(scores) == n

    # Forced NumPy fallback
    with patch.object(gemm_mod, "HAS_NANOGEMM", False):
        scores_np = fast_score_batch(features, weights, bias=bias, activation=activation)

    assert np.allclose(scores, scores_np, atol=1e-4)

    # Validate mathematical correctness
    raw_logits = np.dot(features, weights) + bias
    if activation == "sigmoid":
        expected = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -30.0, 30.0)))
        assert (scores >= 0.0).all() and (scores <= 1.0).all()
    elif activation == "relu":
        expected = np.maximum(0.0, raw_logits)
    else:
        expected = raw_logits
    assert np.allclose(scores, expected, atol=1e-4)


def test_fast_score_batch_array_bias_and_multi_weights():
    features = np.ones((4, 3), dtype=np.float32)
    weights = np.ones((3, 2), dtype=np.float32)
    bias = np.array([0.5, 1.0], dtype=np.float32)

    scores = fast_score_batch(features, weights, bias=bias, activation="linear")
    assert scores.shape == (4, 2)
    # 1*1*3 + [0.5, 1.0] = [3.5, 4.0]
    expected = np.tile(np.array([3.5, 4.0], dtype=np.float32), (4, 1))
    assert np.allclose(scores, expected)


def test_fast_score_batch_pure_python_fallback():
    features = [[1.0, -1.0], [0.5, 0.5]]
    weights = [2.0, 1.0]
    bias = 0.5

    with patch.object(gemm_mod, "HAS_NANOGEMM", False), patch.object(gemm_mod, "HAS_NUMPY", False):
        scores_sig = fast_score_batch(features, weights, bias=bias, activation="sigmoid")
        scores_lin = fast_score_batch(features, weights, bias=bias, activation="linear")
        scores_relu = fast_score_batch(features, weights, bias=bias, activation="relu")

    # Row 0: 1*2 + (-1)*1 + 0.5 = 1.5
    # Row 1: 0.5*2 + 0.5*1 + 0.5 = 2.0
    assert math.isclose(scores_lin[0], 1.5, rel_tol=1e-5)
    assert math.isclose(scores_lin[1], 2.0, rel_tol=1e-5)
    assert math.isclose(scores_sig[0], 1.0 / (1.0 + math.exp(-1.5)), rel_tol=1e-5)
    assert math.isclose(scores_relu[0], 1.5, rel_tol=1e-5)


def test_cosine_similarity_matrix():
    # Orthogonal vectors and identical vectors
    q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    sim = cosine_similarity_matrix(q, p)
    assert sim.shape == (2, 3)

    # q[0] == p[0] -> sim 1.0
    assert math.isclose(sim[0, 0], 1.0, abs_tol=1e-5)
    # q[0] perp p[1] -> sim 0.0
    assert math.isclose(sim[0, 1], 0.0, abs_tol=1e-5)
    # q[1] == p[1] -> sim 1.0
    assert math.isclose(sim[1, 1], 1.0, abs_tol=1e-5)


def test_ai_pattern_scorer_gemm_integration():
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]
    df = {
        "Open": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
        "High": [11.0, 11.5, 12.0, 12.5, 13.0, 13.5],
        "Low": [9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
        "Close": [10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
        "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
    }
    pd = pytest.importorskip("pandas")

    data = pd.DataFrame(df, index=pd.to_datetime(dates))

    scorer = AIPatternScorer(data)
    backend_name = scorer.gemm_backend
    assert isinstance(backend_name, str)

    # Test score_batch_matrix method
    feat = np.ones((5, 4), dtype=np.float32)
    w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    scores = scorer.score_batch_matrix(feat, w, bias=0.0, activation="sigmoid")
    assert len(scores) == 5
    # sum = 1.0 -> sigmoid(1.0) approx 0.731058
    assert math.isclose(float(scores[0]), 1.0 / (1.0 + math.exp(-1.0)), abs_tol=1e-4)
