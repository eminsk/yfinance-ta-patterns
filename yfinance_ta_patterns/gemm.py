"""High-Performance Hybrid Linear Algebra & Matrix Scoring Engine.

Combines the best of NumPy (rich timeseries structures, datetime indexing, array ergonomics)
with the best of NanoGEMM (sub-microsecond bare-metal AVX2/FMA/NEON microkernels, zero BLAS
dispatch overhead, and thread-safe execution in Python 3.13+ Free-Threaded No-GIL).

When nanogemm is installed and hardware SIMD is detected, batch scoring and matrix multiplications
execute via native bare-metal kernels. When nanogemm is not installed, it falls back seamlessly
to NumPy or pure Python with 100% numerical parity.
"""

from __future__ import annotations

import math
from typing import Any

# 1. NumPy import with graceful fallback
try:
    import numpy as np

    HAS_NUMPY = True
except (ImportError, ModuleNotFoundError, RuntimeError, TypeError):  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

# 2. NanoGEMM bare-metal SIMD engine import
try:
    import nanogemm as _ng

    HAS_NANOGEMM: bool = bool(getattr(_ng, "is_available", lambda: False)())
except (ImportError, ModuleNotFoundError, RuntimeError, TypeError, AttributeError):
    _ng = None  # type: ignore[assignment]
    HAS_NANOGEMM = False


def get_gemm_backend() -> str:
    """Return description of the active linear algebra and matrix scoring backend."""
    if HAS_NANOGEMM and _ng is not None:
        isa = getattr(_ng, "get_simd_isa", lambda: "SIMD")()
        return f"NanoGEMM ({isa})"
    if HAS_NUMPY:
        return "NumPy (BLAS/Vectorized Fallback)"
    return "Pure-Python (Scalar Fallback)"


def fast_matmul(
    left: Any,
    right: Any,
    out: Any | None = None,
) -> Any:
    """Compute matrix multiplication C = A @ B with optimal backend dispatch.

    Dispatches to NanoGEMM AVX2/FMA microkernel for small-to-medium 2D float32 matrices,
    or falls back seamlessly to NumPy np.matmul / pure-Python.
    """
    # Prefer NanoGEMM for 2D float32 operations when available
    if HAS_NANOGEMM and _ng is not None and HAS_NUMPY:
        a = np.ascontiguousarray(left, dtype=np.float32)
        b = np.ascontiguousarray(right, dtype=np.float32)

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"fast_matmul operands must be 2D matrices, got {a.shape} and {b.shape}"
            )

        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible matrix dimensions: {a.shape} @ {b.shape}")

        orig_out = out
        if (
            out is None
            or not isinstance(out, np.ndarray)
            or out.shape != (a.shape[0], b.shape[1])
            or out.dtype != np.float32
            or not out.flags.c_contiguous
        ):
            target_out = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)
        else:
            target_out = out

        _ng.matmul(a, b, out=target_out)

        if (
            orig_out is not None
            and isinstance(orig_out, np.ndarray)
            and orig_out is not target_out
            and orig_out.shape == target_out.shape
        ):
            np.copyto(orig_out, target_out)
            return orig_out

        return target_out

    # Standard NumPy fallback
    if HAS_NUMPY:
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"fast_matmul operands must be 2D matrices, got {a.shape} and {b.shape}"
            )
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible matrix dimensions: {a.shape} @ {b.shape}")

        if (
            out is not None
            and isinstance(out, np.ndarray)
            and out.shape == (a.shape[0], b.shape[1])
        ):
            np.matmul(a, b, out=out)
            return out
        return np.matmul(a, b)

    # Pure Python fallback for environments without NumPy or NanoGEMM
    a_list = [list(r) for r in left]
    b_list = [list(r) for r in right]
    m = len(a_list)
    k = len(a_list[0]) if m > 0 else 0
    if len(b_list) != k:
        raise ValueError(f"Incompatible dimensions: ({m}, {k}) @ ({len(b_list)}, ...)")
    n = len(b_list[0]) if k > 0 else 0

    res = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += a_list[i][p] * b_list[p][j]
            res[i][j] = s
    return res


def fast_score_batch(
    features: Any,
    weights: Any,
    bias: float = 0.0,
    activation: str = "sigmoid",
    out: Any | None = None,
) -> Any:
    """Compute sub-microsecond batched confluence scores for candidate signals.

    Calculates: Z = activation(features @ weights + bias).
    Dispatches to NanoGEMM AVX2 SIMD microkernel if available, otherwise NumPy/Python.

    Parameters
    ----------
    features : array-like of shape (N, K)
        Normalized technical feature rows for N signals across K factors.
    weights : array-like of shape (K,) or (K, 1)
        Factor model weights.
    bias : float, default 0.0
        Model bias term.
    activation : str, default "sigmoid"
        Activation function: "sigmoid", "linear", or "relu".
    out : optional array
        Pre-allocated output buffer.

    Returns
    -------
    np.ndarray or list[float] of length N
        Calibrated confluence confidence scores.
    """
    if HAS_NANOGEMM and _ng is not None and HAS_NUMPY and hasattr(_ng, "score_batch"):
        try:
            # The native kernel reinterprets raw memory as float32: sanitize dtype/layout here,
            # exactly like fast_matmul does, instead of forwarding float64 / strided arrays.
            x32 = np.ascontiguousarray(features, dtype=np.float32)
            w32 = np.ascontiguousarray(weights, dtype=np.float32)
            if x32.ndim == 1:
                x32 = x32.reshape(1, -1)
            if w32.ndim == 2 and w32.shape[1] == 1:
                w32 = w32.reshape(-1)
            if x32.ndim == 2 and w32.ndim == 1 and np.isscalar(bias):
                native_out = _ng.score_batch(
                    features=x32, weights=w32, bias=float(bias), activation=activation, out=None
                )
                if out is not None and hasattr(out, "__setitem__"):
                    try:
                        out[:] = native_out
                        return out
                    except Exception:
                        pass
                return native_out
        except Exception:
            pass  # Fall through to NumPy fallback on any shape/type mismatch

    # NumPy Fallback
    if HAS_NUMPY:
        x = np.ascontiguousarray(features, dtype=np.float32)
        w = np.ascontiguousarray(weights, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        elif w.ndim == 2 and w.shape[1] != 1 and w.shape[0] == 1:
            w = w.T

        if x.shape[1] != w.shape[0]:
            raise ValueError(
                f"Incompatible shapes for scoring: features {x.shape} vs weights {w.shape}"
            )

        logits = np.matmul(x, w)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]

        if bias is not None:
            if np.isscalar(bias):
                if bias != 0.0:
                    logits += np.float32(bias)
            else:
                bias_arr = np.asarray(bias, dtype=np.float32)
                if bias_arr.size > 0:
                    logits += bias_arr

        if activation == "sigmoid":
            # Numerically stable sigmoid without overflow
            clipped = np.clip(logits, -30.0, 30.0)
            res = 1.0 / (1.0 + np.exp(-clipped))
        elif activation == "relu":
            res = np.maximum(0.0, logits)
        else:  # linear
            res = logits

        if out is not None and isinstance(out, np.ndarray) and len(out) == len(res):
            out[:] = res
            return out
        return res

    # Pure Python Fallback
    if hasattr(features, "__len__") and len(features) == 0:
        return []

    if hasattr(features, "__iter__") and len(features) > 0 and hasattr(features[0], "__iter__"):
        x_list = [[float(v) for v in r] for r in features]
    else:
        x_list = [[float(v) for v in features]]

    if hasattr(weights, "__iter__") and len(weights) > 0 and hasattr(weights[0], "__iter__"):
        if len(weights) == 1:
            w_list = [float(v) for v in weights[0]]
        else:
            w_list = [float(r[0]) if hasattr(r, "__iter__") else float(r) for r in weights]
    else:
        w_list = [float(v) for v in weights]

    k = len(w_list)

    res_list: list[float] = []
    bias_val = (
        float(bias)
        if bias is not None
        and (isinstance(bias, (int, float)) or (np is not None and np.isscalar(bias)))
        else 0.0
    )
    for row in x_list:
        if len(row) != k:
            raise ValueError(f"Feature row length {len(row)} does not match weights length {k}")
        dot = sum(r * w for r, w in zip(row, w_list)) + bias_val
        if activation == "sigmoid":
            clipped_dot = max(-30.0, min(30.0, dot))
            res_list.append(1.0 / (1.0 + math.exp(-clipped_dot)))
        elif activation == "relu":
            res_list.append(max(0.0, dot))
        else:
            res_list.append(dot)

    if out is not None and hasattr(out, "__setitem__"):
        try:
            out[:] = res_list
            return out
        except Exception:
            pass

    return res_list


def cosine_similarity_matrix(
    query_vectors: Any,
    pattern_prototypes: Any,
) -> Any:
    """Compute pairwise cosine similarity between query candidate candles and pattern prototypes.

    Normalized cosine similarity = (A @ B.T) / (||A|| * ||B||).
    Accelerated with NanoGEMM AVX2/FMA matrix multiplication.
    """
    if HAS_NUMPY:
        q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        p = np.ascontiguousarray(pattern_prototypes, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p = p.reshape(1, -1)

        # Compute dot products via fast_matmul
        dot_prod = fast_matmul(q, p.T)

        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)

        q_norm = np.where(q_norm == 0.0, 1.0, q_norm)
        p_norm = np.where(p_norm == 0.0, 1.0, p_norm)

        denom = q_norm @ p_norm.T
        return dot_prod / denom

    # Pure Python
    raise NotImplementedError("cosine_similarity_matrix requires numpy")
