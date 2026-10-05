"""Explicit BCHW <-> B x (C*H*W) conversion; no library layout conversions."""
from numbers import Integral

import numpy as np


def _dimensions(*dims):
    if any(isinstance(d, bool) or not isinstance(d, Integral) or d <= 0 for d in dims):
        raise ValueError("Dimensions must be positive integers")


def _tensor(array, rank):
    if not isinstance(array, np.ndarray) or array.ndim != rank:
        raise ValueError(f"Expected a rank-{rank} NumPy tensor")
    _dimensions(*array.shape)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("Expected finite real integer or floating-point values")


def forward_index(c, h, w, C, H, W):
    """Return the within-batch offset; width varies fastest."""
    _dimensions(C, H, W)
    if any(not isinstance(i, Integral) or isinstance(i, bool) for i in (c, h, w)):
        raise ValueError("Indices must be integers")
    if not (0 <= c < C and 0 <= h < H and 0 <= w < W):
        raise ValueError("Index outside tensor")
    return (c * H + h) * W + w


def reverse_index(k, C, H, W):
    """Explicit quotient/remainder inverse of forward_index."""
    _dimensions(C, H, W)
    if not isinstance(k, Integral) or isinstance(k, bool) or not 0 <= k < C * H * W:
        raise ValueError("Offset outside tensor")
    c = k // (H * W)
    remainder = k % (H * W)
    h = remainder // W
    w = remainder % W
    return c, h, w


def bchw_to_matrix(tensor):
    """Allocate a matrix and copy every scalar using the explicit forward offset."""
    _tensor(tensor, 4)
    B, C, H, W = tensor.shape
    matrix = np.empty((B, C * H * W), dtype=tensor.dtype)
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    k = (c * H + h) * W + w
                    matrix[b, k] = tensor[b, c, h, w]
    return matrix


def matrix_to_bchw(matrix, C, H, W):
    """Copy each matrix scalar to coordinates computed by the inverse mapping."""
    _tensor(matrix, 2)
    _dimensions(C, H, W)
    B, length = matrix.shape
    if length != C * H * W:
        raise ValueError("Matrix width must equal C * H * W")
    tensor = np.empty((B, C, H, W), dtype=matrix.dtype)
    for b in range(B):
        for k in range(length):
            c = k // (H * W)
            remainder = k % (H * W)
            h = remainder // W
            w = remainder % W
            tensor[b, c, h, w] = matrix[b, k]
    return tensor


def reconstruction_errors(original, reconstructed):
    """Compute E=I-I_hat at every index and return E, max(|E|), mean(|E|).

    Subtract Python integers for integer inputs to prevent unsigned wraparound.
    The saved differences and accumulated metrics use float64 precision.
    """
    _tensor(original, 4)
    _tensor(reconstructed, 4)
    if original.shape != reconstructed.shape:
        raise ValueError("Comparison requires identical shapes")
    differences = np.empty(original.shape, dtype=np.float64)
    maximum = total = 0.0
    integer_inputs = original.dtype.kind in "iu" and reconstructed.dtype.kind in "iu"
    B, C, H, W = original.shape
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    left = original[b, c, h, w].item()
                    right = reconstructed[b, c, h, w].item()
                    error = int(left) - int(right) if integer_inputs else float(left) - float(right)
                    differences[b, c, h, w] = error
                    magnitude = abs(error)
                    maximum = max(maximum, magnitude)
                    total += magnitude
    return differences, float(maximum), total / (B * C * H * W)


def verify_correspondence(original, matrix, reconstructed):
    """Exhaustive verification, independent sequential traversal of matrix columns."""
    B, C, H, W = original.shape
    if matrix.shape != (B, C * H * W) or reconstructed.shape != original.shape:
        raise AssertionError("Incorrect output shape")
    if matrix.dtype != original.dtype or reconstructed.dtype != original.dtype:
        raise AssertionError("Conversion changed dtype")
    for b in range(B):
        column = 0
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    if matrix[b, column] != original[b, c, h, w]:
                        raise AssertionError(f"Incorrect matrix element at {(b, column)}")
                    if reconstructed[b, c, h, w] != original[b, c, h, w]:
                        raise AssertionError(f"Incorrect reconstruction at {(b, c, h, w)}")
                    if forward_index(c, h, w, C, H, W) != column:
                        raise AssertionError("Incorrect forward arithmetic")
                    if reverse_index(column, C, H, W) != (c, h, w):
                        raise AssertionError("Incorrect inverse arithmetic")
                    column += 1
