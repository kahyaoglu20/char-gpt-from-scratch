"""
ops.py -- the differentiable primitives Assignment 2 adds to the Assignment 1 engine.

MAT 451E, Assignment 2 (character-level GPT: training and generation)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

The Assignment 1 `Tensor` (gpt_a1/autograd.py) is reused unchanged.  It already differentiates
+, *, power, exp, sum, matmul, reshape, transpose and indexing.  The GPT head and the GELU need three
more local derivatives, written here with the engine's own `_make` hook so that they join the same
graph:

    tanh(x)          d tanh = (1 - tanh^2) dx
    log(x)           d log  = dx / x
    pick(Z, y)       z_r = Z[r, y_r]: gathers one entry per row; its backward scatters the upstream value

GELU is *not* a new primitive: it is composed from these operations, so its derivative comes from the
chain rule of the engine and can be compared with the hand-derived g'(z) of the assignment.
"""
import numpy as np

from gpt_a1.autograd import Tensor

SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
GELU_C = 0.044715


def tanh(x):
    """Elementwise tanh; the backward pass reuses the forward value y: g * (1 - y^2)."""
    y = np.tanh(x.data)

    def backward(g):
        x._accumulate(g * (1.0 - y * y))
    return x._make(y, (x,), "tanh", backward)


def log(x):
    """Elementwise natural logarithm (inputs must be positive): g / x."""
    def backward(g):
        x._accumulate(g / x.data)
    return x._make(np.log(x.data), (x,), "log", backward)


def pick(Z, y):
    """z_r = Z[r, y_r] for a 2-D Z (R x V) and integer labels y (R,): vectorized target gathering."""
    rows = np.arange(Z.shape[0])
    y = np.asarray(y)

    def backward(g):
        full = np.zeros_like(Z.data)
        full[rows, y] = g                       # each row receives its upstream value at its own label
        Z._accumulate(full)
    return Z._make(Z.data[rows, y], (Z,), "pick", backward)


def gelu(z):
    """g(z) = z/2 * (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3))), built from engine operations."""
    u = (z + z * z * z * GELU_C) * SQRT_2_OVER_PI     # z*z*z: three cheap products instead of a float power
    return z * 0.5 * (tanh(u) + 1.0)


# ------------------------------------------------------------------ plain NumPy versions (no graph)
def gelu_np(z):
    """The same tanh approximation on a NumPy array."""
    return 0.5 * z * (1.0 + np.tanh(SQRT_2_OVER_PI * (z + GELU_C * z ** 3)))


def gelu_grad_np(z):
    """The assignment's derivative: g'(z) = (1 + tanh u)/2 + z/2 (1 - tanh^2 u) a (1 + 3 c z^2)."""
    t = np.tanh(SQRT_2_OVER_PI * (z + GELU_C * z ** 3))
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_C * z * z)


def gelu_exact_np(z):
    """The exact GELU z * Phi(z), with Phi the standard normal CDF (only for the analytical comparison)."""
    from math import erf, sqrt
    z = np.asarray(z, dtype=np.float64)
    return z * 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def const(a):
    """A constant leaf (no gradient)."""
    return a if isinstance(a, Tensor) else Tensor(np.asarray(a, dtype=np.float64))
