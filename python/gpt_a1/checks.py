"""
checks.py -- centred finite differences and error measures.

MAT 451E, Assignment 1 (character-level GPT, foundations)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

D_E J(theta) ~ [J(theta + hE) - J(theta - hE)] / (2h),  exact value <grad J, E>_F.
"""
import numpy as np


def rel_err(a, b):
    """The assignment's relative discrepancy |a - b| / max(1, |a|, |b|)."""
    return float(abs(a - b) / max(1.0, abs(a), abs(b)))


def rel_err_strict(a, b, tiny=1e-300):
    """Scale-free relative error |a - b| / max(|a|, |b|) (stricter for small gradients)."""
    return float(abs(a - b) / max(abs(a), abs(b), tiny))


def central_difference(J, theta, E, h):
    """Centred difference along direction E; theta (a NumPy array) is restored exactly."""
    original = theta.copy()
    theta[...] = original + h * E
    J_plus = J()
    theta[...] = original - h * E
    J_minus = J()
    theta[...] = original
    return (J_plus - J_minus) / (2.0 * h)


def fd_check(J, theta, grad, entries, h_values, rng=None):
    """Compare grad with centred differences: selected entries and one random direction.

    Returns {h: {"entry_rel": max assignment rel. error over entries,
                 "entry_rel_strict": max strict rel. error, "dir_fd", "dir_exact",
                 "dir_rel", "dir_rel_strict"}}.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    E_dir = rng.standard_normal(theta.shape)
    exact_dir = float(np.sum(grad * E_dir))         # <grad J, E>_F
    out = {}
    for h in h_values:
        rel, rel_s = [], []
        for idx in entries:
            E = np.zeros_like(theta)
            E[idx] = 1.0
            fd = central_difference(J, theta, E, h)
            rel.append(rel_err(fd, grad[idx]))
            rel_s.append(rel_err_strict(fd, grad[idx]))
        fd_dir = central_difference(J, theta, E_dir, h)
        out[h] = {"entry_rel": max(rel), "entry_rel_strict": max(rel_s),
                  "dir_fd": float(fd_dir), "dir_exact": exact_dir,
                  "dir_rel": rel_err(fd_dir, exact_dir),
                  "dir_rel_strict": rel_err_strict(fd_dir, exact_dir)}
    return out


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
