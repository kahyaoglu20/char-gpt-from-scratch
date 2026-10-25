"""
params.py -- every trainable tensor of the L-block character GPT, and its count.

MAT 451E, Assignment 1 (character-level GPT, foundations)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Weights ~ N(0, sigma^2) independently, biases = 0, LayerNorm gains = 1 and
LayerNorm shifts = 0.  Matrices act by right multiplication (Y = X W), so a
weight is stored d_in x d_out exactly as in the assignment (no transposed
nn.Linear convention).  No QKV, output-projection or head biases.
"""
import numpy as np

from .autograd import Tensor


def init_params(V, T, d, L, sigma=0.02, rng=None, dtype=np.float64):
    """Return an ordered dict name -> Tensor(requires_grad=True)."""
    rng = np.random.default_rng(451) if rng is None else rng

    def W(*shape):                                  # Gaussian weight, W_ij ~ N(0, sigma^2)
        return Tensor(rng.normal(0.0, sigma, size=shape).astype(dtype), requires_grad=True)

    def zeros(*shape):                              # biases and LayerNorm shifts
        return Tensor(np.zeros(shape, dtype=dtype), requires_grad=True)

    def ones(*shape):                               # LayerNorm gains
        return Tensor(np.ones(shape, dtype=dtype), requires_grad=True)

    P = {"W_E": W(V, d),                            # token embeddings      V x d
         "W_P": W(T, d)}                            # position embeddings   T x d
    for l in range(1, L + 1):
        P[f"block{l}.W_QKV"] = W(d, 3 * d)          # [W_Q W_K W_V]         d x 3d
        P[f"block{l}.W_O"] = W(d, d)                # attention output      d x d
        P[f"block{l}.W_1"] = W(d, 4 * d)            # FFN expand            d x 4d
        P[f"block{l}.b_1"] = zeros(1, 4 * d)        #                       1 x 4d
        P[f"block{l}.W_2"] = W(4 * d, d)            # FFN contract          4d x d
        P[f"block{l}.b_2"] = zeros(1, d)            #                       1 x d
        P[f"block{l}.ln1_gamma"] = ones(1, d)       # LN1 gain / shift      1 x d each
        P[f"block{l}.ln1_beta"] = zeros(1, d)
        P[f"block{l}.ln2_gamma"] = ones(1, d)       # LN2 gain / shift      1 x d each
        P[f"block{l}.ln2_beta"] = zeros(1, d)
    P["lnf_gamma"] = ones(1, d)                     # final LN gain / shift 1 x d each
    P["lnf_beta"] = zeros(1, d)
    P["W_voc"] = W(d, V)                            # vocabulary projection d x V (untied)
    return P


def count_params(P):
    """Programmatic count: total number of scalar entries of all trainable tensors."""
    return int(sum(p.data.size for p in P.values()))


def count_formula(V, T, d, L):
    """P = V d + T d + L (12 d^2 + 9 d) + 2 d + d V."""
    return 2 * V * d + T * d + L * (12 * d * d + 9 * d) + 2 * d
