"""
model.py -- differentiable embeddings, LayerNorm and single-head causal attention.

MAT 451E, Assignment 1 (character-level GPT, foundations)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Forward maps are written once with the from-scratch autograd Tensor, so every
parameter stays in the differentiation graph.  The hand-derived backward maps
(the matrix-calculus identities of the assignment) are written separately on
plain NumPy arrays; the test-suite checks them against autograd and against
centred finite differences.  Batch-first layout: activations are (B, T, d).
"""
import numpy as np

from .autograd import Tensor

COUNTS = {"embed": 0, "layer_norm": 0, "attention": 0}   # tested forward evaluations


# ============================================================== token + position embeddings
def one_hot(X, V, dtype=np.float64):
    """O[b, t, :] = e_{X[b,t]}^T : the (B, T, V) one-hot selection tensor."""
    X = np.asarray(X)
    B, T = X.shape
    O = np.zeros((B, T, V), dtype=dtype)
    O[np.arange(B)[:, None], np.arange(T)[None, :], X] = 1.0
    return O


def embed(X, W_E, W_P):
    """H0_b = O_b W_E + W_P[0:T, :] -> (B, T, d); both tables stay in the graph."""
    COUNTS["embed"] += 1
    B, T = np.asarray(X).shape
    V, d = W_E.shape
    O = one_hot(X, V, W_E.dtype)                    # (B, T, V) constant 0/1 selector
    E = Tensor(O) @ W_E                             # (B, T, V) @ (V, d) -> (B, T, d)
    H = E + W_P[0:T]                                # (T, d) rows broadcast over the batch
    return H, O


def embed_lookup(X, W_E, W_P):
    """The same map written as a direct row lookup W_E[X] (used for comparison)."""
    T = np.asarray(X).shape[1]
    return W_E[np.asarray(X)] + W_P[0:T]            # (B, T, d)


# ========================================================================= layer normalization
def layer_norm(H, gamma, beta, eps=1e-5):
    """Feature-wise LayerNorm of every token row: LN(H) = C / sqrt(v + eps) * gamma + beta."""
    COUNTS["layer_norm"] += 1
    mu = H.mean(axis=-1, keepdims=True)             # (..., T, 1)  mu = (1/d) H 1_d
    C = H - mu                                      # (..., T, d)  C = H - mu 1_d^T
    v = (C * C).mean(axis=-1, keepdims=True)        # (..., T, 1)  v = (1/d) (C o C) 1_d
    s = (v + eps) ** 0.5                            # (..., T, 1)  eps inside the square root
    xhat = C / s                                    # (..., T, d)  standardized rows
    Y = xhat * gamma + beta                         # (1, d) gain and shift rows broadcast
    return Y, {"mu": mu, "v": v, "s": s, "xhat": xhat}


# ===================================================================== causal self-attention
def causal_mask(T, neg=-1e9):
    """M_ij = 0 for j <= i and M_ij = -1e9 for j > i (T x T)."""
    return np.triu(np.full((T, T), neg), k=1)


def softmax_rows(S):
    """Stable softmax across the key axis (last axis): every row becomes a distribution."""
    E = (S - S.max(axis=-1, keepdims=True)).exp()   # exp(S_ij - m_i) <= 1: no overflow
    return E / E.sum(axis=-1, keepdims=True)


def attend(Q, K, Vm, neg=-1e9):
    """S = Q K^T / sqrt(d) + M,  A = softmax_row(S),  Z = A Vm   (batched over leading axes)."""
    COUNTS["attention"] += 1
    T, d = Q.shape[-2], Q.shape[-1]
    S = (Q @ K.mT) * (1.0 / np.sqrt(d)) + causal_mask(T, neg)   # (..., T, T)
    A = softmax_rows(S)                                          # (..., T, T)
    Z = A @ Vm                                                   # (..., T, d)
    return S, A, Z


def causal_self_attention(U, W_QKV, W_O, neg=-1e9):
    """Single-head causal self-attention F_b = softmax(Q_b K_b^T / sqrt(d) + M) V_b W_O.

    Dimension trace: (BT) x d -> (BT) x 3d -> B x T x 3d -> three B x T x d blocks
                     -> B x T x T (one score page per sample) -> B x T x d.
    """
    B, T, d = U.shape
    QKV = U.reshape(B * T, d) @ W_QKV               # (BT, d) @ (d, 3d) -> (BT, 3d)
    QKV = QKV.reshape(B, T, 3 * d)                  # (B, T, 3d)
    Q = QKV[:, :, 0:d]                              # (B, T, d) queries
    K = QKV[:, :, d:2 * d]                          # (B, T, d) keys
    Vm = QKV[:, :, 2 * d:3 * d]                     # (B, T, d) values (not the vocabulary size)
    S, A, Z = attend(Q, K, Vm, neg)                 # (B, T, T), (B, T, T), (B, T, d)
    F = (Z.reshape(B * T, d) @ W_O).reshape(B, T, d)   # (BT, d) @ (d, d) -> (B, T, d)
    return F, {"Q": Q, "K": K, "V": Vm, "S": S, "A": A, "Z": Z}


def forward_a1(X, W_E, W_P, gamma, beta, W_QKV, W_O, eps=1e-5, neg=-1e9):
    """Assignment-1 forward stage: indices -> H0 -> U = LN1(H0) -> F (attention output)."""
    H, O = embed(X, W_E, W_P)                       # (B, T, d)
    U, ln = layer_norm(H, gamma, beta, eps)         # (B, T, d)
    F, att = causal_self_attention(U, W_QKV, W_O, neg)   # (B, T, d)
    return F, {"O": O, "H": H, "U": U, "ln": ln, **att}


# ============================================== hand-derived backward maps (plain NumPy arrays)
def embedding_backward(G_H, O, T_max):
    """dL/dW_E = sum_b O_b^T D_b ;  dL/dW_P[t, :] = sum_b D_b[t, :]  (zero rows for t >= T)."""
    B, T, d = G_H.shape
    V = O.shape[-1]
    G_WE = O.reshape(B * T, V).T @ G_H.reshape(B * T, d)    # (V, BT) @ (BT, d) -> (V, d)
    G_WP = np.zeros((T_max, d), dtype=G_H.dtype)
    G_WP[:T] = G_H.sum(axis=0)                              # positions aggregate over samples
    return G_WE, G_WP


def layer_norm_backward(G_Y, xhat, s, gamma):
    """dL/dx_j = (1/s) [q_j - mean_k q_k - xhat_j mean_k(q_k xhat_k)],  q = g * gamma."""
    q = G_Y * gamma                                         # (..., T, d)
    G_X = (q - q.mean(axis=-1, keepdims=True)
           - xhat * (q * xhat).mean(axis=-1, keepdims=True)) / s
    tokens = tuple(range(G_Y.ndim - 1))                     # every axis except the features
    G_gamma = (G_Y * xhat).sum(axis=tokens).reshape(1, -1)  # sum over tokens -> (1, d)
    G_beta = G_Y.sum(axis=tokens).reshape(1, -1)            # (1, d)
    return G_X, G_gamma, G_beta


def attention_backward(G_F, U, Q, K, Vm, A, Z, W_QKV, W_O):
    """Matrix-calculus backward pass of single-head causal attention (batched over b)."""
    B, T, d = U.shape
    G_WO = Z.reshape(B * T, d).T @ G_F.reshape(B * T, d)    # sum_b Z_b^T G_F,b        (d, d)
    G_Z = G_F @ W_O.T                                       # G_F W_O^T                (B, T, d)
    G_A = G_Z @ np.swapaxes(Vm, -1, -2)                     # G_Z V^T                  (B, T, T)
    G_V = np.swapaxes(A, -1, -2) @ G_Z                      # A^T G_Z                  (B, T, d)
    G_S = A * (G_A - (A * G_A).sum(axis=-1, keepdims=True))  # row-wise softmax Jacobian (B, T, T)
    G_Q = G_S @ K / np.sqrt(d)                              # G_S K / sqrt(d)          (B, T, d)
    G_K = np.swapaxes(G_S, -1, -2) @ Q / np.sqrt(d)         # G_S^T Q / sqrt(d)        (B, T, d)
    G_QKV = np.concatenate([G_Q, G_K, G_V], axis=-1)        # same order as [Q K V]    (B, T, 3d)
    G_WQKV = U.reshape(B * T, d).T @ G_QKV.reshape(B * T, 3 * d)   # [U^T G_Q, U^T G_K, U^T G_V]
    G_U = G_QKV @ W_QKV.T                                   # G_Q W_Q^T + G_K W_K^T + G_V W_V^T
    return {"W_O": G_WO, "Z": G_Z, "A": G_A, "V": G_V, "S": G_S,
            "Q": G_Q, "K": G_K, "W_QKV": G_WQKV, "U": G_U}


def backward_a1(G_F, cache, T_max, gamma, W_QKV, W_O):
    """Chain the hand-derived maps: attention -> LayerNorm -> embeddings."""
    c = {k: (v.data if isinstance(v, Tensor) else v) for k, v in cache.items() if k != "ln"}
    g = attention_backward(G_F, c["U"], c["Q"], c["K"], c["V"], c["A"], c["Z"], W_QKV, W_O)
    G_H, G_gamma, G_beta = layer_norm_backward(g["U"], cache["ln"]["xhat"].data,
                                               cache["ln"]["s"].data, gamma)
    G_WE, G_WP = embedding_backward(G_H, c["O"], T_max)
    g.update({"H": G_H, "gamma": G_gamma, "beta": G_beta, "W_E": G_WE, "W_P": G_WP})
    return g


# ================================================================ straightforward reference
def attention_per_sample(U, W_QKV, W_O, neg=-1e9):
    """Reference implementation: one sample at a time, plain 2-D matrices, no reshapes."""
    B, T, d = U.shape
    M = causal_mask(T, neg)
    W_Q, W_K, W_V = W_QKV[:, :d], W_QKV[:, d:2 * d], W_QKV[:, 2 * d:]
    F = np.empty_like(U)
    for b in range(B):
        Q, K, V = U[b] @ W_Q, U[b] @ W_K, U[b] @ W_V        # (T, d) each
        S = Q @ K.T / np.sqrt(d) + M                        # (T, T)
        E = np.exp(S - S.max(axis=1, keepdims=True))
        A = E / E.sum(axis=1, keepdims=True)                # rows sum to one
        F[b] = (A @ V) @ W_O                                # (T, d)
    return F
