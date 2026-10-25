"""
model.py -- the complete character-level GPT: residual blocks, feed-forward network, vocabulary
projection and cross-entropy, written once on the autodiff Tensor, plus the hand-derived backward maps.

MAT 451E, Assignment 2 (character-level GPT: training and generation)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

Reused unchanged from Assignment 1 (gpt_a1/model.py): `embed` (one-hot selection + positions),
`layer_norm`, `causal_mask`, `softmax_rows`, and the backward maps `embedding_backward`,
`layer_norm_backward`, `attention_backward`.  New here: the attention as it runs *inside* a block
(batched products on the last two axes, no flattening; Case B5 compares it with the Assignment 1
routine), GELU, the feed-forward network, the two residual sums, L blocks, the final LayerNorm, the
logits, the cross-entropy, and the backward maps of every new piece.

Layout: activations are (B, T, d), batch first; the head flattens them to (BT) x d in row-major order,
row r = b*T + t ("time-major within each batch member"), and the targets Y (B x T) are flattened with
the same rule, so row r of the logits is always scored against label y_r.
"""
import numpy as np

from gpt_a1 import model as A1
from gpt_a1.autograd import Tensor

from . import ops


# ================================================================ forward (on Tensors: differentiable)
def attention(U, W_QKV, W_O, neg=-1e9):
    """Single-head causal self-attention of a (B, T, d) tensor, batched over the leading axis.

    QKV = U W_QKV (B,T,3d);  S = Q K^T / sqrt(d) + M (B,T,T);  A = softmax_row(S);  F = (A V) W_O (B,T,d).
    """
    B, T, d = U.shape
    QKV = (U.reshape(B * T, d) @ W_QKV).reshape(B, T, 3 * d)   # one (BT x d)(d x 3d) product for Q, K and V
    Q, K, Vm = QKV[:, :, :d], QKV[:, :, d:2 * d], QKV[:, :, 2 * d:]
    S = (Q @ K.mT) * (1.0 / np.sqrt(d)) + A1.causal_mask(T, neg)   # one T x T score page per sample
    A = A1.softmax_rows(S)                           # rows are probability distributions over keys 0..i
    Z = A @ Vm                                       # (B,T,T) @ (B,T,d) -> (B,T,d): mixes *tokens*
    F = (Z.reshape(B * T, d) @ W_O).reshape(B, T, d)
    return F, {"Q": Q, "K": K, "V": Vm, "S": S, "A": A, "Z": Z}


def feed_forward(D, W_1, b_1, W_2, b_2):
    """G = g(D W1 + 1 b1) W2 + 1 b2: expand to 4d, GELU, contract to d. Mixes *features* of each token.

    No token interacts with another here, so the B x T rows are flattened into one (BT) x d matrix."""
    B, T, d = D.shape
    J = D.reshape(B * T, d) @ W_1 + b_1              # (BT x d)(d x 4d) + 1 b1: bias row broadcast to every row
    Ga = ops.gelu(J)                                 # elementwise tanh-approximation of GELU
    G = (Ga @ W_2 + b_2).reshape(B, T, d)            # (BT x 4d)(4d x d) + 1 b2, back to B x T x d
    return G, {"J": J, "Ga": Ga}


def block(H, P, l, eps=1e-5, neg=-1e9):
    """One pre-normalization Transformer block: R = H + Attn(LN1(H)),  H' = R + FFN(LN2(R))."""
    k = f"block{l}."
    U, ln1 = A1.layer_norm(H, P[k + "ln1_gamma"], P[k + "ln1_beta"], eps)
    F, att = attention(U, P[k + "W_QKV"], P[k + "W_O"], neg)
    R = H + F                                        # first residual: the identity path keeps H
    D, ln2 = A1.layer_norm(R, P[k + "ln2_gamma"], P[k + "ln2_beta"], eps)
    G, ffn = feed_forward(D, P[k + "W_1"], P[k + "b_1"], P[k + "W_2"], P[k + "b_2"])
    Hn = R + G                                       # second residual
    return Hn, {"H": H, "U": U, "ln1": ln1, "F": F, "R": R, "D": D, "ln2": ln2, "G": G, **att, **ffn}


def gpt_forward(X, P, L=2, eps=1e-5, neg=-1e9):
    """Indices X (B x T) -> logits Z ((BT) x V) through embeddings, L blocks, the final LN and W_voc."""
    B, T = np.asarray(X).shape
    H, O = A1.embed(X, P["W_E"], P["W_P"])           # (B,T,d): token rows plus position rows
    blocks = []
    for l in range(1, L + 1):
        H, c = block(H, P, l, eps, neg)
        blocks.append(c)
    Hf, lnf = A1.layer_norm(H, P["lnf_gamma"], P["lnf_beta"], eps)
    d = Hf.shape[-1]
    Hbar = Hf.reshape(B * T, d)                      # row r = b*T + t (row-major flattening)
    Z = Hbar @ P["W_voc"]                            # ((BT) x d) @ (d x V) -> (BT) x V logits
    return Z, {"O": O, "blocks": blocks, "HL": H, "lnf": lnf, "Hbar": Hbar, "B": B, "T": T}


def cross_entropy(Z, y):
    """Mean next-token negative log-likelihood with a stable log-sum-exp and vectorized gathering.

    LSE_r = a_r + log sum_j exp(Z_rj - a_r),  a_r = max_j Z_rj (a constant shift),
    L = -(1/R) sum_r (Z_{r,y_r} - LSE_r).
    """
    R = Z.shape[0]
    a = Z.max(axis=1, keepdims=True)                 # constant: softmax(Z - a) = softmax(Z), so no gradient
    lse = a + ops.log((Z - a).exp().sum(axis=1, keepdims=True))   # (R,1); every exponent is <= 0
    return (lse.reshape(R) - ops.pick(Z, y)).mean()


def loss_fn(X, Y, P, L=2, eps=1e-5):
    """The training objective on one batch: L(theta) = cross_entropy(gpt_forward(X), flatten(Y))."""
    Z, cache = gpt_forward(X, P, L, eps)
    return cross_entropy(Z, np.asarray(Y).reshape(-1)), Z, cache


def as_constants(P):
    """The same arrays as constant Tensors: evaluation and sampling build no graph and touch no gradient."""
    return {k: Tensor(v.data if isinstance(v, Tensor) else v) for k, v in P.items()}


# ================================================================ plain NumPy helpers
def softmax_np(Z):
    E = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)


def cross_entropy_np(Z, y):
    """Loss value and P = softmax(Z) on plain arrays (used by the hand-derived backward pass)."""
    a = Z.max(axis=1, keepdims=True)
    lse = (a + np.log(np.exp(Z - a).sum(axis=1, keepdims=True)))[:, 0]
    return float(np.mean(lse - Z[np.arange(len(y)), y])), softmax_np(Z)


# ================================================================ hand-derived backward maps
def cross_entropy_backward(Z, y):
    """dL/dZ = (P - E) / R, E the one-hot targets: every row sums to zero."""
    R = Z.shape[0]
    G = softmax_np(Z)
    G[np.arange(R), y] -= 1.0
    return G / R


def feed_forward_backward(G_G, D, J, Ga, W_1, W_2):
    """G_W2 = g(J)^T G_G, G_b2 = 1^T G_G, G_g = G_G W2^T, G_J = G_g * g'(J), G_W1 = D^T G_J, G_b1 = 1^T G_J,
    G_D = G_J W1^T (sums over every batch and time row)."""
    d = D.shape[-1]
    G2 = G_G.reshape(-1, d)                          # (BT) x d: the rows are independent
    G_W2 = Ga.T @ G2                                 # (4d x BT)(BT x d)
    G_b2 = G2.sum(axis=0, keepdims=True)             # 1_T^T G_G, summed over every row
    G_Ga = G2 @ W_2.T                                # (BT x d)(d x 4d)
    G_J = G_Ga * ops.gelu_grad_np(J)                 # elementwise chain rule through g
    G_W1 = D.reshape(-1, d).T @ G_J                  # (d x BT)(BT x 4d)
    G_b1 = G_J.sum(axis=0, keepdims=True)
    G_D = (G_J @ W_1.T).reshape(D.shape)             # back to B x T x d
    return {"W_2": G_W2, "b_2": G_b2, "Ga": G_Ga, "J": G_J, "W_1": G_W1, "b_1": G_b1, "D": G_D}


def block_backward(G_Hn, c, P, l):
    """Backward through one block. Each residual sum sends the upstream gradient along the identity path
    unchanged and adds the branch's contribution: dL/dX = G_Y + (dF/dX)^T G_Y."""
    k = f"block{l}."
    val = lambda t: t.data if isinstance(t, Tensor) else t
    g = {}
    # H' = R + G  ->  G_R = G_H' (identity path) + contribution through the feed-forward branch
    ffn = feed_forward_backward(G_Hn, val(c["D"]), val(c["J"]), val(c["Ga"]), val(P[k + "W_1"]), val(P[k + "W_2"]))
    G_R_ln, g_ln2, b_ln2 = A1.layer_norm_backward(ffn["D"], val(c["ln2"]["xhat"]), val(c["ln2"]["s"]), val(P[k + "ln2_gamma"]))
    G_R = G_Hn + G_R_ln
    # R = H + F  ->  G_H = G_R (identity path) + contribution through attention and LN1
    att = A1.attention_backward(G_R, val(c["U"]), val(c["Q"]), val(c["K"]), val(c["V"]), val(c["A"]), val(c["Z"]),
                                val(P[k + "W_QKV"]), val(P[k + "W_O"]))
    G_H_ln, g_ln1, b_ln1 = A1.layer_norm_backward(att["U"], val(c["ln1"]["xhat"]), val(c["ln1"]["s"]), val(P[k + "ln1_gamma"]))
    G_H = G_R + G_H_ln
    g.update({k + "W_1": ffn["W_1"], k + "b_1": ffn["b_1"], k + "W_2": ffn["W_2"], k + "b_2": ffn["b_2"],
              k + "ln2_gamma": g_ln2, k + "ln2_beta": b_ln2, k + "W_QKV": att["W_QKV"], k + "W_O": att["W_O"],
              k + "ln1_gamma": g_ln1, k + "ln1_beta": b_ln1})
    return G_H, g, {"R": G_R, "F": G_R, "D": ffn["D"], "J": ffn["J"], "U": att["U"]}


def gpt_backward(G_Z, cache, P, L=2):
    """Every parameter gradient from dL/dZ: head -> final LN -> blocks L..1 -> embeddings."""
    val = lambda t: t.data if isinstance(t, Tensor) else t
    B, T = cache["B"], cache["T"]
    Hbar = val(cache["Hbar"])
    d = Hbar.shape[1]
    g = {"W_voc": Hbar.T @ G_Z}                                     # d x V = (d x BT)(BT x V)
    G_Hf = (G_Z @ val(P["W_voc"]).T).reshape(B, T, d)              # undo the flattening with the same rule
    G_H, g["lnf_gamma"], g["lnf_beta"] = A1.layer_norm_backward(G_Hf, val(cache["lnf"]["xhat"]), val(cache["lnf"]["s"]),
                                                                val(P["lnf_gamma"]))
    inner = {}
    for l in range(L, 0, -1):
        G_H, gb, inner[l] = block_backward(G_H, cache["blocks"][l - 1], P, l)
        g.update(gb)
    g["W_E"], g["W_P"] = A1.embedding_backward(G_H, cache["O"], val(P["W_P"]).shape[0])
    g["H0"] = G_H
    return g, inner


# ================================================================ straightforward references
def attention_per_sample(U, W_QKV, W_O, neg=-1e9):
    """One sample at a time with plain 2-D matrices (the acceptance-test reference)."""
    return A1.attention_per_sample(U, W_QKV, W_O, neg)
