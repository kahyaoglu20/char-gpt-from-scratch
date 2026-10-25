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
