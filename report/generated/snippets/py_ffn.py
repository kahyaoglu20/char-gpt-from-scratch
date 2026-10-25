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
