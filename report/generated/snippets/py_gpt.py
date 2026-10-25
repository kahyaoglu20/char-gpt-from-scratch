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
