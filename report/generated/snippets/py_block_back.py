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
