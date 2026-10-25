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
