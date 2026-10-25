def cross_entropy_backward(Z, y):
    """dL/dZ = (P - E) / R, E the one-hot targets: every row sums to zero."""
    R = Z.shape[0]
    G = softmax_np(Z)
    G[np.arange(R), y] -= 1.0
    return G / R
