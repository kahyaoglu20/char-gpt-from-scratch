def gelu_grad_np(z):
    """The assignment's derivative: g'(z) = (1 + tanh u)/2 + z/2 (1 - tanh^2 u) a (1 + 3 c z^2)."""
    t = np.tanh(SQRT_2_OVER_PI * (z + GELU_C * z ** 3))
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_C * z * z)
