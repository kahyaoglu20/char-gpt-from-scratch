class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}   # moments start at zero
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Update every array of `params` in place from `grads` (same keys); returns the applied steps."""
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        c1, c2 = 1.0 - b1 ** self.t, 1.0 - b2 ** self.t
        applied = {}
        for k, theta in params.items():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
            self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g * g)
            m_hat, v_hat = self.m[k] / c1, self.v[k] / c2
            delta = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            theta += delta                                         # in place: the model sees the new value
            applied[k] = delta
        return applied

    def fingerprint(self):
        """A cheap digest of the optimizer state (used to prove that validation never changes it)."""
        return (self.t, float(sum(np.abs(m).sum() for m in self.m.values())), float(sum(v.sum() for v in self.v.values())))
