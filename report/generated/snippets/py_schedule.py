def schedule(n_train, B, T, epochs=50, s_max=0):
    s_epoch = n_train // (B * T)
    s_total = epochs * s_epoch if s_max <= 0 else min(epochs * s_epoch, s_max)
    return s_epoch, s_total

def evaluate(P, X, Y, L=2):
    """Loss on one batch with constant parameters (no graph is recorded)."""
    Z, _ = gpt_forward(X, as_constants(P), L)
    return float(cross_entropy(Z, np.asarray(Y).reshape(-1)).data)

def fixed_windows(n, T, count):
    """A documented, deterministic set of window starts spread evenly over a split of length n."""
    return np.unique(np.linspace(0, n - T - 1, count).round().astype(np.int64))

def evaluate_fixed(P, ids, T, starts, L=2, chunk=64):
    """Mean loss over fixed windows (equal weight per predicted character): a low-noise estimate."""
    total, count = 0.0, 0
    for i in range(0, len(starts), chunk):
        r = starts[i:i + chunk]
        X = ids[r[:, None] + np.arange(T)]
        Y = ids[r[:, None] + np.arange(T) + 1]
        total += evaluate(P, X, Y, L) * X.size
        count += X.size
    return total / count
