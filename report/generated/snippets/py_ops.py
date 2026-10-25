def tanh(x):
    """Elementwise tanh; the backward pass reuses the forward value y: g * (1 - y^2)."""
    y = np.tanh(x.data)

    def backward(g):
        x._accumulate(g * (1.0 - y * y))
    return x._make(y, (x,), "tanh", backward)

def log(x):
    """Elementwise natural logarithm (inputs must be positive): g / x."""
    def backward(g):
        x._accumulate(g / x.data)
    return x._make(np.log(x.data), (x,), "log", backward)

def pick(Z, y):
    """z_r = Z[r, y_r] for a 2-D Z (R x V) and integer labels y (R,): vectorized target gathering."""
    rows = np.arange(Z.shape[0])
    y = np.asarray(y)

    def backward(g):
        full = np.zeros_like(Z.data)
        full[rows, y] = g                       # each row receives its upstream value at its own label
        Z._accumulate(full)
    return Z._make(Z.data[rows, y], (Z,), "pick", backward)

def gelu(z):
    """g(z) = z/2 * (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3))), built from engine operations."""
    u = (z + z * z * z * GELU_C) * SQRT_2_OVER_PI     # z*z*z: three cheap products instead of a float power
    return z * 0.5 * (tanh(u) + 1.0)
