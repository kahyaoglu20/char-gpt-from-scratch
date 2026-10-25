"""
autograd.py -- a minimal reverse-mode automatic-differentiation engine.

MAT 451E, Assignment 1 (character-level GPT, foundations)
Author: Ömer Kahyaoğlu <kahyaoglu20@itu.edu.tr>, student no. 090240741

This is micrograd (A. Karpathy) lifted from scalars to tensors.  A Tensor wraps
a NumPy array (.data).  When a Tensor is produced by an operation it remembers
its parents and a small closure that applies the chain rule *locally*:

        parent.grad  +=  (d out / d parent)^T  out.grad

Calling .backward() on a scalar visits the graph in reverse topological order,
so every Tensor that (transitively) requires a gradient ends up holding
dJ/dTensor in .grad -- leaves (the parameters) as well as intermediates such as
the attention matrix A, which lets us check every matrix-calculus identity of
the assignment term by term.  Only fundamental tensor operations are used:
add, multiply, power, exp, sum, matmul, reshape, transpose and indexing.
"""
import numpy as np


def _unbroadcast(g, shape):
    """Sum a broadcast gradient back to the shape of the operand it belongs to."""
    while g.ndim > len(shape):                      # leading axes created by broadcasting
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):                  # axes stretched from size 1
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


class Tensor:
    """A NumPy array that remembers how it was computed (one node of the graph)."""

    def __init__(self, data, requires_grad=False, _parents=(), _op="", name=""):
        data = np.asarray(data)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)
        self.data = data
        self.requires_grad = bool(requires_grad)
        self.grad = None                            # dJ/dself after backward()
        self._parents = _parents
        self._backward = None
        self._op = _op
        self.name = name

    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    dtype = property(lambda self: self.data.dtype)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}', requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------ plumbing
    def _const(self, x):
        """Wrap a NumPy array / Python number as a constant leaf (no gradient)."""
        return x if isinstance(x, Tensor) else Tensor(np.asarray(x, dtype=self.data.dtype))

    def _make(self, data, parents, op, backward):
        """Create the output node; record the graph only if a parent needs a gradient."""
        needs = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs, _parents=parents if needs else (), _op=op)
        if needs:
            out._backward = lambda: backward(out.grad)
        return out

    def _accumulate(self, g):
        """grad += g  (summing over broadcast axes); repeated uses of a tensor add up."""
        if self.requires_grad:
            g = _unbroadcast(np.asarray(g, dtype=self.data.dtype), self.data.shape)
            self.grad = g.copy() if self.grad is None else self.grad + g

    # --------------------------------------------------------- elementwise algebra
    def __add__(self, other):
        other = self._const(other)

        def backward(g):                            # d(a+b) = da + db
            self._accumulate(g)
            other._accumulate(g)
        return self._make(self.data + other.data, (self, other), "add", backward)

    __radd__ = __add__

    def __mul__(self, other):
        other = self._const(other)

        def backward(g):                            # d(a*b) = b*da + a*db  (Hadamard)
            self._accumulate(g * other.data)
            other._accumulate(g * self.data)
        return self._make(self.data * other.data, (self, other), "mul", backward)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._const(other))

    def __rsub__(self, other):
        return self._const(other) + (-self)

    def __pow__(self, p):                           # p is a Python number
        def backward(g):                            # d(a^p) = p a^(p-1) da
            self._accumulate(g * p * self.data ** (p - 1))
        return self._make(self.data ** p, (self,), f"pow({p})", backward)

    def __truediv__(self, other):
        return self * self._const(other) ** -1.0

    def __rtruediv__(self, other):
        return self._const(other) * self ** -1.0

    def exp(self):
        y = np.exp(self.data)

        def backward(g):                            # d exp(a) = exp(a) da
            self._accumulate(g * y)
        return self._make(y, (self,), "exp", backward)

    # ------------------------------------------------------------- linear algebra
    def __matmul__(self, other):
        other = self._const(other)

        def backward(g):                            # C = AB:  G_A = G_C B^T,  G_B = A^T G_C
            self._accumulate(g @ np.swapaxes(other.data, -1, -2))
            other._accumulate(np.swapaxes(self.data, -1, -2) @ g)
        return self._make(self.data @ other.data, (self, other), "matmul", backward)

    # ----------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        def backward(g):                            # every summand receives the upstream value
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.data.shape))
        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", backward)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else int(np.prod(np.array(self.data.shape)[np.atleast_1d(axis)]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims=False):
        """Maximum as a *constant* (no gradient).  It is only used as the softmax shift
        m_i, and softmax(s - m) does not depend on m, so its exact gradient is zero."""
        return Tensor(self.data.max(axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------- shape handling
    def reshape(self, *shape):
        def backward(g):
            self._accumulate(g.reshape(self.data.shape))
        return self._make(self.data.reshape(*shape), (self,), "reshape", backward)

    def transpose(self, *axes):
        inverse = np.argsort(axes)

        def backward(g):
            self._accumulate(g.transpose(inverse))
        return self._make(self.data.transpose(*axes), (self,), "transpose", backward)

    @property
    def mT(self):
        """Transpose of the last two axes (K^T for a (..., T, d) tensor)."""
        axes = list(range(self.ndim))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        return self.transpose(*axes)

    def __getitem__(self, idx):
        def backward(g):                            # scatter-add: repeated indices accumulate
            full = np.zeros_like(self.data)
            np.add.at(full, idx, g)
            self._accumulate(full)
        return self._make(self.data[idx], (self,), "getitem", backward)

    def detach(self):
        """A numerical copy with no history: gradients cannot flow through it."""
        return Tensor(self.data.copy())

    # ------------------------------------------------------------------- backward
    def backward(self, seed=None):
        """Reverse-mode sweep: seed dJ/dJ = 1 and apply every local chain rule."""
        order, seen = [], set()

        def visit(v):                               # depth-first topological order
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = np.ones_like(self.data) if seed is None else np.asarray(seed, dtype=self.data.dtype)
        for v in reversed(order):
            if v._backward is not None:
                v._backward()


def parameter(array, name=""):
    """A leaf Tensor that requires a gradient (a trainable parameter)."""
    return Tensor(np.array(array, copy=True), requires_grad=True, name=name)
