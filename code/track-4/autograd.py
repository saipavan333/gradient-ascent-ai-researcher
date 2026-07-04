"""
Track 4 · Lesson 2 — Build an autograd engine (a tiny 'micrograd')
Run:  python autograd.py
A `Value` wraps a number, remembers how it was computed, and knows how to push
gradients backward through that computation via the chain rule. This ~60-line
engine is, in miniature, exactly what PyTorch's autograd does.
"""
import math
import random


class Value:
    """A scalar value in a computational graph, with automatic differentiation."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None      # how to propagate grad to inputs
        self._prev = set(_children)        # the Values that produced this one
        self._op = _op                     # the operation (for debugging/printing)

    # ---- operations: each builds an output node AND a local backward rule ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad          # d(a+b)/da = 1
            other.grad += out.grad         # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad   # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Value(self.data ** p, (self,), f"**{p}")
        def _backward():
            self.grad += p * self.data ** (p - 1) * out.grad   # power rule
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad   # d/dx tanh = 1 - tanh^2
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad      # d/dx e^x = e^x
        out._backward = _backward
        return out

    # ---- the heart: reverse-mode autodiff over the whole graph ----
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)                 # topological order: inputs before outputs
        self.grad = 1.0             # seed: d(self)/d(self) = 1
        for v in reversed(topo):    # walk outputs -> inputs, applying chain rule
            v._backward()

    # ---- convenience so Values behave like numbers ----
    def __neg__(self): return self * -1
    def __radd__(self, o): return self + o
    def __sub__(self, o): return self + (-o)
    def __rsub__(self, o): return o + (-self)
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * o ** -1
    def __rtruediv__(self, o): return o * self ** -1
    def __repr__(self): return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def main():
    # The example from the lesson: e = a*b, d = e + c
    a = Value(2.0); b = Value(-3.0); c = Value(10.0)
    e = a * b; d = e + c
    d.backward()
    print("forward:  e =", e.data, " d =", d.data)              # -6, 4
    print("grads:    a =", a.grad, " b =", b.grad, " c =", c.grad)  # -3, 2, 1

    # Gradient check against finite differences -- proves the engine is correct
    def f(av, bv, cv):
        return (Value(av) * Value(bv) + Value(cv)).data
    h = 1e-6
    a2, b2, c2 = 2.0, -3.0, 10.0
    num_a = (f(a2+h, b2, c2) - f(a2-h, b2, c2)) / (2*h)
    print("finite-diff da:", round(num_a, 5), " engine da:", a.grad)   # both -3

    # A deeper graph through a tanh neuron, all differentiated automatically
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    bias = Value(6.881373587)
    n = x1*w1 + x2*w2 + bias
    o = n.tanh()
    o.backward()
    print("neuron out:", round(o.data, 4), " dx1:", round(x1.grad, 4))  # ~0.7071, ~-1.5

    # --- Try it yourself -----------------------------------------------------
    # 1) Add a `.sigmoid()` method and gradient-check it against finite differences.


if __name__ == "__main__":
    main()
