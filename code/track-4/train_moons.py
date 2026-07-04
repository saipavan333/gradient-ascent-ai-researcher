"""
Track 4 · Lesson 4 — Train a neural network with your own autograd
Run:  python train_moons.py
The payoff: a multi-layer perceptron, built on the Value engine from Lesson 2,
learns a curved decision boundary on the two-moons dataset — using gradients
computed entirely by our hand-written backprop. No PyTorch, no NumPy autograd.
"""
import math
import random

random.seed(0)


# ---- the autograd engine (Lesson 2), compact and self-contained ----
class Value:
    def __init__(self, data, _children=()):
        self.data = data; self.grad = 0.0
        self._backward = lambda: None; self._prev = set(_children)

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def b(): self.grad += out.grad; o.grad += out.grad
        out._backward = b; return out

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def b(): self.grad += o.data*out.grad; o.grad += self.data*out.grad
        out._backward = b; return out

    def __pow__(self, p):
        out = Value(self.data**p, (self,))
        def b(): self.grad += p*self.data**(p-1)*out.grad
        out._backward = b; return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def b(): self.grad += (1-t*t)*out.grad
        out._backward = b; return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()

    def __neg__(self): return self*-1
    def __radd__(self, o): return self+o
    def __sub__(self, o): return self+(-o)
    def __rmul__(self, o): return self*o


# ---- a neural net built on Value ----
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def parameters(self): return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout): self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]


def make_moons(n):
    pts, labels = [], []
    for _ in range(n):
        t = random.uniform(0, math.pi)
        pts.append([math.cos(t)+random.gauss(0, 0.12), math.sin(t)+random.gauss(0, 0.12)]); labels.append(-1.0)
        pts.append([1-math.cos(t)+random.gauss(0, 0.12), 0.4-math.sin(t)+random.gauss(0, 0.12)]); labels.append(1.0)
    return pts, labels


def main():
    X, Y = make_moons(50)                       # 100 points total
    model = MLP(2, [8, 8, 1])                    # 2 inputs -> two hidden layers -> 1 output
    print("parameters:", len(model.parameters()))

    for epoch in range(80):
        # forward: mean squared error of tanh output vs target in {-1,+1}
        preds = [model([Value(x[0]), Value(x[1])]) for x in X]
        loss = sum((p - y)**2 for p, y in zip(preds, Y)) * (1.0/len(X))
        # backward
        for p in model.parameters(): p.grad = 0.0
        loss.backward()
        # SGD update (learning rate decays a little)
        lr = 0.5 - 0.4*epoch/80
        for p in model.parameters(): p.data -= lr * p.grad
        if epoch % 20 == 0 or epoch == 79:
            acc = sum((pr.data > 0) == (y > 0) for pr, y in zip(preds, Y)) / len(X)
            print(f"epoch {epoch:3d}  loss {loss.data:.4f}  train-acc {acc:.0%}")

    print("Done — the MLP learned the moons using gradients from our own engine.")


if __name__ == "__main__":
    main()
