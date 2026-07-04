"""
Track 14 · Lesson 2 — Probing a network's hidden representations
Run:  python probing.py

Mechanistic interpretability asks: what has a network actually learned to represent?
A LINEAR PROBE tests whether a concept is decodable from a hidden layer: freeze the
network, train a simple linear classifier on its activations to predict the concept.
High probe accuracy = the concept is linearly represented there; chance = it isn't.
"""
import numpy as np
rng = np.random.default_rng(0)

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1/(1+np.exp(-z))

def make(n):
    X = rng.uniform(-1, 1, (n, 2))
    q1 = (X[:,0] > 0).astype(int); q2 = (X[:,1] > 0).astype(int)
    y = (q1 ^ q2)                                   # task: XOR of the two signs
    return X, y, q1, q2

def train_mlp(X, y, H=16, epochs=3000, lr=0.1):
    W1 = rng.normal(0, 0.5, (2, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 0.5, (H, 1)); b2 = 0.0
    for _ in range(epochs):
        h = relu(X@W1+b1); p = sigmoid(h@W2+b2).ravel()
        dp = (p - y)/len(y)
        dW2 = h.T@dp[:,None]; db2 = dp.sum()
        dh = (dp[:,None]@W2.T)*(h>0)
        dW1 = X.T@dh; db1 = dh.sum(0)
        W1-=lr*dW1; b1-=lr*db1; W2-=lr*dW2; b2-=lr*db2
    return (W1,b1,W2,b2)

def hidden(net, X): W1,b1,_,_ = net; return relu(X@W1+b1)

def linear_probe(feats, target, epochs=500, lr=0.1):
    w = np.zeros(feats.shape[1]); b = 0.0
    for _ in range(epochs):
        p = sigmoid(feats@w+b); g = p-target
        w -= lr*feats.T@g/len(feats); b -= lr*g.mean()
    return ((sigmoid(feats@w+b)>0.5)==target).mean()

def main():
    X, y, q1, q2 = make(1500); net = train_mlp(X, y)
    h = hidden(net, X)
    acc = ((sigmoid(h@net[2]+net[3]).ravel()>0.5)==y).mean()
    print(f"MLP solves the XOR-of-signs task: accuracy {acc:.3f}\n")
    print("Linear-probing the hidden layer for various concepts:")
    print(f"  sign of x1 (needed for the task) : probe acc {linear_probe(h, q1):.3f}")
    print(f"  sign of x2 (needed for the task) : probe acc {linear_probe(h, q2):.3f}")
    rand = rng.integers(0,2,len(y))
    print(f"  a RANDOM label (not represented) : probe acc {linear_probe(h, rand):.3f}")
    print("\nThe hidden layer linearly encodes exactly the concepts the task needs")
    print("(both signs), and NOT random noise — a probe reveals what a layer 'knows'.")

if __name__ == "__main__":
    main()
