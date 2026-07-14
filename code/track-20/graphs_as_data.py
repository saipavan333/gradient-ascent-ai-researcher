"""
Track 14 (Graph Neural Networks) · Lesson 0 — Graphs as data
============================================================
A graph is nodes joined by edges. We store the edges in an ADJACENCY MATRIX A:
A[i,j]=1 if node i links to node j. From A we get each node's degree (its number
of neighbours) and the "normalised adjacency" that GNNs actually use. This file
builds a two-community graph and reads its structure straight off the matrices.

Run:  python graphs_as_data.py
"""
import numpy as np
rng = np.random.default_rng(0)

def stochastic_block_model(n_per=20, p_in=0.35, p_out=0.03):
    """Two communities; dense links inside, sparse links between."""
    N = 2 * n_per
    comm = np.array([0] * n_per + [1] * n_per)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p:
                A[i, j] = A[j, i] = 1
    return A, comm

def normalised_adjacency(A):
    """Â = D^-1/2 (A+I) D^-1/2 — add self-loops, then scale by degree."""
    A = A + np.eye(len(A))
    d = A.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(d))
    return Dinv @ A @ Dinv

if __name__ == "__main__":
    A, comm = stochastic_block_model()
    N = len(A)
    deg = A.sum(1)                                  # degree = row sum
    print(f"{N} nodes, {int(A.sum() // 2)} edges. Average degree {deg.mean():.1f}.")
    # A is symmetric (undirected) and has no self-loops
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0)
    # count intra- vs inter-community edges from the matrix
    intra = inter = 0
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                if comm[i] == comm[j]: intra += 1
                else: inter += 1
    print(f"edges inside a community: {intra}   between communities: {inter}")
    assert intra > inter                            # the community structure is real
    # A^2 counts length-2 paths (shared neighbours)
    A2 = A @ A
    print(f"node 0 shares {int(A2[0].sum() - A2[0,0]):d} two-step paths with others")
    # the normalised adjacency is what a GNN multiplies by
    Ahat = normalised_adjacency(A)
    assert np.allclose(Ahat, Ahat.T)
    print(f"normalised adjacency Â is symmetric, rows scaled by degree — ready for a GNN.")
    print("OK — a graph is just nodes + an adjacency matrix; degree and Â read straight off it.")
