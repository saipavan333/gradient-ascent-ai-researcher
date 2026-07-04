"""
Track 1 · Lesson 4 — Linear combinations, span & basis
Run:  python span_basis.py
Given a few vectors, which points can you reach (span)? Which vectors pull
their weight (independence)? And what are a point's coordinates in a basis?
"""
import numpy as np


def linear_combination(coeffs, vectors):
    """c1*v1 + c2*v2 + ... given lists of scalars and vectors."""
    out = np.zeros(len(vectors[0]), dtype=float)
    for c, v in zip(coeffs, vectors):
        out += c * np.asarray(v, float)
    return out


def independent_2d(a, b):
    """Two 2-D vectors are independent iff a1*b2 - a2*b1 != 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    cross = a[0] * b[1] - a[1] * b[0]
    return not np.isclose(cross, 0.0)


def coordinates_in_basis(p, f1, f2):
    """Solve c*f1 + d*f2 = p for the coordinates (c, d)."""
    F = np.column_stack([f1, f2])         # 2x2 matrix whose columns are the basis
    return np.linalg.solve(F, p)          # unique solution (basis => invertible)


def main():
    print("2u + 1w =", linear_combination([2, 1], [(2.0, 0.5), (0.6, 1.8)]))  # [4.6 2.8]

    print("indep (1,0),(0,1):", independent_2d((1, 0), (0, 1)))   # True
    print("indep (2,1),(4,2):", independent_2d((2, 1), (4, 2)))   # False -> (4,2)=2*(2,1)

    # Coordinates of (3,2) in the skewed basis f1=(1,0.4), f2=(0.4,1)
    c, d = coordinates_in_basis(np.array([3.0, 2.0]),
                                np.array([1.0, 0.4]),
                                np.array([0.4, 1.0]))
    print("coords of (3,2):", round(c, 2), round(d, 2))           # 2.62 0.95

    # Same point (7,4) in two different bases
    c2, d2 = coordinates_in_basis(np.array([7.0, 4.0]),
                                  np.array([1.0, 1.0]),
                                  np.array([1.0, -1.0]))
    print("coords of (7,4) in {(1,1),(1,-1)}:", c2, d2)           # 5.5 1.5

    # --- Try it yourself -----------------------------------------------------
    # 1) Do (1,1) and (2,2) span the plane? Use independent_2d to check.
    # 2) Reconstruct (7,4) from the basis and its coordinates; confirm you get back (7,4).


if __name__ == "__main__":
    main()
