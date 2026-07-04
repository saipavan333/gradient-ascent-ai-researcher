"""
Track 1 · Lesson 7 — The determinant: area, volume, invertibility
Run:  python determinant.py
det A is the (signed) factor by which a matrix scales area/volume; it is zero
exactly when the matrix collapses dimension and is not invertible.
"""
import numpy as np


def det2x2(A):
    """The 2x2 formula: ad - bc."""
    (a, b), (c, d) = A
    return a*d - b*c


def polygon_area(P):
    """Shoelace formula for the area of a polygon given its vertices."""
    x, y = P[:, 0], P[:, 1]
    return 0.5*abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]])
    print("det2x2      :", det2x2(A))          # 3.0
    print("np.linalg.det:", np.linalg.det(A))  # ~3.0

    # Area scaling: a region's area is multiplied by |det A|
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    mapped = square @ A.T
    print("area 1 ->", round(polygon_area(mapped), 3))   # 3.0

    # det(AB) = det(A) det(B)
    B = np.array([[1.0, 2.0], [0.0, 1.0]])     # a shear, det = 1
    print("det(AB)==det(A)det(B):",
          np.isclose(np.linalg.det(A@B), np.linalg.det(A)*np.linalg.det(B)))  # True

    # det = 0  <=>  singular  <=>  dependent columns  <=>  not invertible
    S = np.array([[1.0, 2.0], [2.0, 4.0]])     # column 2 = 2 * column 1
    print("singular det:", round(np.linalg.det(S), 12))   # 0.0
    try:
        np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        print("not invertible:", e)

    # det(A^-1) = 1 / det(A)
    M = np.random.randn(3, 3)
    print("det(inv)==1/det:",
          np.isclose(np.linalg.det(np.linalg.inv(M)), 1/np.linalg.det(M)))   # True

    # --- Try it yourself -----------------------------------------------------
    # 1) Build a reflection [[ -1,0],[0,1]] and confirm its determinant is -1.
    # 2) Scale a 2x2 matrix by k and watch det multiply by k^2.


if __name__ == "__main__":
    main()
