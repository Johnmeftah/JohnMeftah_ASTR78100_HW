import math
import numpy as np


def gaussian_elimination(A, b):
    """
    solving A x = b using basic Gaussian elimination with partial pivoting.

    returning the solution vector x.
    """

    # copying inputs to avoid modifying the originals
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    # getting system size
    n = len(b)

    # performing forward elimination
    for i in range(n):
        # finding pivot row
        pivot_row = i + np.argmax(np.abs(A[i:, i]))
        if A[pivot_row, i] == 0:
            print("matrix is singular or nearly singular.")
            return None

        # swapping rows if needed
        if pivot_row != i:
            A[[i, pivot_row]] = A[[pivot_row, i]]
            b[[i, pivot_row]] = b[[pivot_row, i]]

        # eliminating entries below pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
            b[j] = b[j] - factor * b[i]

    # performing back substitution
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


def solve_resistor_network(V_plus):
    """
    solving the resistor network node voltages V1, V2, V3, V4.

    using KCL at each node with all resistors equal.
    taking the top node as V_plus and the bottom node as 0.
    """

    # building the matrix from KCL equations
    # equation at V1: 4V1 - V2 - V3 - V4 = V_plus
    # equation at V2: -V1 + 3V2 - V4 = 0
    # equation at V3: -V1 + 3V3 - V4 = V_plus
    # equation at V4: -V1 - V2 - V3 + 4V4 = 0
    A = np.array([
        [4.0, -1.0, -1.0, -1.0],
        [-1.0, 3.0,  0.0, -1.0],
        [-1.0, 0.0,  3.0, -1.0],
        [-1.0, -1.0, -1.0, 4.0],
    ])

    # building the right-hand side vector
    b = np.array([V_plus, 0.0, V_plus, 0.0], dtype=float)

    # solving using numpy.linalg.solve
    x_lu = np.linalg.solve(A, b)

    # solving using Gaussian elimination
    x_ge = gaussian_elimination(A, b)

    return A, b, x_lu, x_ge


# running the script when executed directly
if __name__ == "__main__":
    # asking for the top voltage
    v_text = input("enter top voltage V_plus in volts (default 5): ").strip()
    V_plus = float(v_text) if v_text else 5.0

    # solving the system
    A, b, x_lu, x_ge = solve_resistor_network(V_plus)

    # unpacking results
    V1_lu, V2_lu, V3_lu, V4_lu = x_lu

    # printing results
    print("\nresistor network solver")
    print(f"V_plus = {V_plus} V, ground = 0 V\n")

    print("solution using numpy.linalg.solve:")
    print(f"V1 = {V1_lu}")
    print(f"V2 = {V2_lu}")
    print(f"V3 = {V3_lu}")
    print(f"V4 = {V4_lu}\n")

    if x_ge is not None:
        V1_ge, V2_ge, V3_ge, V4_ge = x_ge

        print("solution using Gaussian elimination:")
        print(f"V1 = {V1_ge}")
        print(f"V2 = {V2_ge}")
        print(f"V3 = {V3_ge}")
        print(f"V4 = {V4_ge}\n")

        # checking agreement
        diff = np.max(np.abs(x_lu - x_ge))
        print("check:")
        print(f"max |difference| = {diff}\n")
