import math
import numpy as np
import matplotlib.pyplot as plt


def make_charge_density(N, q1=1, q2=-1):
    """
    creating a simple charge density with one positive and one negative blob.

    returning rho on an N x N grid.
    """

    rho = np.zeros((N, N), dtype=float)

    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # setting blob centers
    x1, y1 = int(0.7 * N), int(0.3 * N)
    x2, y2 = int(0.3 * N), int(0.7 * N)

    # setting blob width
    sigma = 0.08 * N
    s2 = sigma * sigma

    # creating gaussian blobs
    rho += q1 * np.exp(-((X - x1)**2 + (Y - y1)**2) / (2 * s2))
    rho += q2 * np.exp(-((X - x2)**2 + (Y - y2)**2) / (2 * s2))

    return rho


def solve_poisson_gauss_seidel(rho, h=1, tol=1e-6, max_iter=500000):
    """
    solving the 2D Poisson equation using Gauss-Seidel relaxation.

    using zero potential on the boundary.
    stopping when max update is below tol.
    """

    N = rho.shape[0]
    phi = np.zeros((N, N), dtype=float)

    h2 = h * h

    for it in range(max_iter):
        max_delta = 0

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                old = phi[i, j]

                phi[i, j] = 0.25 * (
                    phi[i + 1, j]
                    + phi[i - 1, j]
                    + phi[i, j + 1]
                    + phi[i, j - 1]
                    + h2 * rho[i, j]
                )

                delta = abs(phi[i, j] - old)
                if delta > max_delta:
                    max_delta = delta

        if max_delta < tol:
            return phi, it + 1, max_delta

    return phi, max_iter, max_delta


# running the script when executed directly
if __name__ == "__main__":
    # asking for grid size
    n_text = input("enter grid size N (default 101): ").strip()
    N = int(n_text) if n_text else 101

    # building the charge density
    rho = make_charge_density(N)

    # solving Poisson's equation
    phi, iterations, final_delta = solve_poisson_gauss_seidel(rho, h=1, tol=1e-6)

    # printing convergence info
    print("\npoisson solver")
    print(f"grid size N = {N}")
    print(f"iterations = {iterations}")
    print(f"final max update = {final_delta}\n")

    # plotting the potential using a colored colormap
    plt.figure()
    plt.imshow(phi, origin="upper", cmap="coolwarm")
    plt.colorbar(label="potential")
    plt.title("poisson equation solution (gauss-seidel)")
    plt.show()
