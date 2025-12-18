import math
import numpy as np

def f(x):
    # defining the function to integrate
    return x**4 - 2*x + 1


def gauss_legendre_quadrature(a, b, N):
    """
    computing the integral using Gauss-Legendre quadrature.

    using N Gauss points on [-1, 1] and mapping them to [a, b].
    """

    # getting nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(N)

    # mapping nodes from [-1, 1] to [a, b]
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)

    # scaling weights for the interval [a, b]
    wp = 0.5 * (b - a) * w

    # computing the weighted sum
    s = 0.0
    for k in range(N):
        s += wp[k] * f(xp[k])

    # returning the final integral value
    return s


# running the script when executed directly
if __name__ == "__main__":
    # setting the integration limits
    a = 0.0
    b = 2.0

    # asking for the number of Gauss points
    N_text = input("enter number of Gauss points N (default 3): ").strip()
    N = int(N_text) if N_text else 3

    # computing the integral
    result = gauss_legendre_quadrature(a, b, N)

    # printing results
    print("\ngaussian quadrature integration")
    print("integrating f(x) = x^4 - 2x + 1 from 0 to 2\n")
    print(f"N = {N}")
    print(f"integral ≈ {result}\n")
