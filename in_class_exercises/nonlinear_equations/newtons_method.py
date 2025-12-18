import math
import numpy as np
import matplotlib.pyplot as plt


def P(x):
    # defining the polynomial
    return (
        924*x**6
        - 2772*x**5
        + 3150*x**4
        - 1680*x**3
        + 420*x**2
        - 42*x
        + 1
    )


def dP(x):
    # defining the derivative of the polynomial
    return (
        5544*x**5
        - 13860*x**4
        + 12600*x**3
        - 5040*x**2
        + 840*x
        - 42
    )


def newtons_method(x0, tol=1e-12, max_iter=10000):
    """
    solving P(x) = 0 using Newton's method.

    starting from initial guess x0.
    stopping when successive updates are smaller than tol.
    """

    # setting the initial guess
    x = x0

    # iterating until convergence
    for _ in range(max_iter):
        fx = P(x)
        dfx = dP(x)

        # stopping if derivative is too small
        if abs(dfx) < 1e-14:
            return None

        x_new = x - fx / dfx

        # checking convergence
        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    return None


# running the script when executed directly
if __name__ == "__main__":
    # creating points for plotting
    x_plot = np.linspace(0.0, 1.0, 1000)
    y_plot = P(x_plot)

    # plotting the polynomial for inspection
    plt.figure()
    plt.plot(x_plot, y_plot)
    plt.axhline(0.0)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.title("sixth-order polynomial P(x) on [0, 1]")
    plt.show()

    # setting initial guesses near each root
    initial_guesses = [0.05, 0.15, 0.35, 0.55, 0.75, 0.95]
    roots = []

    # running Newton's method from each guess
    for x0 in initial_guesses:
        root = newtons_method(x0)
        if root is not None:
            roots.append(root)

    # removing duplicates caused by nearby guesses
    roots = np.array(roots)
    roots = np.unique(np.round(roots, 12))

    # printing results
    print("\nroots of P(x) using Newton's method:\n")
    for r in roots:
        print(f"{r:.10f}")
