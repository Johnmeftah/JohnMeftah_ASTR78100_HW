import math
import numpy as np
import matplotlib.pyplot as plt


def relaxation_solver(c, x0=0.5, tol=1e-8, max_iter=10000):
    """
    solving x = 1 - exp(-c x) using the relaxation method.

    starting from an initial guess x0.
    stopping when successive updates differ by less than tol.
    """

    # setting the initial guess
    x_old = x0

    # iterating until convergence
    for _ in range(max_iter):
        x_new = 1.0 - math.exp(-c * x_old)

        # checking convergence
        if abs(x_new - x_old) < tol:
            return x_new

        x_old = x_new

    # warning if convergence fails
    print(f"did not converge for c = {c}")
    return x_old


# running the script when executed directly
if __name__ == "__main__":
    # solving the equation for c = 2
    c_test = 2.0
    x_c2 = relaxation_solver(c_test)

    print("\nrelaxation method solution")
    print(f"c = {c_test}")
    print(f"x ≈ {x_c2}\n")

    # creating c values from 0 to 3
    c_values = np.arange(0.0, 3.01, 0.1)
    x_values = []

    # using the previous solution as the next initial guess
    x_guess = 0.0

    # solving for each c value
    for c in c_values:
        x_guess = relaxation_solver(c, x0=x_guess)
        x_values.append(x_guess)

    # plotting x as a function of c
    plt.figure()
    plt.plot(c_values, x_values, "o-", label="relaxation solution")

    plt.xlabel("c")
    plt.ylabel("x")
    plt.title("solution of x = 1 - exp(-c x)")
    plt.legend()
    plt.show()
