import math

def f(x):
    # defining the function to integrate
    return x**4 - 2*x + 1


def trapezoidal_rule(a, b, N):
    """
    computing the integral using the trapezoidal rule.

    integrating f(x) from a to b using N slices.
    """

    # computing the step size
    h = (b - a) / N

    # initializing the sum with endpoint contributions
    s = 0.5 * f(a) + 0.5 * f(b)

    # adding interior points
    for k in range(1, N):
        s += f(a + k * h)

    # returning the final integral value
    return h * s


# running the script when executed directly
if __name__ == "__main__":
    # setting the integration limits
    a = 0.0
    b = 2.0

    # setting the step counts
    N1 = 10
    N2 = 20

    # computing the integrals
    I_10 = trapezoidal_rule(a, b, N1)
    I_20 = trapezoidal_rule(a, b, N2)

    # estimating the error using step doubling
    error_estimate = abs(I_20 - I_10)

    # printing results
    print("\ntrapezoidal rule error estimation")
    print("integrating f(x) = x^4 - 2x + 1 from 0 to 2\n")

    print(f"N = {N1}, integral ≈ {I_10}")
    print(f"N = {N2}, integral ≈ {I_20}\n")

    print(f"estimated error at N = {N2} ≈ {error_estimate}")
