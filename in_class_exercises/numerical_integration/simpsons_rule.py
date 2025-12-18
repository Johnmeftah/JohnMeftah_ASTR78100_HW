import math

def f(x):
    # defining the function to integrate
    return x**4 - 2*x + 1


def simpsons_rule(a, b, N):
    """
    computing the integral using Simpson's rule.

    integrating f(x) from a to b using N slices.
    requiring N to be even.
    """

    # stopping if N is not valid
    if N <= 0:
        print("N must be a positive integer.")
        return None

    # stopping if N is odd (Simpson's rule requires even N)
    if N % 2 != 0:
        print("N must be even for Simpson's rule.")
        return None

    # computing the step size
    h = (b - a) / N

    # starting the sum with endpoints
    s = f(a) + f(b)

    # adding odd-indexed terms with weight 4
    for k in range(1, N, 2):
        s += 4 * f(a + k * h)

    # adding even-indexed terms with weight 2
    for k in range(2, N, 2):
        s += 2 * f(a + k * h)

    # returning the final integral value
    return (h / 3) * s


# running the script when executed directly
if __name__ == "__main__":
    # setting the integration limits
    a = 0.0
    b = 2.0

    # asking for the number of slices
    N_text = input("enter N (even, e.g. 10 or 10,100,1000): ").strip()
    N_values = [int(n.strip()) for n in N_text.split(",") if n.strip()]

    # printing a clean header
    print("\nsimpson's rule integration")
    print("integrating f(x) = x^4 - 2x + 1 from 0 to 2\n")

    # computing and printing results
    for N in N_values:
        result = simpsons_rule(a, b, N)
        if result is not None:
            print(f"N = {N}")
            print(f"integral ≈ {result}\n")
