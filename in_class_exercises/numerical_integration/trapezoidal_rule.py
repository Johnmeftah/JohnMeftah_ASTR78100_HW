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

    # asking for the number of slices
    N_text = input("enter N (e.g. 10 or 10,100,1000): ").strip()
    N_values = [int(n.strip()) for n in N_text.split(",") if n.strip()]

    # printing a clean header
    print("\ntrapezoidal rule integration")
    print("integrating f(x) = x^4 - 2x + 1 from 0 to 2\n")

    # computing and printing results
    for N in N_values:
        result = trapezoidal_rule(a, b, N)
        print(f"N = {N}")
        print(f"integral ≈ {result}\n")
