import math
import numpy as np


def f(x):
    # computing the integrand safely
    return math.sin(1 / (x * (2 - x))) ** 2


def hit_or_miss_integral(N, seed=1):
    """
    estimating the integral using the hit-or-miss method.

    using a bounding box with height A = 1 over x in [0, 2].
    returning the estimate and the error.
    """

    rng = np.random.default_rng(seed)

    # setting bounds
    a = 0
    b = 2
    A = 1

    # avoiding endpoints to prevent division by zero
    eps = 1e-12

    # sampling random points in the rectangle
    x = rng.uniform(a + eps, b - eps, N)
    y = rng.uniform(0, A, N)

    # counting hits under the curve
    hits = 0
    for i in range(N):
        if y[i] <= f(x[i]):
            hits += 1

    # estimating the area under the curve
    p_hat = hits / N
    I_hat = (b - a) * A * p_hat

    # estimating the error using binomial variance
    sigma = (b - a) * A * math.sqrt(p_hat * (1 - p_hat) / N)

    return I_hat, sigma


def mean_value_integral(N, seed=1):
    """
    estimating the integral using the mean value method.

    sampling x uniformly in [0, 2] and averaging f(x).
    returning the estimate and the error.
    """

    rng = np.random.default_rng(seed)

    # setting bounds
    a = 0
    b = 2

    # avoiding endpoints to prevent division by zero
    eps = 1e-12

    # sampling x values
    x = rng.uniform(a + eps, b - eps, N)

    # evaluating f(x)
    fx = np.array([f(val) for val in x], dtype=float)

    # computing the integral estimate
    I_hat = (b - a) * np.mean(fx)

    # computing the error from the sample variance
    var_f = np.var(fx, ddof=1)
    sigma = (b - a) * math.sqrt(var_f / N)

    return I_hat, sigma


# running the script when executed directly
if __name__ == "__main__":
    # setting number of points
    N_text = input("enter number of points N (default 10000): ").strip()
    N = int(N_text) if N_text else 10000

    # computing hit-or-miss estimate
    I_hm, err_hm = hit_or_miss_integral(N, seed=1)

    # computing mean value estimate
    I_mv, err_mv = mean_value_integral(N, seed=1)

    # printing results
    print("\nmonte carlo integration")
    print("integrating sin^2[1/(x(2-x))] from 0 to 2\n")

    print("hit-or-miss method:")
    print(f"I about {I_hm}")
    print(f"error about {err_hm}\n")

    print("mean value method:")
    print(f"I about {I_mv}")
    print(f"error about {err_mv}\n")
