import math
import numpy as np
import matplotlib.pyplot as plt


def generate_decay_times(N, half_life, seed=1):
    """
    generating decay times using the transformation method.

    sampling from the exponential decay distribution.
    """

    rng = np.random.default_rng(seed)

    # generating uniform random numbers
    u = rng.random(N)

    # computing decay times
    tau = half_life / math.log(2.0)
    t_decay = -tau * np.log(u)

    return t_decay


def remaining_atoms_vs_time(t_decay):
    """
    computing the number of atoms remaining as a function of time.

    using sorted decay times.
    """

    # sorting decay times
    t_sorted = np.sort(t_decay)

    # computing remaining atoms
    N = len(t_sorted)
    remaining = N - np.arange(N)

    return t_sorted, remaining


# running the script when executed directly
if __name__ == "__main__":
    # setting parameters
    N_atoms = 1000
    half_life = 6.075

    # generating decay times
    t_decay = generate_decay_times(N_atoms, half_life)

    # computing remaining atoms vs time
    t_vals, N_remaining = remaining_atoms_vs_time(t_decay)

    # printing a quick check
    print("\nradioactive decay using transformation method")
    print(f"initial atoms = {N_atoms}")
    print(f"half-life = {half_life} days\n")

    # plotting results
    plt.figure()
    plt.step(t_vals, N_remaining, where="post")
    plt.xlabel("time (days)")
    plt.ylabel("number of atoms not decayed")
    plt.title("Ni-56 decay using inverse transform sampling")
    plt.show()
