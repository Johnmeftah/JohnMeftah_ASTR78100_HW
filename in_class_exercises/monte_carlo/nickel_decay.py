import math
import numpy as np
import matplotlib.pyplot as plt


def decay_probability(dt, half_life):
    """
    computing the decay probability in a time step dt using a half-life.

    using p = 1 - 2^(-dt / half_life).
    """

    return 1.0 - 2.0 ** (-dt / half_life)


def run_decay_simulation(N0, dt, t_end, tau_ni=6.075, tau_co=77.236, seed=1):
    """
    simulating Ni-56 -> Co-56 -> Fe-56 using a Monte Carlo method.

    updating in reverse order each step:
    decaying Co to Fe first, then decaying Ni to Co.
    """

    # setting the random seed for reproducibility
    rng = np.random.default_rng(seed)

    # setting initial populations
    N_ni = int(N0)
    N_co = 0
    N_fe = 0

    # computing number of steps
    n_steps = int(round(t_end / dt))

    # preparing arrays for storing results
    t_vals = np.zeros(n_steps + 1)
    ni_vals = np.zeros(n_steps + 1, dtype=int)
    co_vals = np.zeros(n_steps + 1, dtype=int)
    fe_vals = np.zeros(n_steps + 1, dtype=int)

    # storing initial values
    ni_vals[0] = N_ni
    co_vals[0] = N_co
    fe_vals[0] = N_fe

    # precomputing decay probabilities per step
    p_ni = decay_probability(dt, tau_ni)
    p_co = decay_probability(dt, tau_co)

    # looping over time steps
    for i in range(1, n_steps + 1):
        t_vals[i] = i * dt

        # decaying Co -> Fe first
        if N_co > 0:
            decays_co = rng.binomial(N_co, p_co)
        else:
            decays_co = 0

        N_co -= decays_co
        N_fe += decays_co

        # decaying Ni -> Co second
        if N_ni > 0:
            decays_ni = rng.binomial(N_ni, p_ni)
        else:
            decays_ni = 0

        N_ni -= decays_ni
        N_co += decays_ni

        # storing results
        ni_vals[i] = N_ni
        co_vals[i] = N_co
        fe_vals[i] = N_fe

    return t_vals, ni_vals, co_vals, fe_vals


# running the script when executed directly
if __name__ == "__main__":
    # asking for initial number of Ni atoms
    n_text = input("enter initial number of Ni-56 atoms N0 (default 10000): ").strip()
    N0 = int(n_text) if n_text else 10000

    # asking for dt in days
    dt_text = input("enter time step dt in days (default 0.5): ").strip()
    dt = float(dt_text) if dt_text else 0.5

    # asking for total simulation time in days
    t_text = input("enter total time t_end in days (default 200): ").strip()
    t_end = float(t_text) if t_text else 200.0

    # running the simulation
    t, Nni, Nco, Nfe = run_decay_simulation(N0, dt, t_end)

    # printing final populations
    print("\nnuclear decay monte carlo")
    print(f"final Ni = {Nni[-1]}")
    print(f"final Co = {Nco[-1]}")
    print(f"final Fe = {Nfe[-1]}\n")

    # plotting results
    plt.figure()
    plt.plot(t, Nni, label="Ni-56")
    plt.plot(t, Nco, label="Co-56")
    plt.plot(t, Nfe, label="Fe-56")
    plt.xlabel("time (days)")
    plt.ylabel("number of atoms")
    plt.title("Ni-56 -> Co-56 -> Fe-56 decay (monte carlo)")
    plt.legend()
    plt.show()
