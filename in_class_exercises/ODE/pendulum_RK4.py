import math
import numpy as np
import matplotlib.pyplot as plt


def pendulum_derivs(state, g, L):
    """
    computing derivatives for the nonlinear pendulum.

    using state = [theta, omega].
    returning [dtheta/dt, domega/dt].
    """

    theta, omega = state
    dtheta = omega
    domega = -(g / L) * math.sin(theta)
    return np.array([dtheta, domega], dtype=float)


def rk4_step(state, t, h, g, L):
    """
    advancing one step using fourth-order Runge-Kutta.

    returning the updated state.
    """

    k1 = pendulum_derivs(state, g, L)
    k2 = pendulum_derivs(state + 0.5 * h * k1, g, L)
    k3 = pendulum_derivs(state + 0.5 * h * k2, g, L)
    k4 = pendulum_derivs(state + h * k3, g, L)

    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_pendulum(theta0_deg, omega0, L, g, h, t_end):
    """
    simulating the pendulum motion using RK4.

    returning time array and theta array in degrees.
    """

    n_steps = int(round(t_end / h))

    t_vals = np.zeros(n_steps + 1)
    theta_vals = np.zeros(n_steps + 1)

    # setting initial conditions
    theta0 = math.radians(theta0_deg)
    state = np.array([theta0, omega0], dtype=float)

    theta_vals[0] = theta0_deg

    # stepping forward in time
    for i in range(1, n_steps + 1):
        t_vals[i] = i * h
        state = rk4_step(state, t_vals[i - 1], h, g, L)
        theta_vals[i] = math.degrees(state[0])

    return t_vals, theta_vals


# running the script when executed directly
if __name__ == "__main__":
    # setting physical parameters
    L = 0.1
    g = 9.81

    # setting initial conditions
    theta0_deg = 179
    omega0 = 0

    # asking for time step
    h_text = input("enter time step h in seconds (default 0.001): ").strip()
    h = float(h_text) if h_text else 0.001

    # estimating small-angle period to pick a reasonable simulation length
    T0 = 2 * math.pi * math.sqrt(L / g)

    # asking for number of periods to simulate
    p_text = input("enter number of periods to simulate (default 5): ").strip()
    n_periods = int(p_text) if p_text else 5

    # setting total time
    t_end = n_periods * T0

    # running the simulation
    t, theta_deg = simulate_pendulum(theta0_deg, omega0, L, g, h, t_end)

    # plotting theta vs time
    plt.figure()
    plt.plot(t, theta_deg)
    plt.xlabel("time (s)")
    plt.ylabel("theta (deg)")
    plt.title("pendulum angle vs time (rk4)")
    plt.show()
