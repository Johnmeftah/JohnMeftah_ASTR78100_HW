import math
import numpy as np
import matplotlib.pyplot as plt


def surface_temperature(t, A, B, tau):
    # computing the time-varying surface temperature
    return A + B * math.sin(2 * math.pi * t / tau)


def simulate_crust_diffusion(
    L=20,
    D=0.1,
    tau=365,
    A=10,
    B=12,
    T_bottom=11,
    dx=0.1,
    dt=0.01,
    total_years=20,
    settle_years=9,
):
    """
    simulating 1d heat diffusion in the earth's crust with a seasonal surface boundary.

    using an explicit finite-difference method.
    returning depth values and temperature profiles saved during year 10 every 3 months.
    """

    # building the spatial grid
    Nx = int(round(L / dx)) + 1
    x = np.linspace(0, L, Nx)

    # checking stability
    r = D * dt / (dx * dx)
    if r > 0.5:
        print("dt is too large for stability, lowering dt or increasing dx is needed.")
        print(f"current r = {r}")
        return None, None, None

    # setting initial condition
    T = np.full(Nx, A, dtype=float)

    # enforcing the bottom boundary
    T[-1] = T_bottom

    # setting total time
    total_days = int(round(total_years * tau))
    n_steps = int(round(total_days / dt))

    # setting snapshot times during year 10 every 3 months
    year10_start = settle_years * tau
    snap_times = [
        year10_start + 0,
        year10_start + tau / 4,
        year10_start + tau / 2,
        year10_start + 3 * tau / 4,
        year10_start + tau,
    ]

    # preparing storage
    saved_times = []
    saved_profiles = []

    # simulating forward in time
    step = 0
    snap_index = 0
    while step <= n_steps:
        t = step * dt

        # enforcing the surface boundary
        T[0] = surface_temperature(t, A, B, tau)

        # enforcing the bottom boundary
        T[-1] = T_bottom

        # saving snapshots when crossing target times
        if snap_index < len(snap_times) and t >= snap_times[snap_index]:
            saved_times.append(t)
            saved_profiles.append(T.copy())
            snap_index += 1

        # stopping early once all snapshots are saved
        if snap_index >= len(snap_times) and t > year10_start + tau:
            break

        # updating interior points using the diffusion equation
        T_new = T.copy()
        T_new[1:-1] = T[1:-1] + r * (T[2:] - 2 * T[1:-1] + T[:-2])
        T = T_new

        step += 1

    return x, saved_times, saved_profiles


# running the script when executed directly
if __name__ == "__main__":
    # asking for dx
    dx_text = input("enter spatial step dx in meters (default 0.1): ").strip()
    dx = float(dx_text) if dx_text else 0.1

    # asking for dt
    dt_text = input("enter time step dt in days (default 0.01): ").strip()
    dt = float(dt_text) if dt_text else 0.01

    # running the simulation
    x, times, profiles = simulate_crust_diffusion(dx=dx, dt=dt)

    if x is None:
        raise SystemExit

    # plotting temperature profiles in year 10 every 3 months
    plt.figure()
    for t, T in zip(times, profiles):
        label = f"t = {t/365:.2f} years"
        plt.plot(x, T, label=label)

    plt.xlabel("depth (m)")
    plt.ylabel("temperature (C)")
    plt.title("temperature vs depth in year 10 (every 3 months)")
    plt.legend()
    plt.show()
