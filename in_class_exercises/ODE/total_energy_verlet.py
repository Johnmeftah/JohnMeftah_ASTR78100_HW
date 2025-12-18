import math
import numpy as np
import matplotlib.pyplot as plt


def acceleration(r, G, M):
    # computing gravitational acceleration for a 2D position vector r
    x, y = r
    dist = math.sqrt(x*x + y*y)
    factor = -G * M / (dist**3)
    return np.array([factor * x, factor * y], dtype=float)


def energies(r, v, G, M, m):
    # computing kinetic, potential, and total energy
    x, y = r
    vx, vy = v

    dist = math.sqrt(x*x + y*y)
    speed2 = vx*vx + vy*vy

    KE = 0.5 * m * speed2
    PE = -(G * M * m) / dist
    E = KE + PE

    return KE, PE, E


def verlet_orbit_with_energy(r0, v0, h, n_steps, G, M, m):
    """
    simulating an orbit using the velocity Verlet method.

    returning time, x, y, KE, PE, and total energy arrays.
    """

    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)

    t_vals = np.zeros(n_steps + 1)
    x_vals = np.zeros(n_steps + 1)
    y_vals = np.zeros(n_steps + 1)

    KE_vals = np.zeros(n_steps + 1)
    PE_vals = np.zeros(n_steps + 1)
    E_vals = np.zeros(n_steps + 1)

    x_vals[0] = r[0]
    y_vals[0] = r[1]

    KE, PE, E = energies(r, v, G, M, m)
    KE_vals[0] = KE
    PE_vals[0] = PE
    E_vals[0] = E

    a = acceleration(r, G, M)

    for i in range(1, n_steps + 1):
        t_vals[i] = i * h

        r_new = r + v * h + 0.5 * a * h * h
        a_new = acceleration(r_new, G, M)
        v_new = v + 0.5 * (a + a_new) * h

        r = r_new
        v = v_new
        a = a_new

        x_vals[i] = r[0]
        y_vals[i] = r[1]

        KE, PE, E = energies(r, v, G, M, m)
        KE_vals[i] = KE
        PE_vals[i] = PE
        E_vals[i] = E

    return t_vals, x_vals, y_vals, KE_vals, PE_vals, E_vals


# running the script when executed directly
if __name__ == "__main__":
    # setting constants
    G = 6.6738e-11
    M = 1.9891e30
    m = 5.9722e24

    # setting initial conditions at perihelion
    r0 = [1.4710e11, 0]
    v0 = [0, 3.0287e4]

    # setting time step
    h = 3600

    # asking for number of orbits
    o_text = input("enter number of orbits to simulate (default 3): ").strip()
    n_orbits = int(o_text) if o_text else 3

    # estimating orbital period for step count
    r_mag = math.sqrt(r0[0]*r0[0] + r0[1]*r0[1])
    T_est = 2 * math.pi * math.sqrt(r_mag**3 / (G * M))

    t_end = n_orbits * T_est
    n_steps = int(round(t_end / h))

    t, x, y, KE, PE, E = verlet_orbit_with_energy(r0, v0, h, n_steps, G, M, m)

    # plotting the orbit
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("earth orbit using verlet method")
    plt.axis("equal")
    plt.show()

    # plotting KE, PE, and total energy
    plt.figure()
    plt.plot(t, KE, label="kinetic energy")
    plt.plot(t, PE, label="potential energy")
    plt.plot(t, E, label="total energy")
    plt.xlabel("time (s)")
    plt.ylabel("energy (J)")
    plt.title("energy components vs time")
    plt.legend()
    plt.show()

    # plotting total energy alone
    plt.figure()
    plt.plot(t, E)
    plt.xlabel("time (s)")
    plt.ylabel("total energy (J)")
    plt.title("total energy vs time")
    plt.show()
