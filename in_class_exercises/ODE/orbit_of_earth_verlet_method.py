import math
import numpy as np
import matplotlib.pyplot as plt


def acceleration(r, G, M):
    """
    computing gravitational acceleration for a 2D position vector r.

    using a = -G M r / |r|^3.
    """

    x, y = r
    dist = math.sqrt(x*x + y*y)
    factor = -G * M / (dist**3)
    return np.array([factor * x, factor * y], dtype=float)


def verlet_orbit(r0, v0, h, n_steps, G, M):
    """
    simulating an orbit using the position Verlet method.

    returning arrays of x and y positions.
    """

    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)

    x_vals = np.zeros(n_steps + 1)
    y_vals = np.zeros(n_steps + 1)

    x_vals[0] = r[0]
    y_vals[0] = r[1]

    a = acceleration(r, G, M)

    for i in range(1, n_steps + 1):
        r_new = r + v * h + 0.5 * a * h * h
        a_new = acceleration(r_new, G, M)
        v_new = v + 0.5 * (a + a_new) * h

        r = r_new
        v = v_new
        a = a_new

        x_vals[i] = r[0]
        y_vals[i] = r[1]

    return x_vals, y_vals


# running the script when executed directly
if __name__ == "__main__":
    # setting physical constants
    G = 6.6738e-11
    M = 1.9891e30

    # setting initial conditions at perihelion
    r0 = [1.4710e11, 0]
    v0 = [0, 3.0287e4]

    # setting time step
    h = 3600

    # asking for number of orbits
    o_text = input("enter number of orbits to simulate (default 3): ").strip()
    n_orbits = int(o_text) if o_text else 3

    # estimating orbital period using circular approximation for step count
    r_mag = math.sqrt(r0[0]*r0[0] + r0[1]*r0[1])
    T_est = 2 * math.pi * math.sqrt(r_mag**3 / (G * M))

    t_end = n_orbits * T_est
    n_steps = int(round(t_end / h))

    # running the simulation
    x, y = verlet_orbit(r0, v0, h, n_steps, G, M)

    # plotting the orbit
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("earth orbit using verlet method")
    plt.axis("equal")
    plt.show()
