import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def parse_float_list(text):
    # parsing a comma-separated list of floats
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def run_interpolation(x_data, y_data, n_new):
    """
    computing linear and cubic interpolations using the suggested python methods.

    using numpy.interp for linear interpolation.
    using scipy.interpolate.interp1d with kind='cubic' for cubic interpolation.
    """

    # converting inputs to numpy arrays
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)

    # stopping if lengths do not match
    if len(x) != len(y):
        print("x and y must have the same number of values.")
        return

    # stopping if not enough points exist for cubic interpolation
    if len(x) < 4:
        print("need at least 4 points for cubic interpolation.")
        return

    # sorting data by x (interp requires increasing x)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # stopping if x has duplicates (interp1d will break)
    if np.any(np.diff(x) == 0):
        print("x values must be strictly increasing (no duplicates).")
        return

    # building the new x grid inside the data range
    x_new = np.linspace(x.min(), x.max(), int(n_new))

    # computing linear interpolation using numpy
    y_linear = np.interp(x_new, x, y)

    # computing cubic interpolation using scipy
    cubic_interp = interp1d(x, y, kind="cubic")
    y_cubic = cubic_interp(x_new)

    # plotting results
    plt.figure()
    plt.plot(x_new, y_linear, label="linear (numpy.interp)")
    plt.plot(x_new, y_cubic, label="cubic (scipy interp1d)")
    plt.plot(x, y, "o", label="data points")

    # labeling the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("linear vs cubic interpolation")
    plt.legend()
    plt.show()


# running the script when executed directly
if __name__ == "__main__":
    # printing a clean header
    print("\ninterpolation demo")
    print("using numpy.interp (linear) and scipy interp1d(kind='cubic')\n")

    # asking for x values
    x_text = input("enter x values (comma-separated, default 0,1,2,3): ").strip()
    x_vals = parse_float_list(x_text) if x_text else [0.0, 1.0, 2.0, 3.0]

    # asking for y values
    y_text = input("enter y values (comma-separated, default 0.4,0.55,0.95,0.1): ").strip()
    y_vals = parse_float_list(y_text) if y_text else [0.4, 0.55, 0.95, 0.1]

    # asking for how many new points to evaluate
    n_text = input("enter number of interpolation points (default 200): ").strip()
    n_new = int(n_text) if n_text else 200

    run_interpolation(x_vals, y_vals, n_new)
