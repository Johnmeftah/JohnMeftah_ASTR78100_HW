import math
import numpy as np
import matplotlib.pyplot as plt


def load_sunspots():
    """
    loading sunspot data from sunspots.txt in the current directory.

    assuming column 0 is month index and column 1 is sunspot number.
    """

    data = np.loadtxt("sunspots.txt")
    months = data[:, 0]
    spots = data[:, 1]
    return months, spots


def estimate_cycle_from_fft(spots):
    """
    estimating the dominant cycle using the FFT power spectrum.

    returning the peak index k_peak and the estimated period in months.
    """

    N = len(spots)

    # removing the mean to reduce the k=0 spike
    y = spots - np.mean(spots)

    # computing the FFT
    c = np.fft.rfft(y)

    # computing the power spectrum
    power = np.abs(c) ** 2

    # ignoring k = 0
    power[0] = 0.0

    # finding the peak
    k_peak = int(np.argmax(power))

    # computing the period in months (dt = 1 month per sample)
    period_months = N / k_peak if k_peak != 0 else float("inf")

    return k_peak, period_months, power


# running the script when executed directly
if __name__ == "__main__":
    # loading the data
    months, spots = load_sunspots()

    # plotting sunspots as a function of time
    plt.figure()
    plt.plot(months, spots)
    plt.xlabel("month index")
    plt.ylabel("sunspot number")
    plt.title("sunspots vs time")
    plt.show()

    # computing the dominant cycle from the FFT
    k_peak, period_months, power = estimate_cycle_from_fft(spots)

    # building k values for the rfft output
    k_vals = np.arange(len(power))

    # plotting the power spectrum
    plt.figure()
    plt.plot(k_vals, power)
    plt.xlabel("k")
    plt.ylabel("|c_k|^2")
    plt.title("power spectrum of sunspot signal")
    plt.show()

    # printing the peak and estimated period
    print("\nfourier analysis results")
    print(f"N = {len(spots)}")
    print(f"peak k ≈ {k_peak}")
    print(f"estimated period ≈ {period_months:.2f} months")
    print(f"estimated period ≈ {period_months/12.0:.2f} years\n")
