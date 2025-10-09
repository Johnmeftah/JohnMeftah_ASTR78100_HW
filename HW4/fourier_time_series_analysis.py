import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# I used AI to help include the data in a way that makes it readable when the code is downloaded from GitHub
# AI code begins:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(SCRIPT_DIR, "data", "tic0011113164.fits")  # <- dataset path in repo
# AI code ends

# data
def _read_fits(filename):
    """loading times/fluxes using Ari's script."""
    if not os.path.exists(filename):
        sys.exit(f"Error: file '{filename}' not found in this folder.")
    with fits.open(filename) as hdul:
        times  = hdul[1].data['times']
        fluxes = hdul[1].data['fluxes']
        ferrs  = hdul[1].data['ferrs']
    return np.array(times, float), np.array(fluxes, float)

# light curve
def plot_light_curve(filename):
    """ploting flux versus time."""
    t, f = _read_fits(filename)
    plt.errorbar(t, f, fmt='.', ms=2, alpha=0.7)
    plt.xlabel("Time"); plt.ylabel("Flux"); plt.title(f"Light Curve – {os.path.basename(filename)}")
    plt.tight_layout(); plt.show()

# choosing a continuous observing block 
def _pick_densest_epoch(t):
    """
    breaking the data wherever there’s a big time gap (more than 5 times the usual spacing between points), and keep only the longest chunk.
    basically, the part with the most data points.”
    """
    t = np.asarray(t)
    if t.size <= 1:
        return 0, t.size
    dt = np.diff(t)
    med_dt = np.median(dt)
    if not (np.isfinite(med_dt) and med_dt > 0):
        return 0, t.size
    thr = 5.0 * med_dt
    cuts = np.where(dt > thr)[0] + 1
    edges = np.concatenate(([0], cuts, [t.size]))
    lengths = edges[1:] - edges[:-1]
    k = int(np.argmax(lengths))
    return int(edges[k]), int(edges[k + 1])

# Ari's DFT, thank you for sharing your code with us!  
def ari_dft(y):
    """
    computing one sided DFT coefficients c_k up to N//2 using
    c_k = sum_n (y_n - mean(y)) * exp(-2πikn/N). Returns (c, mean).
    """
    mu = np.mean(y)
    y0 = y - mu
    N = y0.size
    kmax = N // 2 + 1
    n = np.arange(N)
    c = np.zeros(kmax, dtype=complex)
    for k in range(kmax):
        c[k] = np.sum(y0 * np.exp(-2j * np.pi * k * n / N))
    return c, mu

def dft_power_report(te, c, top_n=6):
    """printing top coefficients ordered by power |c_k|^2 and report freq/period. I think those are enough, no?"""
    power = np.abs(c)**2
    N = (len(c) - 1) * 2 if len(c) > 1 else 1
    dte = np.diff(te)
    ok = np.isfinite(dte)
    dt_med = np.median(dte[ok]) if np.any(ok) else np.nan
    if np.isfinite(dt_med) and dt_med > 0:
        freq = np.arange(len(c)) / (N * dt_med)
    else:
        freq = np.zeros(len(c))
    idx = np.argsort(power[1:])[::-1][:top_n] + 1 if len(c) > 1 else np.array([], int)
    print("# Top DFT coefficients (Ari method, power = |c_k|^2):")
    for j in idx:
        f0 = freq[j]
        period = (1.0 / f0) if (np.isfinite(f0) and f0 > 0) else np.nan
        print(f"  k={j:4d}  freq={f0:.6f} 1/time  period≈{period:.6f}  |c_k|^2={power[j]:.3e}")
    return idx, power, freq

def plot_power_k(c, title="Power Spectrum (Ari Method)"):
    """ploting power |c_k|^2 versus harmonic index k."""
    k = np.arange(len(c))
    plt.figure()
    plt.plot(k, np.abs(c)**2)
    plt.xlabel("k"); plt.ylabel(r"$|c_k|^2$"); plt.title(title)
    plt.tight_layout(); plt.show()

# filling small TESS gaps by linear interpolation. I didn't know we could do that!  
def interp_uniform(te, fe):
    """
    building a uniform time grid via median cadence; linearly interpolate flux.
    TESS is nearly uniform, so this barely changes the peaks.
    """
    te = np.asarray(te); fe = np.asarray(fe)
    if te.size < 3:
        return te, fe
    dts = np.diff(te)
    dt = np.median(dts)
    if not (np.isfinite(dt) and dt > 0):
        return te, fe
    M = int(round((te[-1] - te[0]) / dt)) + 1
    tu = te[0] + dt * np.arange(M)
    fu = np.interp(tu, te, fe)
    return tu, fu

# reconstructing using only the top-K coefficients 
def inverse_from_topK(c1, K, N, keep_dc=True):
    """
    keeping K largest |c_k| (k>=1), inverse transform,
    and return zero mean reconstruction y_rec0.
    """
    kmax = len(c1) - 1
    has_nyq = (N % 2 == 0) and (kmax == N // 2)

    C = np.zeros(N, dtype=complex)
    if keep_dc and len(c1) > 0:
        C[0] = c1[0]

    ks = np.arange(1, kmax + 1)
    if has_nyq:
        ks = ks[ks != (N // 2)]

    # picking top K by power
    powers = np.abs(c1[ks])**2
    order = np.argsort(powers)[::-1]
    keep = ks[order[:K]]

    for k in keep:
        C[k] = c1[k]
        C[-k] = np.conj(c1[k])

    n = np.arange(N)
    # outer product form kept 
    expo = np.exp(2j * np.pi * np.arange(N)[:, None] * n / N)
    y_rec0 = (1.0 / N) * np.real(np.sum(C[:, None] * expo, axis=0))
    return y_rec0

# phase fold helpers 
def _peak_period_from_dft(te, c):
    """geting fundamental period from strongest peak."""
    power = np.abs(c)**2
    N = (len(c)-1)*2 if len(c)>1 else 1
# I used AI for dte and dt, kept getting errors.
    dte = np.diff(te); ok = np.isfinite(dte) 
    dt = np.median(dte[ok]) if np.any(ok) else np.nan
    if not (np.isfinite(dt) and dt>0):
        return np.nan
    k = 1 + np.argmax(power[1:])           # strongest peak
    f0 = k / (N*dt)
    return (1.0/f0) if (f0 > 0) else np.nan

# used AI to help me define the phase fold.
# AI code begins:
def phase_fold(t, f, period, nbins=120):
    """Phase-fold data on 'period' and overlay binned median."""
    ph = (t % period) / period
    order = np.argsort(ph)
    bins = np.linspace(0, 1, nbins+1)
    idx = np.digitize(ph, bins) - 1
    xb = 0.5*(bins[:-1]+bins[1:])
    yb = np.array([np.median(f[idx==i]) if np.any(idx==i) else np.nan for i in range(nbins)])
# AI code ends
    plt.figure()
    plt.plot(ph[order], f[order], ".", ms=1.5, alpha=0.4, label="data")
    plt.plot(xb, yb, "-", lw=2, label="binned median")
    plt.xlabel("Phase"); plt.ylabel("Flux"); plt.title(f"Phase-folded (P≈{period:.6f})")
    plt.legend(); plt.tight_layout(); plt.show()

# spectral window (sampling function power)
# I used AI to help me define the window
# AI code begins:
def run_window(filename):
    """
    Plot the spectral window: the power spectrum of the sampling mask over a uniform grid.
    Reveals alias structure due to gaps and near uniform sampling.
    """
    t, _ = _read_fits(filename)
    i0, i1 = _pick_densest_epoch(t)
    te = t[i0:i1]
    if te.size < 32:
        sys.exit("Error: selected epoch is too short for spectral window.")
    # Build uniform grid based on median cadence and place 1's at observed indices
    dt = np.median(np.diff(te))
    if not (np.isfinite(dt) and dt > 0):
        sys.exit("Error: could not determine median cadence for spectral window.")
    M = int(round((te[-1] - te[0]) / dt)) + 1
    idx = np.rint((te - te[0]) / dt).astype(int)
    idx = idx[(idx >= 0) & (idx < M)]
    mask = np.zeros(M, dtype=float)
    np.add.at(mask, idx, 1.0)  # Mark (and count) observed samples on the grid
    c_win, _ = ari_dft(mask)   # Mean is removed inside ari_dft; DC suppressed
    plot_power_k(c_win, title="Spectral Window (Sampling Function Power)")
# AI code ends

# commands
def run_fourier(filename, top_n=6):
    """DFT on the densest epoch (Ari method, thank you again!) and power report."""
    t, f = _read_fits(filename)
    i0, i1 = _pick_densest_epoch(t)
    te, fe = t[i0:i1], f[i0:i1]
    c, _ = ari_dft(fe)
    plot_power_k(c)
    dft_power_report(te, c, top_n=top_n)

def run_fourier_interp(filename, top_n=6):
    """same as run_fourier, but first interpolate to a uniform grid."""
    t, f = _read_fits(filename)
    i0, i1 = _pick_densest_epoch(t)
    te, fe = t[i0:i1], f[i0:i1]
    tu, fu = interp_uniform(te, fe)
    c, _ = ari_dft(fu)
    plot_power_k(c, "Power Spectrum (Ari Method, Linear Interpolation)")
    dft_power_report(tu, c, top_n=top_n)

def run_reconstruct(filename, K=3):
    """reconstructing the densest epoch using only K strongest coefficients."""
    t, f = _read_fits(filename)
    i0, i1 = _pick_densest_epoch(t)
    te, fe = t[i0:i1], f[i0:i1]
    c, mu = ari_dft(fe)
    dft_power_report(te, c, top_n=max(K, 6))
    y = inverse_from_topK(c, K=K, N=fe.size, keep_dc=True) + mu
    plt.figure()
    plt.plot(te, fe, ".", ms=2, alpha=0.6, label="original")
    plt.plot(te, y,  "-", lw=1.5, label=f"reconstruction (K={K})")
    plt.xlabel("Time"); plt.ylabel("Flux"); plt.title("K-term Inverse DFT Reconstruction (Ari Method)")
    plt.legend(); plt.tight_layout(); plt.show()

def run_fold(filename):
    """folding densest epoch on DFT peak period and plot (gold standard sanity check)."""
    t, f = _read_fits(filename)
    i0, i1 = _pick_densest_epoch(t)
    te, fe = t[i0:i1], f[i0:i1]
    c, _ = ari_dft(fe)
    P = _peak_period_from_dft(te, c)
    phase_fold(te, fe, P, nbins=120)

# CLI 
def main():
    p = argparse.ArgumentParser(
        description=("To plot a light curve, run:\n"
                     "  python fourier_time_series_analysis.py light_curve\n\n"
                     "To compute a Fourier power spectrum, run:\n"
                     "  python fourier_time_series_analysis.py fourier\n\n"
                     "To compute a Fourier power spectrum after linear interpolation, run:\n"
                     "  python fourier_time_series_analysis.py fourier_interp\n\n"
                     "To reconstruct using K Fourier coefficients, run:\n"
                     "  python fourier_time_series_analysis.py reconstruct\n\n"
                     "To plot a phase folded light curve (auto period), run:\n"
                     "  python fourier_time_series_analysis.py fold\n\n"
                     "To plot the spectral window of the sampling, run:\n"
                     "  python fourier_time_series_analysis.py window"),
        formatter_class=argparse.RawTextHelpFormatter # this line was written by AI in previouse HW, I used it again and just wanted to clarify that. 
    )
    p.add_argument("plot_type", nargs="?", help="Type: light_curve, fourier, fourier_interp, reconstruct, fold, window")
    p.add_argument("--k", type=int, default=3, help="Coefficients for reconstruction (reconstruct)")
    a = p.parse_args()

    if not a.plot_type:
        print("Error: you must specify the type of plot (e.g. 'light_curve').")
        print("For help, run: python fourier_time_series_analysis.py --help\n"); sys.exit(1)

    if a.plot_type == "light_curve":
        plot_light_curve(data)
    elif a.plot_type == "fourier":
        run_fourier(data)
    elif a.plot_type == "fourier_interp":
        run_fourier_interp(data)
    elif a.plot_type == "reconstruct":
        run_reconstruct(data, K=a.k)
    elif a.plot_type == "fold":
        run_fold(data)
    elif a.plot_type == "window":
        run_window(data)
    else:
        sys.exit(f"Error: Unknown plot type '{a.plot_type}'. Use 'light_curve', 'fourier', 'fourier_interp', 'reconstruct', 'fold', or 'window'.")

if __name__ == "__main__":
    main()
