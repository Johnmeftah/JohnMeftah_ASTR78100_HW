import math, random, sys
import matplotlib.pyplot as plt

MAX_PLOT = 30   # how many photon tracks to draw
FPS = 30        

# consts
SIGMA_T   = 6.652e-29 # Thomson cross section
C_LIGHT   = 2.99792458e8 # speed of light (m/s)
R_SUN     = 6.9634e8 # solar radius (m)
ESCAPE_R  = 0.9 * R_SUN # where we say it escaped
MAX_SCAT  = 5_000_000 # safety cap so runs don't hang
DENSITY_SCALE = 1  # scales n_e in Sun mode (1 speeds up the run)

def print_help():
    
    msg = """
Monte Carlo Photon Scattering:

Simulating an isotropic photon scattering in a slab.

Usage:
  python photon_slab.py N WIDTH [SEED] [NE]
  python photon_slab.py animate N WIDTH [SEED] [NE]

Commands:
  N      Number of photons
  WIDTH  Slab width in meters
  SEED   Random seed (default 0)
  NE     Electron density in cm^-3 (default 1e20)

Examples:
  python photon_slab.py 200 1000
  python photon_slab.py animate 200 1000
  python photon_slab.py sun N [SEED] [SCALE]
  python sun.py sun 200          # physical density (very slow)
  python sun.py sun 200 0 1e-12  # scaled down density for fast testing
"""
    print(msg)

def print_stats(stats):
    # printing final summary 
    print(f"\n Sim  Results ")
    print(f"Photons launched: {stats['N']}")
    print(f"Transmitted: {stats['trans']} ({100*stats['trans']/stats['N']:.1f}%)")
    print(f"Reflected:   {stats['refl']} ({100*stats['refl']/stats['N']:.1f}%)")
    print(f"Scatterings per photon: mean={stats['avg']:.2f}, median={stats['med']}, 95th perc={stats['p95']}")


def run_sim(N, width, seed, ne_cm3):
    # deterministic runs if seed is given
    random.seed(seed)

    # Thomson cross section
    sigma_T = 6.652e-29

    # converting cm^-3 to  m^-3
    n_e = ne_cm3 * 1e6

    # mean free path 
    mfp = 1 / (n_e * sigma_T)

    # optical depth 
    tau = width / mfp

    # print setup info
    print(f"\n Sim  Parameters")
    print(f"Photons launched: {N}")
    print(f"Slab width: {width} m")
    print(f"Mean free path: {mfp:.3f} m")
    print(f"Optical depth:  {tau:.3f}")

    transmitted = 0
    reflected   = 0
    scatters_all = []
    paths = []

    # shooting N photons
    for i in range(N):
        z = 1e-9                      # starting just inside the slab
        mu = 2*random.random() - 1    # cos(theta) is about  U[-1,1]
        scatters = 0
        path = [z]

        # walking until the photon exits
        while True:
            # expo free path:
            step = -mfp * math.log(max(random.random(), 1e-16))

            # moving along z
            z_new = z + mu*step

            # top boundary - transmit
            if z_new >= width:
                transmitted += 1
                path.append(width)
                break

            # bottom boundary - reflect
            if z_new <= 0:
                reflected += 1
                path.append(0.0)
                break

            # still inside, updating state
            z = z_new
            scatters += 1
            path.append(z)

            # new direction after scatter (isotropic)
            mu = 2*random.random() - 1

            # cap to avoid long walks
            if scatters > 20000:
                break

        scatters_all.append(scatters)
        if i < MAX_PLOT:
            paths.append(path)

    # stats over scatter counts
    scatters_all.sort()
    avg = sum(scatters_all)/len(scatters_all) if scatters_all else 0.0
    med = scatters_all[len(scatters_all)//2]  if scatters_all else 0
    p95 = scatters_all[int(0.95*(len(scatters_all)-1))] if scatters_all else 0

    stats = dict(N=N, trans=transmitted, refl=reflected, avg=avg, med=med, p95=p95)
    return paths, stats
# I used chatGPT to help with the animation
# AI code begins:
def animate_manual(paths, width, stats):
    """Manual animation using plt.pause()."""
    # If nothing to draw, exit
    if not paths:
        return

    max_steps = max(len(p) for p in paths)

    # plotting
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Photon random walk in a slab")
    ax.set_xlabel("Photon ID")
    ax.set_ylabel("z (m)")
    ax.set_xlim(-1, len(paths))
    ax.set_ylim(-0.1*width, 1.1*width)
    ax.axhline(0,     color="black", lw=2)
    ax.axhline(width, color="black", lw=2)
    
    # One line + one dot per photon
    lines, dots = [], []
    for _ in range(len(paths)):
        (l,) = ax.plot([], [], lw=2)
        (d,) = ax.plot([], [], "o")
        lines.append(l); dots.append(d)

    # Iteration tracker
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    plt.ion()

    # Animate photon motion step by step
    for frame in range(max_steps):
        for i, p in enumerate(paths):
            n = min(frame+1, len(p))
            x_vals = [i]*n
            y_vals = p[:n]
            lines[i].set_data(x_vals, y_vals)
            dots[i].set_data([x_vals[-1]], [y_vals[-1]])
            if n == len(p):
                dots[i].set_color("green" if p[-1] >= width else "red")
        txt.set_text(f"Step: {frame}")
        fig.canvas.draw()
        plt.pause(1.0/FPS)

    plt.show()
#AI code ends

    # print results after window closes
    print_stats(stats)

# Sun mode 
def ne_cm3_bahcall(r):
    # electron density
    return 2.5e26 * math.exp(- r / (0.095 * R_SUN))

def one_photon_sun():
    # simulatimg one photon in radial geometry; return (escaped?, time_s, scatters)
    r = 0.0           # radius from center (m)
    s = 0.0           # total path traveled (m)
    scat = 0
    mu = 1.0          # start outward at the center

    while True:
        # local density (scaled) - mean free path
        ne_m3 = ne_cm3_bahcall(r) * 1e6 * DENSITY_SCALE
        if ne_m3 <= 0.0:
            return True, s / C_LIGHT, scat

        ell = 1.0 / (ne_m3 * SIGMA_T)

        # drawing expo step
        step = -ell * math.log(1.0 - random.random())

        # only the radial component matters here
        dr = mu * step
        r_next = r + dr

        # reached the escape radius
        if r_next >= ESCAPE_R:
            # clip to boundary 
            s += max(0.0, ESCAPE_R - r)
            return True, s / C_LIGHT, scat

        # hit the center -  reflect outward
        if r_next < 0.0:
            s += r
            r = 0.0
            mu = 1.0
            continue

        # interior scatter
        r = r_next
        s += abs(dr)
        scat += 1

        # safety cap 
        if scat >= MAX_SCAT:
            return False, s / C_LIGHT, scat

        # new direction relative to radial, isotropic
        mu = 2.0*random.random() - 1.0

def run_sun(N, seed=0):
    # run many photons in Sun mode and collect stats
    random.seed(seed)

    times = []
    scatters = []
    capped = 0

    for _ in range(N):
        escaped, t, k = one_photon_sun()
        if escaped:
            times.append(t)
        else:
            capped += 1
        scatters.append(k)

    # summaries
    times.sort(); scatters.sort()
    def pct(a, q):
        if not a: return 0.0
        i = max(0, min(len(a)-1, int(q*(len(a)-1))))
        return a[i]

    t_mean   = (sum(times)/len(times)) if times else 0.0
    t_median = times[len(times)//2] if times else 0.0
    t_p95    = pct(times, 0.95)

    avg = sum(scatters)/len(scatters) if scatters else 0.0
    med = scatters[len(scatters)//2] if scatters else 0
    p95 = int(pct(scatters, 0.95)) if scatters else 0

    # Sun info
    print(f"\n Sim  Results")
    print(f"Photons launched: {N}")
    print(f"Escape time t: mean={t_mean:.3e}s, median={t_median:.3e}s, 95th={t_p95:.3e}s")
    print(f"Scatterings per photon: mean={avg:.2f}, median={med}, 95th perc={p95}")
    if capped > 0:
        print(f"{capped} photon(s) hit MAX_SCAT={MAX_SCAT} and did not escape.")

# Needed AI help defining this function
# AI code begins:
def main():
    # help msgs
    if len(sys.argv)==1 or sys.argv[1] in ("--h","--help","-h"):
        print_help(); sys.exit(0)

    # Sun mode branch (kept simple)
    if sys.argv[1].lower() == "sun":
        if len(sys.argv) < 3:
            print_help(); sys.exit(1)
        N    = int(sys.argv[2])
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        # optional density scale
        global DENSITY_SCALE
        if len(sys.argv) > 4:
            DENSITY_SCALE = float(sys.argv[4])
# AI code ends
        print("\n Sun Model Parameters")
        print(f"Photons launched: {N}")
        print(f"Escape radius: {ESCAPE_R:.3e} m (0.9 R_sun)")
        print(f"Density scale: {DENSITY_SCALE:g}  (1.0 = physical)")
        run_sun(N, seed)
        return


    # slab modes (original flow) 
    animate_mode = (sys.argv[1].lower() == "animate")
    args = sys.argv[2:] if animate_mode else sys.argv[1:]

    # need N and WIDTH
    if len(args) < 2:
        print_help(); sys.exit(1)

    # parse args
    N     = int(args[0])
    width = float(args[1])
    seed  = int(args[2]) if len(args) > 2 else 0
    ne    = float(args[3]) if len(args) > 3 else 1e20  # cm^-3

    # running and either animate or just print stats
    paths, stats = run_sim(N, width, seed, ne)
    if animate_mode:
        animate_manual(paths, width, stats)
    else:
        print_stats(stats)

if __name__ == "__main__":
    main()
