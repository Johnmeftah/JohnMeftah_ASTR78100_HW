import argparse
import sys

# parse args 
p = argparse.ArgumentParser(
    description="""Computing Earth to L1 distance using Newton's method.

Usage examples:
1- Use default values:
   L1_distance_newton.py
   (Default values: --guess 0.85 --tol 1e-12 --max-iter 50)

2- Custom values:
   L1_distance_newton.py --guess <value> --tol <value>
   Ex: script_name.py --guess 0.84 --tol 1e-10

3- Custom iterations:
   L1_distance_newton.py --guess 0.90 --tol 1e-9 --max-iter 100

4- Sun–Earth system:
   L1_distance_newton.py --system sun-earth
   Computes the L1 point between the Sun and Earth

5- Schematic line plot:
   L1_distance_newton.pyy --plot-line
   Shows Earth, L1, and Moon along a 1D line with distances
""",
    formatter_class=argparse.RawTextHelpFormatter # I used AI for this line to keep the help msg neat
)

# pick a system
p.add_argument(
    "--system",
    choices=["earth-moon", "sun-earth"],
    default="earth-moon",
    help="Pick the 2-body system to solve (default: earth-moon)."
)

p.add_argument("--guess", type=float, default=0.85,
               help="Initial guess as a FRACTION of R (0<guess<1). Default: 0.85")
p.add_argument("--tol", type=float, default=1e-12,
               help="Convergence tolerance in meters. Default: 1e-12")
p.add_argument("--max-iter", type=int, default=50,
               help="Max Newton iterations. Default: 50")

# schematic plotting
p.add_argument("--plot-line", action="store_true",
               help="Show a simple line plot with Earth, L1, and Moon/Sun markers and labeled distances.")

args = p.parse_args()

if not (0 < args.guess < 1): # keeping initial guess fraction between 0 and 1
    sys.exit("Error: --guess must be between 0 and 1 (fraction of R).")
if args.tol <= 0:
    sys.exit("Error: --tol must be positive.")
if args.max_iter <= 0:
    sys.exit("Error: --max-iter must be positive.")

# constants 
G = 6.67430e-11 # gravitational const

if args.system == "earth-moon":
    M = 5.9722e24 # Earth's mass (in kg)     
    m = 7.348e22  # Moon's mass (in kg)          
    R = 3.844e8   # Earth to Moon distance (in m)
elif args.system == "sun-earth":
    M = 1.9885e30      # Sun's mass (in kg)
    m = 5.9722e24      # Earth's mass (in kg)
    R = 1.495978707e11 # 1 AU (in m)

omega = (G*(M+m)/R**3)**0.5 # angular speed 

# using Newton's method 
r = args.guess * R  # initial guess for Earth to L1 distance 

for i in range(args.max_iter):
    
    fr  = G*M/r**2 - G*m/(R - r)**2 - omega**2 * r # function 

    dfr = -2*G*M/r**3 - 2*G*m/(R - r)**3 - omega**2 # derivative
    if dfr == 0.0:
        sys.exit("Error: derivative became zero, try a different --guess.")
    r_new = r - fr/dfr
    if abs(r_new - r) < args.tol:
        r = r_new
        break
    r = r_new
else:
    sys.exit("Error: did not converge within --max-iter iterations.")

other_body = "Moon" if args.system == "earth-moon" else "Sun"

print(f"Earth to L1 distance r  = {r:.4e} m")
print(f"As fraction of R       = {r/R:.4f}")
print(f"Distance from {other_body:<9} = {R - r:.4e} m")
print(f"Iterations used        = {i+1}")

# schematic plot ... I used AI's help plotting
# AI lines begin:

if args.plot_line:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        sys.exit(f"Error: matplotlib is required for --plot-line but could not be imported: {e}")

    # Convert to km for cleaner labels
    R_km = R / 1e3
    r_km = r / 1e3
    Rm_r_km = (R - r) / 1e3
    
# AI lines end
    
    plt.figure()
    plt.plot([0, R_km], [0, 0], color="black")  # baseline
    plt.scatter([0], [0], label="Earth", zorder=3, color="blue")
    plt.scatter([r_km], [0], label="L1", zorder=3, color="red")
    plt.scatter([R_km], [0], label=other_body, zorder=3, color="green")

    # labels
    plt.text(0, 0.02, "Earth", ha="center", va="bottom")
    plt.text(r_km, 0.02, f"L1\n{r_km:.0f} km", ha="center", va="bottom")
    plt.text(R_km, 0.02, other_body, ha="center", va="bottom")

# AI lines begin:
    # Distance annotations
    plt.annotate(f"{r_km:.0f} km",
                 xy=(0, 0), xytext=(r_km/2, -0.03),
                 arrowprops=dict(arrowstyle="-"), ha="center", va="top")
    plt.annotate(f"{Rm_r_km:.0f} km",
                 xy=(r_km, 0), xytext=(r_km + (R_km - r_km)/2, -0.03),
                 arrowprops=dict(arrowstyle="-"), ha="center", va="top")
# AI lines end
    
    plt.yticks([])  
    plt.xlabel("Distance along Earth–other-body line (km)")
    plt.title(f"L1 position for {args.system}  (R approx {R_km:.0f} km)")
    plt.legend()
    plt.tight_layout()
    plt.show()

