import argparse

parser = argparse.ArgumentParser(description="calculating objects fall time on Earth") 
parser.add_argument("height", type=float, help="enter height in meters")
parser.add_argument("--g", type=float, default=9.8, help="gravity, default=9.8")
args = parser.parse_args()

# defining time
t = ((2 * args.height) / args.g) ** 0.5

print(f"Object falling from {args.height} meters will take {t:.2f} seconds on Earth")


