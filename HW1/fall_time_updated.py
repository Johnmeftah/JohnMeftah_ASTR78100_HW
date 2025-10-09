import argparse
import sys

# planet gravities
planet_gravity = {
    'Mercury': 3.59, 'Venus': 8.87, 'Earth': 9.81, 'Moon': 1.62, 'Mars': 3.77,
    'Jupiter': 25.95, 'Saturn': 11.08, 'Uranus': 10.67, 'Neptune': 14.07, 'Pluto': 0.42,
}

# custom parser with single error message.
# Professor, please note that this CustomParser clean error message style was generated with Chat GPT.

class CustomParser(argparse.ArgumentParser):
    def error(self, message):
        print("Error: please enter the name of a planet (case sensitive) followed by the height in meters.\n"
              "Tip: you can also use 'all' followed by a height to see results for all planets.")
        sys.exit(2)

parser = CustomParser(
    description=(
        "Calculate fall time from a height on a planet. "
        "Usage examples:\n"
        "  python fall_time.py Earth 20   -> time on Earth from 20 m\n"

        "  python fall_time.py all 20     -> times on ALL planets from 20 m\n\n"
        "- Air resistance is neglected in all cases."
    )
)

parser.add_argument("planet", choices=list(planet_gravity.keys()) + ["all"],
                    help="Planet name (case sensitive) or 'all' for all planets")
parser.add_argument("height", type=float,
                    help="Height in meters")

args = parser.parse_args()

# time calculations
if args.planet == "all":
    print(f"Fall times from {args.height:.2f} m on all planets:")
    for planet, g in planet_gravity.items():
        t = ((2 * args.height) / g) ** 0.5
        print(f"  {planet}: {t:.2f} s")
else:
    g = planet_gravity[args.planet]
    t = ((2 * args.height) / g) ** 0.5
    print(f"On {args.planet}, an object falling from {args.height:.2f} m takes {t:.2f} s.")
