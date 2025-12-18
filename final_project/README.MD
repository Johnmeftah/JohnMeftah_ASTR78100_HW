# Black Hole Binary Inspiral Simulation

python simulation of two black holes spiraling into each other using post-newtonian dynamics. simulates the inspiral phase like what LIGO detects.

## quick start

basic run (GW150914-like):
```bash
python bh_simulation.py --m1 36 --m2 29 --r0 5000 --extend-past-isco
```

interactive mode (enter all parameters manually):
```bash
python bh_simulation.py --custom
```

validate against peters formula:
```bash
python bh_simulation.py --test
```

## examples

```bash
# massive black holes
python bh_simulation.py --m1 80 --m2 60 --r0 8000 --extend-past-isco

# equal mass
python bh_simulation.py --m1 30 --m2 30 --r0 4000 --extend-past-isco

# eccentric orbit
python bh_simulation.py --m1 36 --m2 29 --r0 5000 -e 0.5 --extend-past-isco
```

## outputs

- CSV file with trajectory, velocities, GW frequency, strain
- diagnostic plots (orbit, separation decay, chirp, energy loss)
- waveform plots (h+ and h×)

## key parameters

- `--m1`, `--m2` - masses in solar masses
- `--r0` - initial separation in km
- `-e` - eccentricity (0 = circular)
- `--extend-past-isco` - continue past ISCO to horizon contact
- `--custom` - interactive mode

run `python bh_simulation.py --help` for all options.

## physics

uses post-newtonian approximations (0PN, 1PN, 2PN, 2.5PN). the 2.5PN radiation reaction term causes the inspiral. stops at ISCO (r = 6GM/c²) unless you use `--extend-past-isco`.

based on: peters & mathews (1963-64), burke & thorne (1970), kidder (1995), blanchet (2014).


## credits

John Meftah  
CUNY Graduate Center  
December 2025
