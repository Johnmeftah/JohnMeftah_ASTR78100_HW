## Key Points

- The rod creates a non central gravity field, which instantly exposes weak integrators
- Included methods: RK4, Leapfrog, Velocity Verlet, Midpoint, Bulirsch Stoer
- Leapfrog and Verlet are symplectic. They keep energy tight and are built for orbits
- RK4 looks smooth at first but leaks energy and fails for long term orbital motion
- Uses Plotly for interactive visual output

---

## How to Run

Run one method:


python rod_orbit.py RK4
python rod_orbit.py leapfrog
python rod_orbit.py verlet
python rod_orbit.py midpoint
python rod_orbit.py BS


Run energy comparison for all methods:


python rod_orbit.py energy


---

## Tips

- If you care about long term accuracy, start with leapfrog or verlet  
- RK4 is fine for short runs, but do not trust it for orbital physics  
- If the orbit blows up, your time step is probably too big. Lower it and try again  
- Use the energy mode to see very clearly which methods you should never use for orbits  

