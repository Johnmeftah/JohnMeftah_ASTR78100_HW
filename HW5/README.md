# Monte Carlo Photon Scattering

This script simulates how photons bounce around in a medium — either a flat slab or inside the Sun.  
It’s basically a random walk problem using simple physics.



## What It Does

- **Slab mode:** photons start at one side, scatter around, and either escape or bounce back.  
- **Sun mode:** photons start at the center of the Sun and move through a simple density model until they escape.  
- Optional **animation** shows photon paths live — green = escaped, red = reflected.



## What You Get

After the run, it prints:
- Number of photons that escaped vs. reflected  
- Average and median scatter count  
- For Sun mode: average escape time  



## Key Points

- Uses **Thomson scattering** (`σ_T = 6.652e-29 m²`)  
- Mean free path: `ℓ = 1 / (n_e * σ_T)`  
- All units are SI, except electron density (input in cm⁻³)  
- Animation runs with `matplotlib` — no extra dependencies needed  



## Tips

- Start with small photon counts if you’re testing  
- Real Sun densities are slow — reduce the scale for quick runs  
- Animation needs a display (won’t show in headless terminals)



