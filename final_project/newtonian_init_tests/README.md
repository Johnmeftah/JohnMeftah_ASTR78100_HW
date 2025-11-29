# Newtonian Binary Orbit Simulation 

A clean, minimal numerical sim of a two-body Newtonian binary using the **relative vector** in the **center of mass frame**.  
This is the baseline step before adding radiation-reaction and full inspiral physics.

---

##  What This Script Does
- Integrates the orbit using a **leapfrog (velocity-Verlet)** integrator  
- Sets correct **circular-orbit initial conditions**  
- Supports:
  - **DEBUG mode** – short run, plots, full diagnostics  
  - **PROD mode** – long run, no plots, cluster-friendly  
- Saves data as **NPZ**, **CSV**, or both  
- Offers an **interactive setup** 
- Tracks:
  - radius variation  
  - energy drift  
  - orbit stability  

---

##  Why This Step Matters
This run is the sanity check.  
Before adding any GR effects, we confirm:
- the Newtonian orbit stays circular  
- leapfrog conserves energy well  
- our setup and ICs are correct

Once this behaves perfectly, we move on to radiation-reaction and inspiral.