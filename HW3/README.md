# L1 Locator

This program finds the **L1 point**, the point between two objects (like Earth and the Moon, or the Sun and the Earth) where their gravity balances in just the right way.  
It uses **Newton’s method** to calculate the position and tells you:

- Distance from Earth to L1  
- Fraction of the total distance  
- Distance from the other body  

You can also add a simple plot that shows Earth, L1, and the other body lined up on one axis.

---

## How to Run

1- **Basic run (Earth–Moon system):**
L1_distance_newton.py

2- **Use custom values:**
L1_distance_newton.py --guess 0.84 --tol 1e-10 --max-iter 100

3- **Switch to the Sun–Earth system:**
L1_distance_newton.py --system sun-earth

4- **Add a simple plot:**
L1_distance_newton.py --plot-line



