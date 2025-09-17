# HW #2 (part 1): calculating the integral

import matplotlib.pyplot as plt

# approximating Taylor series
# per professor, make sure to not use (exp) to define a function when using packages like math, use f(x) for ex
def exp(z, terms=20):  
    s = 1  # 1st term in the series
    term = 1
    for n in range(1, terms):   
        term *= z / n           # each new term = (previous term * z/n)
        s += term
    return s

def f(t):  # defining the function we're integrating
    return exp(-t*t)

def trapz(a, b, n):  # trapezoidal rule
    h = (b - a) / n                 # step size
    s = 0.5 * (f(a) + f(b))         # initial sum with half weights at endpoints
    for k in range(1, n):
        s += f(a + k*h)
    return h * s

# main program starts here 
if __name__ == "__main__":
    print("\n=== HW #2 (Part 1) ===")
    print("Numerical evaluation of E(x) = ∫(0 -> x) {e^(-t^2)} dt")
    print("Using the trapezoidal rule with 1000 slices.\n")
    print("   x       E(x)")
    print(" ----------------")

    xs = []   # storing x values
    Es = []   # storing E(x) values

    for k in range(31):       # k = 0 --> 30 
        x = k * 0.1
        val = trapz(0, x, 1000)
        print(f"{x:5.2f}   {val:8.2f}")
        xs.append(x)
        Es.append(val)

# part (2): plotting
    plt.plot(xs, Es, marker='o', label="Numerical E(x)")
    plt.xlabel("x")
    plt.ylabel("E(x)")
    plt.title("Numerical evaluation of E(x) = ∫(0 -> x) {e^(-t^2)} dt")
    plt.legend()
    plt.show()

