import numpy as np
import matplotlib.pyplot as plt

print("\ngenerative model: straight line with noise")

m = 1
b = 0
sigma = 0.2
N = 50

x = np.random.uniform(0, 1, N)
y_true = m * x + b
y_obs = y_true + np.random.normal(0, sigma, N)

x_line = np.linspace(0, 1, 200)
y_line = m * x_line + b

plt.figure()
plt.plot(x_line, y_line, label="true model y = x")
plt.scatter(x, y_obs, s=25, label="generated data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
