import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def f_true(x):
    # defining the true function
    return np.sin(x / (2 * np.pi))


def sin_model(x, a, c):
    # defining a simple sin model with amplitude and offset
    return a * np.sin(x / (2 * np.pi)) + c


def chi2(y_obs, y_model, sigma_y):
    # computing chi^2 assuming constant sigma_y
    return np.sum(((y_obs - y_model) / sigma_y) ** 2)


print("\nnonlinear fit comparison")

sigma_y = 0.2
N = 50

# generating x values in [0, 1]
x = np.random.uniform(0, 1, N)

# generating noisy y values
y = f_true(x) + np.random.normal(0, sigma_y, N)

# fitting a line with polyfit
p1 = np.polyfit(x, y, 1)
m1 = p1[0]
b1 = p1[1]

# fitting a quadratic with polyfit
p2 = np.polyfit(x, y, 2)
a2 = p2[0]
b2 = p2[1]
c2 = p2[2]

# fitting a sin model with curve_fit
p0 = [1, 0]
popt, pcov = curve_fit(sin_model, x, y, p0=p0)
a_fit = popt[0]
c_fit = popt[1]

# evaluating models on the data points
y_line = m1 * x + b1
y_quad = a2 * x ** 2 + b2 * x + c2
y_sin = sin_model(x, a_fit, c_fit)

# computing chi^2 values
chi2_line = chi2(y, y_line, sigma_y)
chi2_quad = chi2(y, y_quad, sigma_y)
chi2_sin = chi2(y, y_sin, sigma_y)

print(f"\nchi^2 values (sigma_y = {sigma_y})")
print(f"line      chi^2 = {chi2_line}")
print(f"quadratic chi^2 = {chi2_quad}")
print(f"sin       chi^2 = {chi2_sin}\n")

# plotting the data and fitted curves
x_plot = np.linspace(0, 1, 400)

y_line_plot = m1 * x_plot + b1
y_quad_plot = a2 * x_plot ** 2 + b2 * x_plot + c2
y_sin_plot = sin_model(x_plot, a_fit, c_fit)

plt.figure()
plt.scatter(x, y, s=25, label="data")
plt.plot(x_plot, y_line_plot, label="line fit")
plt.plot(x_plot, y_quad_plot, label="quadratic fit")
plt.plot(x_plot, y_sin_plot, label="sin fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("line vs quadratic vs sin fit")
plt.legend()
plt.show()
