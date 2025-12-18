import numpy as np
import matplotlib.pyplot as plt


def generate_data_with_x_error(N, m, b, sigma_x, sigma_y):
    # generating true x values
    x_true = np.random.uniform(0, 1, N)

    # generating observed x values with gaussian x-noise
    x_obs = x_true + np.random.normal(0, sigma_x, N)

    # generating observed y values using true x plus gaussian y-noise
    y_obs = m * x_true + b + np.random.normal(0, sigma_y, N)

    return x_obs, y_obs


def york_fit(x, y, sigma_x, sigma_y, tol=1e-12, max_iter=200):
    """
    fitting y = m x + b with constant, uncorrelated errors in x and y.

    returning m, b, sigma_m, sigma_b.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # setting inverse variances
    w_x = 1 / (sigma_x ** 2)
    w_y = 1 / (sigma_y ** 2)

    # getting an initial slope guess using ordinary least squares
    m = np.polyfit(x, y, 1)[0]

    for _ in range(max_iter):
        # computing effective weights for the current slope
        w = (w_x * w_y) / (w_x + (m ** 2) * w_y)

        # computing weighted means
        x_bar = np.sum(w * x) / np.sum(w)
        y_bar = np.sum(w * y) / np.sum(w)

        # computing centered variables
        U = x - x_bar
        V = y - y_bar

        # computing beta values
        beta = w * (w_y * U + m * w_x * V) / (w_x + (m ** 2) * w_y)

        # updating the slope
        m_new = np.sum(beta * V) / np.sum(beta * U)

        if abs(m_new - m) < tol:
            m = m_new
            break

        m = m_new

    # computing intercept
    b = y_bar - m * x_bar

    # estimating parameter uncertainties
    w = (w_x * w_y) / (w_x + (m ** 2) * w_y)
    x_bar = np.sum(w * x) / np.sum(w)
    U = x - x_bar

    sigma_m = np.sqrt(1 / np.sum(w * (U ** 2)))
    sigma_b = np.sqrt((1 / np.sum(w)) + (x_bar ** 2) * (sigma_m ** 2))

    return m, b, sigma_m, sigma_b


print("\nline fit with x-error")

m_true = 1
b_true = 0

sigma_x = 0.2
sigma_y = 0.2

N = 50
n_runs = 30

m_list = []
b_list = []
sm_list = []
sb_list = []

m_poly_list = []
b_poly_list = []

for _ in range(n_runs):
    x_obs, y_obs = generate_data_with_x_error(N, m_true, b_true, sigma_x, sigma_y)

    m_fit, b_fit, sm, sb = york_fit(x_obs, y_obs, sigma_x, sigma_y)

    p_poly = np.polyfit(x_obs, y_obs, 1)
    m_poly = p_poly[0]
    b_poly = p_poly[1]

    m_list.append(m_fit)
    b_list.append(b_fit)
    sm_list.append(sm)
    sb_list.append(sb)

    m_poly_list.append(m_poly)
    b_poly_list.append(b_poly)

m_list = np.array(m_list, dtype=float)
b_list = np.array(b_list, dtype=float)
sm_list = np.array(sm_list, dtype=float)
sb_list = np.array(sb_list, dtype=float)

m_poly_list = np.array(m_poly_list, dtype=float)
b_poly_list = np.array(b_poly_list, dtype=float)

print(f"\nruns = {n_runs}")
print(f"mean(m_fit) = {np.mean(m_list)}")
print(f"std(m_fit)  = {np.std(m_list, ddof=1)}")
print(f"mean(sigma_m_est) = {np.mean(sm_list)}")

print(f"\nmean(b_fit) = {np.mean(b_list)}")
print(f"std(b_fit)  = {np.std(b_list, ddof=1)}")
print(f"mean(sigma_b_est) = {np.mean(sb_list)}")

print(f"\npolyfit baseline (ignoring x-error)")
print(f"mean(m_poly) = {np.mean(m_poly_list)}")
print(f"mean(b_poly) = {np.mean(b_poly_list)}\n")

plt.figure()
plt.hist(m_list, bins=12, alpha=0.8, label="york fit")
plt.axvline(m_true, linestyle="--", label="true m")
plt.xlabel("m")
plt.ylabel("count")
plt.title("distribution of slope with x-error")
plt.legend()
plt.show()

plt.figure()
plt.hist(b_list, bins=12, alpha=0.8, label="york fit")
plt.axvline(b_true, linestyle="--", label="true b")
plt.xlabel("b")
plt.ylabel("count")
plt.title("distribution of intercept with x-error")
plt.legend()
plt.show()
