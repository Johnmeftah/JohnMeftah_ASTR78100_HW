import numpy as np
import matplotlib.pyplot as plt


def generate_data(N, m, b, sigma):
    # generating synthetic line data with gaussian noise
    x = np.random.uniform(0, 1, N)
    y_true = m * x + b
    y_toggle = y_true + np.random.normal(0, sigma, N)
    return x, y_toggle


def fit_line_polyfit(x, y):
    # fitting y = m x + b using polyfit and returning fit and covariance
    p, cov = np.polyfit(x, y, 1, cov=True)
    m_fit = p[0]
    b_fit = p[1]
    return m_fit, b_fit, cov


print("\npolyfit test on generated data")

m_true = 1
b_true = 0
sigma = 0.2
N = 50
n_runs = 30

dm_list = []
db_list = []

sm_list = []
sb_list = []

for _ in range(n_runs):
    x, y = generate_data(N, m_true, b_true, sigma)
    m_fit, b_fit, cov = fit_line_polyfit(x, y)

    dm = m_fit - m_true
    db = b_fit - b_true

    # pulling 1-sigma uncertainties from the covariance diagonal
    sm = np.sqrt(cov[0, 0])
    sb = np.sqrt(cov[1, 1])

    dm_list.append(dm)
    db_list.append(db)

    sm_list.append(sm)
    sb_list.append(sb)

dm_list = np.array(dm_list, dtype=float)
db_list = np.array(db_list, dtype=float)

sm_list = np.array(sm_list, dtype=float)
sb_list = np.array(sb_list, dtype=float)

print(f"\nruns = {n_runs}")
print(f"mean(dm) = {np.mean(dm_list)}")
print(f"std(dm)  = {np.std(dm_list, ddof=1)}")
print(f"mean(polyfit sigma_m) = {np.mean(sm_list)}")

print(f"\nmean(db) = {np.mean(db_list)}")
print(f"std(db)  = {np.std(db_list, ddof=1)}")
print(f"mean(polyfit sigma_b) = {np.mean(sb_list)}\n")

plt.figure()
plt.hist(dm_list, bins=10, alpha=0.8)
plt.xlabel("m_fit - m_true")
plt.ylabel("count")
plt.title("distribution of slope differences")
plt.show()

plt.figure()
plt.hist(db_list, bins=10, alpha=0.8)
plt.xlabel("b_fit - b_true")
plt.ylabel("count")
plt.title("distribution of intercept differences")
plt.show()
