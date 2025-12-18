import numpy as np
import matplotlib.pyplot as plt


def generate_data(N, m, b, sigma):
    x = np.random.uniform(0, 1, N)
    y_true = m * x + b
    y_obs = y_true + np.random.normal(0, sigma, N)
    return x, y_obs


def bootstrap_fit(x, y, n_boot):
    N = len(x)
    m_list = []
    b_list = []

    for _ in range(n_boot):
        idx = np.random.randint(0, N, N)
        x_boot = x[idx]
        y_boot = y[idx]

        p = np.polyfit(x_boot, y_boot, 1)
        m_list.append(p[0])
        b_list.append(p[1])

    return np.array(m_list), np.array(b_list)


print("\nbootstrap resampling with polyfit")

m_true = 1
b_true = 0
sigma = 0.2
N = 50
n_boot = 1000

x, y = generate_data(N, m_true, b_true, sigma)

m_boot, b_boot = bootstrap_fit(x, y, n_boot)

print(f"\nbootstrap results ({n_boot} resamples)")
print(f"std(m_boot) = {np.std(m_boot, ddof=1)}")
print(f"std(b_boot) = {np.std(b_boot, ddof=1)}\n")

plt.figure()
plt.hist(m_boot, bins=20, alpha=0.8)
plt.axvline(m_true, linestyle="--")
plt.xlabel("m")
plt.ylabel("count")
plt.title("bootstrap distribution of slope")
plt.show()

plt.figure()
plt.hist(b_boot, bins=20, alpha=0.8)
plt.axvline(b_true, linestyle="--")
plt.xlabel("b")
plt.ylabel("count")
plt.title("bootstrap distribution of intercept")
plt.show()
