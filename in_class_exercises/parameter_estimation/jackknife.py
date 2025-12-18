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


def jackknife_fit(x, y):
    N = len(x)
    m_list = []
    b_list = []

    for i in range(N):
        x_j = np.delete(x, i)
        y_j = np.delete(y, i)

        p = np.polyfit(x_j, y_j, 1)
        m_list.append(p[0])
        b_list.append(p[1])

    return np.array(m_list), np.array(b_list)


def jackknife_std(theta_list):
    N = len(theta_list)
    theta_bar = np.mean(theta_list)
    var = (N - 1) / N * np.sum((theta_list - theta_bar) ** 2)
    return math.sqrt(var)


print("\njackknife vs bootstrap")

m_true = 1
b_true = 0
sigma = 0.2
N = 50
n_boot = 1000

x, y = generate_data(N, m_true, b_true, sigma)

p_full = np.polyfit(x, y, 1)
m_full = p_full[0]
b_full = p_full[1]

m_boot, b_boot = bootstrap_fit(x, y, n_boot)
m_jack, b_jack = jackknife_fit(x, y)

m_boot_std = np.std(m_boot, ddof=1)
b_boot_std = np.std(b_boot, ddof=1)

m_jack_mean = np.mean(m_jack)
b_jack_mean = np.mean(b_jack)

m_jack_var = (N - 1) / N * np.sum((m_jack - m_jack_mean) ** 2)
b_jack_var = (N - 1) / N * np.sum((b_jack - b_jack_mean) ** 2)

m_jack_std = np.sqrt(m_jack_var)
b_jack_std = np.sqrt(b_jack_var)

print(f"\nfull fit:")
print(f"m_full = {m_full}")
print(f"b_full = {b_full}")

print(f"\nbootstrap std:")
print(f"std(m) = {m_boot_std}")
print(f"std(b) = {b_boot_std}")

print(f"\njackknife std:")
print(f"std(m) = {m_jack_std}")
print(f"std(b) = {b_jack_std}\n")

plt.figure()
plt.hist(m_jack, bins=15, alpha=0.8)
plt.axvline(m_full, linestyle="--", label="full fit")
plt.xlabel("m")
plt.ylabel("count")
plt.title("jackknife distribution of slope")
plt.legend()
plt.show()

plt.figure()
plt.hist(b_jack, bins=15, alpha=0.8)
plt.axvline(b_full, linestyle="--", label="full fit")
plt.xlabel("b")
plt.ylabel("count")
plt.title("jackknife distribution of intercept")
plt.legend()
plt.show()
