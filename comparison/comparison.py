import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def binomial_tail_bound_max(m, k, delta_pc):
    """
    max p, s.t. Bin(m, k, p) >= delta_pc。
    """
    p_values = np.linspace(1, 0, 1000)
    for p in p_values:
        if binom.cdf(k, m, p) >= delta_pc:
            return p
    return 0.0

def binomial_tail_bound_min(m, k, delta_pc):
    """
    min p, s.t. 1 - Bin(m, k, p) >= delta_pc。
    """
    p_values = np.linspace(0, 1, 1000)
    for p in p_values:
        if 1 - binom.cdf(k, m, p) >= delta_pc:
            return p
    return 1.0

m = 100  # number of the training samples
delta = 0.05
c_S_hat = np.arange(0, m + 1)  # number of the empirical errors

# assume P(c) = 1/N
N = 20  # number of the classifiers
P_c = np.ones(len(c_S_hat)) / N  # prior probability of the interested classifier (1/N) for the current c_S_hat

# without relaxation
binomial_bound_upper = np.array([binomial_tail_bound_max(m, k, delta * P_c[i]) for i, k in enumerate(c_S_hat)])
binomial_bound_lower = np.array([binomial_tail_bound_min(m, k, delta * P_c[i]) for i, k in enumerate(c_S_hat)])

# Chernoff Bound relaxation
chernoff_bound_upper = c_S_hat / m + np.sqrt((np.log(1/P_c) + np.log(1/delta)) / (2*m))
chernoff_bound_lower = c_S_hat / m - np.sqrt((np.log(1/P_c) + np.log(1/delta)) / (2*m))

plt.figure(figsize=(8, 6))
plt.plot(c_S_hat / m, binomial_bound_upper, label="Binomial Tail Inversion Upper Bound", linestyle='-', color='blue')
plt.plot(c_S_hat / m, binomial_bound_lower, label="Binomial Tail Inversion Lower Bound", linestyle='-', color='cyan')

plt.plot(c_S_hat / m, chernoff_bound_upper, label="Chernoff Upper Bound", linestyle='--', color='red')
plt.plot(c_S_hat / m, chernoff_bound_lower, label="Chernoff Lower Bound", linestyle='--', color='orange')

plt.xlabel("Empirical Error Rate (c_S_hat / m)")
plt.ylabel("Error Bound")
plt.title("Comparison of Upper and Lower Bounds with Relaxation")
plt.legend()
plt.grid(True)
plt.show()
