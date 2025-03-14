import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

m = 100  # number of training samples
delta = 0.05
p_real = np.linspace(0, 0.5, 100)  # range of true error rate

# without relaxation
binomial_bound = np.array([binom.ppf(1 - delta, m, p) / m for p in p_real])

# Chernoff Bound Relaxation
chernoff_bound = p_real + np.sqrt(np.log(1/delta) / (2*m))

# plot error bound
plt.figure(figsize=(8, 6))
plt.plot(p_real, binomial_bound, label="Binomial Bound (No Relaxation)", linestyle='-', color='blue')
plt.plot(p_real, chernoff_bound, label="Chernoff Bound (Relaxation)", linestyle='--', color='red')

plt.xlabel("True Error Rate (p)")
plt.ylabel("Error Bound")
plt.title("Comparison of Relaxation Before and After")
plt.legend()
plt.grid(True)
plt.show()
