import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def estimate_Pc(weights, sigma):
    # approximate P(c), assume weights follows Gaussian.
    mean_weight = np.mean(weights, axis=1)  # calculate the mean of the weights
    P_c = norm.pdf(mean_weight, loc=0, scale=sigma)  # PDF of P(c)
    return P_c

# set the sigma
sigma = 0.1  # standard deviation of the weights
num_samples = 10000  # sample times
d = 1284  # number of the weights

# generate num_samples weights
weights = np.random.randn(num_samples, d) * sigma  

# calculate P(c) with samples
P_c = estimate_Pc(weights, sigma)

plt.hist(P_c, bins=50, density=True)
plt.xlabel("P(c) Values")
plt.ylabel("Density")
plt.title("Estimated P(c) Distribution for Neural Network")
plt.show()
