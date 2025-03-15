import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def estimate_Pc(weights, sigma):
    # approximate P(c), assume weights follow Gaussian.
    mean_weight = np.mean(weights, axis=1)  # calculate the mean of the weights
    P_c = norm.pdf(mean_weight, loc=0, scale=sigma)  # PDF of P(c)
    
    # Normalize P(c) to make the sum equal to 1
    P_c = P_c / np.sum(P_c) 
    
    return P_c

# Set parameters
sigma = 0.1  # standard deviation of the weights
num_samples = 10000  # number of sampled classifiers
d = 1284  # number of weights per classifier

# Generate num_samples weight vectors
weights = np.random.randn(num_samples, d) * sigma  

# Calculate P(c) with samples
P_c = estimate_Pc(weights, sigma)

# Plot histogram of P(c) WITHOUT density normalization
plt.hist(P_c, bins=50)  # density=False is default
plt.xlabel("P(c) Values")
plt.ylabel("Probability Mass")  # Change label to indicate sum(P_c) = 1
plt.title("Estimated P(c) Distribution for Neural Network (Normalized)")

# Verify that sum(P_c) ≈ 1
print(f"Sum of P(c): {np.sum(P_c)}")

plt.show()
