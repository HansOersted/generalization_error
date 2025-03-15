from scipy.stats import multivariate_normal
import numpy as np

# the weight after training is c_star
c_star = np.random.randn(1284) * 0.1  # replace using the authentic weight

# calculate P(c^*), assuming that weight ~ N(0, sigma^2)
sigma = 0.5  # standard deviation of the weights
mean = np.zeros_like(c_star)  # mean of the weights
cov = np.eye(len(c_star)) * sigma**2  # covariance matrix of the weights

P_c_star = multivariate_normal.pdf(c_star, mean=mean, cov=cov)

print("c_star =", c_star)
print("P(c*) =", P_c_star)
