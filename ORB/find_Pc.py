from scipy.stats import multivariate_normal
import numpy as np

# 假设你训练后的神经网络最终权重为 c_star
c_star = np.random.randn(1000) * 0.1  # 这里只是示例，真实 weight 应该是你训练好的参数

# 计算 P(c^*)，假设所有 weight 服从 N(0, sigma^2)
sigma = 0.1  # 设定方差
mean = np.zeros_like(c_star)  # 均值假设为0
cov = np.eye(len(c_star)) * sigma**2  # 协方差矩阵假设为独立同分布

P_c_star = multivariate_normal.pdf(c_star, mean=mean, cov=cov)

print("P(c*) =", P_c_star)
