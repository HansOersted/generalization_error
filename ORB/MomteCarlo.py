import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def estimate_Pc(weights, sigma):
    """ 计算 P(c) 近似值，假设 weights 服从高斯分布 """
    mean_weight = np.mean(weights, axis=1)  # 计算每组权重的均值
    P_c = norm.pdf(mean_weight, loc=0, scale=sigma)  # 计算 P(c) 概率密度
    return P_c

# 设定神经网络权重的标准差
sigma = 0.1  
num_samples = 10000  # 采样 10000 组
d = 1282  # 参数个数

# 生成 10000 组神经网络权重
weights = np.random.randn(num_samples, d) * sigma  

# 计算 P(c)
P_c = estimate_Pc(weights, sigma)

# 画图
plt.hist(P_c, bins=50, density=True)
plt.xlabel("P(c) Values")
plt.ylabel("Density")
plt.title("Estimated P(c) Distribution")
plt.show()
