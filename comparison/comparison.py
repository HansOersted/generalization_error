import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def binomial_tail_bound_max(m, k, delta_pc):
    """
    计算 Binomial Tail Inversion Bound，寻找最大概率 p，使得 Bin(m, k, p) >= delta_pc。
    """
    p_values = np.linspace(1, 0, 1000)  # 逆序搜索
    for p in p_values:
        if binom.cdf(k, m, p) >= delta_pc:  # CDF 计算 Pr(sum Z_i <= k)
            return p
    return 0.0  # 默认返回 0.0

# 设置参数
m = 100  # 训练样本数量
delta = 0.05  # 置信水平
c_S_hat = np.arange(0, m + 1)  # 经验误差 k 的取值范围 (整数)
k_values = c_S_hat  # 直接使用经验误差值 k

# 假设先验 P(c) 为均匀分布，即所有分类器的 P(c) = 1/N
N = len(c_S_hat)  # 分类器的总数
P_c = np.ones(N) / N  # 先验概率 P(c)

# 计算 Binomial Tail Inversion Bound（严格的二项分布界限）
binomial_bound = np.array([binomial_tail_bound_max(m, k, delta * P_c[i]) for i, k in enumerate(k_values)])

# 计算 Chernoff Bound 近似（Relaxation）
chernoff_bound = c_S_hat / m + np.sqrt((np.log(1/P_c) + np.log(1/delta)) / (2*m))

# 绘制误差界限
plt.figure(figsize=(8, 6))
plt.plot(c_S_hat / m, binomial_bound, label="Binomial Tail Inversion Bound (No Relaxation)", linestyle='-', color='blue')
plt.plot(c_S_hat / m, chernoff_bound, label="Chernoff Bound (Relaxation)", linestyle='--', color='red')

# 标注
plt.xlabel("Empirical Error Rate (c_S_hat / m)")
plt.ylabel("Error Bound")
plt.title("Comparison of Relaxation Before and After")
plt.legend()
plt.grid(True)
plt.show()
