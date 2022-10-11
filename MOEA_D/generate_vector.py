"""
生成C(m-1, H+m-1)个均匀分布的权重向量
参考博主地址：https://www.cnblogs.com/Twobox/p/16408751.html
"""
import numpy as np


def get_mean_vectors(sum, m):
    # 取m个数，数的和为N
    if m == 1:
        return [[sum]]
    vectors = []
    for i in range(1, sum - (m - 1) + 1):
        right_vec = get_mean_vectors(sum - i, m - 1)
        a = [i]
        for item in right_vec:
            vectors.append(a + item)
    return vectors


N = 30
# 每个维度上的间隔数。加上边缘点就是N+1个点
m = 4  # 目标维度

vectors = get_mean_vectors(N + m, m)
vectors = (np.array(vectors) - 1) / N
print(len(vectors))  # len = C m-1, N+m-1
print(vectors)