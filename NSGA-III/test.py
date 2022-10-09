from scipy.special import comb
from itertools import combinations
import numpy as np
import copy
import math


def uniformpoint(N, M):
    H1 = 1
    while (comb(H1 + M - 1, M - 1) <= N):
        H1 = H1 + 1
    H1 = H1 - 1
    W = np.array(list(combinations(range(H1 + M - 1), M - 1))) - np.tile(np.array(list(range(M - 1))),
                                                                         (int(comb(H1 + M - 1, M - 1)), 1))
    W = (np.hstack((W, H1 + np.zeros((W.shape[0], 1)))) - np.hstack((np.zeros((W.shape[0], 1)), W))) / H1
    if H1 < M:
        H2 = 0
        while (comb(H1 + M - 1, M - 1) + comb(H2 + M - 1, M - 1) <= N):
            H2 = H2 + 1
        H2 = H2 - 1
        if H2 > 0:
            W2 = np.array(list(combinations(range(H2 + M - 1), M - 1))) - np.tile(np.array(list(range(M - 1))),
                                                                                  (int(comb(H2 + M - 1, M - 1)), 1))
            W2 = (np.hstack((W2, H2 + np.zeros((W2.shape[0], 1)))) - np.hstack((np.zeros((W2.shape[0], 1)), W2))) / H2
            W2 = W2 / 2 + 1 / (2 * M)
            W = np.vstack((W, W2))  # 按列合并
    W[W < 1e-6] = 1e-6
    N = W.shape[0]
    return W, N

a, b = uniformpoint(5, 5)