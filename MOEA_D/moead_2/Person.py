import numpy as np


class Person:
    def __init__(self, dna):
        self.dna = dna
        self.weight_vector = None
        self.neighbor = None
        self.o_func = None  # 目标函数

        self.dns_len = len(dna)

    def set_info(self, weight_vector, neighbor, o_func):
        self.weight_vector = weight_vector
        self.neighbor = neighbor
        self.o_func = o_func# 目标函数

    def mutation_dna(self, one_dna):
        i = np.random.randint(0, self.dns_len)
        low = self.o_func.domain[i][0]
        high = self.o_func.domain[i][1]
        new_v = np.random.rand() * (high - low) + low
        one_dna[i] = new_v
        return one_dna

    def mutation(self):
        i = np.random.randint(0, self.dns_len)
        low = self.o_func.domain[i][0]
        high = self.o_func.domain[i][1]
        new_v = np.random.rand() * (high - low) + low
        self.dna[i] = new_v

    @staticmethod
    def cross_get_two_new_dna(p1, p2):
        # 单点交叉
        cut_i = np.random.randint(1, p1.dns_len - 1)
        dna1 = p1.dna.copy()
        dna2 = p2.dna.copy()
        temp = dna1[cut_i:].copy()
        dna1[cut_i:] = dna2[cut_i:]
        dna2[cut_i:] = temp
        return dna1, dna2

    def compare(self, son_dna):
        F = self.o_func.f_funcs
        f_x_son_dna = []
        f_x_self = []
        for f in F:
            f_x_son_dna.append(f(son_dna))
            f_x_self.append(f(self.dna))
        fit_son_dna = np.array(f_x_son_dna) * self.weight_vector
        fit_self = np.array(f_x_self) * self.weight_vector
        return fit_son_dna.sum() - fit_self.sum()

    def accept_new_dna(self, new_dna):
        self.dna = new_dna

    def choice_two_person(self):
        neighbor = self.neighbor
        neighbor_len = len(neighbor)
        idx = np.random.randint(0, neighbor_len, size=2)
        p1 = self.neighbor[idx[0]]
        p2 = self.neighbor[idx[1]]
        return p1, p2