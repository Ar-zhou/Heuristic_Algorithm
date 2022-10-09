import numpy as np

import Person


def distribution_number(sum, m):
    # 取m个数，数的和为N
    if m == 1:
        return [[sum]]
    vectors = []
    for i in range(1, sum - (m - 1) + 1):
        right_vec = distribution_number(sum - i, m - 1)
        a = [i]
        for item in right_vec:
            vectors.append(a + item)
    return vectors


class Moead_Util:
    def __init__(self, N, m, T, o_func, pm):
        self.N = N
        self.m = m
        self.T = T  # 邻居大小限制
        self.o_func = o_func
        self.pm = pm  # 变异概率

        self.Z = np.zeros(shape=m)

        self.EP = []  # 前沿
        self.EP_fx = []  # ep对应的目标值
        self.weight_vectors = None  # 均匀权重向量
        self.Euler_distance = None  # 欧拉距离矩阵
        self.pip_size = -1

        self.pop = None
        # self.pop_dna = None

    def init_mean_vector(self):
        vectors = distribution_number(self.N + self.m, self.m)
        vectors = (np.array(vectors) - 1) / self.N
        self.weight_vectors = vectors
        self.pip_size = len(vectors)
        return vectors

    def init_Euler_distance(self):
        vectors = self.weight_vectors
        v_len = len(vectors)

        Euler_distance = np.zeros((v_len, v_len))
        for i in range(v_len):
            for j in range(v_len):
                distance = ((vectors[i] - vectors[j]) ** 2).sum()
                Euler_distance[i][j] = distance

        self.Euler_distance = Euler_distance
        return Euler_distance

    def init_population(self):
        pop_size = self.pip_size
        dna_len = len(self.o_func.domain)
        pop = []
        pop_dna = np.random.random(size=(pop_size, dna_len))
        # 初始个体的 dna
        for i in range(pop_size):
            pop.append(Person(pop_dna[i]))

        # 初始个体的 weight_vector, neighbor, o_func
        for i in range(pop_size):
            # weight_vector, neighbor, o_func
            person = pop[i]
            distance = self.Euler_distance[i]
            sort_arg = np.argsort(distance)
            weight_vector = self.weight_vectors[i]
            # neighbor = pop[sort_arg][:self.T]
            neighbor = []
            for i in range(self.T):
                neighbor.append(pop[sort_arg[i]])

            o_func = self.o_func
            person.set_info(weight_vector, neighbor, o_func)
        self.pop = pop
        # self.pop_dna = pop_dna

        return pop

    def init_Z(self):
        Z = np.full(shape=self.m, fill_value=float("inf"))
        for person in self.pop:
            for i in range(len(self.o_func.f_funcs)):
                f = self.o_func.f_funcs[i]
                # f_x_i：某个体，在第i目标上的值
                f_x_i = f(person.dna)
                if f_x_i < Z[i]:
                    Z[i] = f_x_i

        self.Z = Z

    def get_fx(self, dna):
        fx = []
        for f in self.o_func.f_funcs:
            fx.append(f(dna))
        return fx

    def update_ep(self, new_dna):
        # 将新解与EP每一个进行比较，删除被新解支配的
        # 如果新解没有被旧解支配，则保留
        new_dna_fx = self.get_fx(new_dna)
        accept_new = True  # 是否将新解加入EP
        # print(f"准备开始循环: EP长度{len(self.EP)}")
        for i in range(len(self.EP) - 1, -1, -1):  # 从后往前遍历
            old_ep_item = self.EP[i]
            old_fx = self.EP_fx[i]
            # old_fx = self.get_fx(old_ep_item)
            a_b = True  # 老支配行
            b_a = True  # 新支配老
            for j in range(len(self.o_func.f_funcs)):
                if old_fx[j] < new_dna_fx[j]:
                    b_a = False
                if old_fx[j] > new_dna_fx[j]:
                    a_b = False
            # T T : fx相等      直接不改变EP
            # T F ：老支配新     留老，一定不要新，结束循环.
            # F T ：新支配老     留新，一定不要这个老，继续循环
            # F F : 非支配关系   不操作，循环下一个
            # TF为什么结束循环，FT为什么继续循环，你们可以琢磨下
            if a_b:
                accept_new = False
                break
            if not a_b and b_a:
                if len(self.EP) <= i:
                    print(len(self.EP), i)
                del self.EP[i]
                del self.EP_fx[i]
                continue

        if accept_new:
            self.EP.append(new_dna)
            self.EP_fx.append(new_dna_fx)
        return self.EP, self.EP_fx

    def update_Z(self, new_dna):
        new_dna_fx = self.get_fx(new_dna)
        Z = self.Z
        for i in range(len(self.o_func.f_funcs)):
            if new_dna_fx[i] < Z[i]:
                Z[i] = new_dna_fx[i]
        return Z