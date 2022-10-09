import random

import numpy as np

import Moead_Util
import Objective_Function
import Person

import matplotlib.pyplot as plt


def draw(x, y):
    plt.scatter(x, y, s=10, c="grey")  # s 点的大小  c 点的颜色 alpha 透明度
    plt.show()


iterations = 1000  # 迭代次数
N = 400
m = 2
T = 5
o_func = Objective_Function.get_one_function("zdt4")
pm = 0.7

moead = Moead_Util(N, m, T, o_func, pm)

moead.init_mean_vector()
moead.init_Euler_distance()
pop = moead.init_population()
moead.init_Z()

for i in range(iterations):
    print(i, len(moead.EP))
    for person in pop:
        p1, p2 = person.choice_two_person()
        d1, d2 = Person.cross_get_two_new_dna(p1, p2)

        if np.random.rand() < pm:
            p1.mutation_dna(d1)
        if np.random.rand() < pm:
            p1.mutation_dna(d2)

        moead.update_Z(d1)
        moead.update_Z(d2)
        t1, t2 = person.choice_two_person()
        if t1.compare(d1) < 0:
            t1.accept_new_dna(d1)
            moead.update_ep(d1)
        if t2.compare(d1) < 0:
            t2.accept_new_dna(d2)
            moead.update_ep(d1)

# 输出结果画图
EP_fx = np.array(moead.EP_fx)

x = EP_fx[:, 0]
y = EP_fx[:, 1]
draw(x, y)