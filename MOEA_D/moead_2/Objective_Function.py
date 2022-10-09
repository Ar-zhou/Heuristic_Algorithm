import defaultdict

import numpy as np


def zdt4_f1(x_list):
    return x_list[0]


def zdt4_gx(x_list):
    sum = 1 + 10 * (10 - 1)
    for i in range(1, 10):
        sum += x_list[i] ** 2 - 10 * np.cos(4 * np.pi * x_list[i])
    return sum


def zdt4_f2(x_list):
    gx_ans = zdt4_gx(x_list)
    if x_list[0] < 0:
        print("????: x_list[0] < 0:", x_list[0])
    if gx_ans < 0:
        print("gx_ans < 0", gx_ans)
    if (x_list[0] / gx_ans) <= 0:
        print("x_list[0] / gx_ans<0：", x_list[0] / gx_ans)

    ans = 1 - np.sqrt(x_list[0] / gx_ans)
    return ans

def zdt3_f1(x):
    return x[0]


def zdt3_gx(x):
    if x[:].sum() < 0:
        print(x[1:].sum(), x[1:])
    ans = 1 + 9 / 29 * x[1:].sum()
    return ans


def zdt3_f2(x):
    g = zdt3_gx(x)
    ans = 1 - np.sqrt(x[0] / g) - (x[0] / g) * np.sin(10 * np.pi * x[0])
    return ans


class Objective_Function:
    function_dic = defaultdict(lambda: None)

    def __init__(self, f_funcs, domain):
        self.f_funcs = f_funcs
        self.domain = domain

    @staticmethod
    def get_one_function(name):
        if Objective_Function.function_dic[name] is not None:
            return Objective_Function.function_dic[name]

        if name == "zdt4":
            f_funcs = [zdt4_f1, zdt4_f2]
            domain = [[0, 1]]
            for i in range(9):
                domain.append([-5, 5])
            Objective_Function.function_dic[name] = Objective_Function(f_funcs, domain)
            return Objective_Function.function_dic[name]

        if name == "zdt3":
            f_funcs = [zdt3_f1, zdt3_f2]
            domain = [[0, 1] for i in range(30)]
            Objective_Function.function_dic[name] = Objective_Function(f_funcs, domain)
            return Objective_Function.function_dic[name]