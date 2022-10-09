import math
import pandas as pd
import numpy as np
import time
import copy
from itertools import combinations
from scipy.special import comb
import random
import matplotlib.pyplot as plt


#
# def read_excel(path):
#     para_tmp = pd.read_excel(path, index_col=[0])
#     Job_processnum = list(map(int, para_tmp.iloc[:, 0]))
#     Job_index = list(map(int, para_tmp.index))
#     Job_num = len(Job_index)
#     pt = []
#     for j in range(Job_num):
#         p = Job_processnum[j]
#         ptarr = np.zeros((p, 6))
#         ptlist = list(map(int, para_tmp.iloc[j][1:].dropna()))
#         process = 0
#         while ptlist != []:
#             num = ptlist[0]  # 2
#             for i in range(num):
#                 ptarr[process][ptlist[2 * i + 1] - 1] = ptlist[2 * i + 2]
#             del ptlist[0:num * 2 + 1]
#             process += 1
#         pt.append(ptarr)
#     return pt, Job_num, Job_processsnum, Job_index


class Item:

    def __init__(self):  # 初始化函数，定义几个参数
        self.start = []
        self.end = []
        self._on = []
        self.T = []
        self.last_ot = 0  # 最后一个工序的完成时间
        self.L = 0

    def update(self, s, e, on, t):  # 类中的函数，更新状态
        self.start.append(s)  # 记录工序的开始时间列表
        self.end.append(e)  # 记录工序的结束时间列表
        self._on.append(on)  # 记录加工工序的机器编号（无论是jobs还是machine_list
        self.T.append(t)  # 加工的时间序列
        self.last_ot = e  # 最后的时间为end
        self.L += t  # 时间累加，即机器的负载时间

    @property
    def on(self):
        return self._on


class Scheduling:
    def __init__(self, job_processnum, machine, pt):
        self.M = machine  # 机器集合，[0, 1, 2, 3, 4, 5]
        self.Job_num = len(job_processnum)
        self.Job_processnum = job_processnum  # 各个工件的工序数量列表
        self.pt = pt  # 加工时间
        self.Jobs = []
        self.Machine_list = []  # Jobs和Machine_list 需要在createJob()上方先定义
        self.createJob()
        self.createMachine()
        self.makespan = 0
        self.energycost = 0
        self.maxload = 0
        self.empty_power = [0.6, 0.6, 0.5, 0.4, 0.4, 0.6]  # 空载功率
        self.load_power = [2, 1.6, 1.8, 2.4, 2.4, 4.1]  # 负载功率

    def createJob(self):
        for i in range(self.Job_num):
            J = Item()
            self.Jobs.append(J)

    def createMachine(self):
        for i in range(len(self.M)):
            M_i = Item()
            self.Machine_list.append(M_i)

    def decode(self, Chromo_Job, Chromo_machine):  # 解码主要对Jobs和Machine_list进行更新
        jobcount = [-1 for i in range(self.Job_num)]
        begintime = 0
        endtime = 0
        loadtime = 0
        for i in range(len(Chromo_Job)):
            jobcount[Chromo_Job[i]] += 1
            process_time = \
                self.pt[Chromo_Job[i]][jobcount[Chromo_Job[i]]][Chromo_machine[i]]
            if (not self.Jobs[Chromo_Job[i]].start) and (not self.Machine_list[Chromo_machine[i]].start):
                endtime = begintime + process_time
            elif (not self.Jobs[Chromo_Job[i]].start) and self.Machine_list[Chromo_machine[i]].start:
                begintime = self.Machine_list[Chromo_machine[i]].end[-1]
                endtime = begintime + process_time
            elif self.Jobs[Chromo_Job[i]].start and (not self.Machine_list[Chromo_machine[i]].start):
                begintime = self.Jobs[Chromo_Job[i]].end[-1]
                endtime = begintime + process_time
            else:
                begintime = max(self.Jobs[Chromo_Job[i]].end[-1], self.Machine_list[Chromo_machine[i]].end[-1])
                endtime = begintime + process_time
            loadtime = process_time
            self.Jobs[Chromo_Job[i]].update(begintime, endtime, Chromo_machine[i], loadtime)
            self.Machine_list[Chromo_machine[i]].update(begintime, endtime, Chromo_machine[i], loadtime)

        # 计算makespan、负载、能耗

        self.makespan = max([self.Jobs[i].end[-1] for i in range(self.Job_num)])
        self.maxload = max([self.Machine_list[i].L for i in range(len(self.M))])
        for i in range(len(self.M)):
            if not self.Machine_list[i].end:  # 当该机器未排产时，赋值0
                eneryconsumption = 0
            else:
                eneryconsumption = self.Machine_list[i].L * self.load_power[i] + \
                               (self.Machine_list[i].end[-1] - self.Machine_list[i].start[0] - self.Machine_list[i].L) \
                               * self.empty_power[i]
            self.energycost += eneryconsumption
        return [self.makespan, self.maxload, self.energycost]

    def gant(self):
        return 0


def fintness_calculation(poplation_list):
    chromo_obj_record = np.array([])
    for chromo in poplation_list:
        schedule = Scheduling(job_processnum=Job_processnum, machine=Machine, pt=pt)
        np.append(chromo_obj_record,np.array(schedule.decode(Chromo_Job=chromo[0],
                                                 Chromo_machine=chromo[1])), 0)
    return chromo_obj_record


#
def dominate(p, q, chroms_obj_record):
    if (chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] < chroms_obj_record[q][1]
        and chroms_obj_record[p][2] < chroms_obj_record[q][2]) or (
            chroms_obj_record[p][0] <= chroms_obj_record[q][0] and chroms_obj_record[p][1] <
            chroms_obj_record[q][1]
            and chroms_obj_record[p][2] < chroms_obj_record[q][2]) or (
            chroms_obj_record[p][0] <= chroms_obj_record[q][0] and chroms_obj_record[p][1] <=
            chroms_obj_record[q][1]
            and chroms_obj_record[p][2] < chroms_obj_record[q][2]) or (
            chroms_obj_record[p][0] <= chroms_obj_record[q][0] and chroms_obj_record[p][1] <
            chroms_obj_record[q][1]
            and chroms_obj_record[p][2] <= chroms_obj_record[q][2]) or (
            chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] <=
            chroms_obj_record[q][1]
            and chroms_obj_record[p][2] < chroms_obj_record[q][2]) or (
            chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] <=
            chroms_obj_record[q][1]
            and chroms_obj_record[p][2] <= chroms_obj_record[q][2]) or (
            chroms_obj_record[p][0] < chroms_obj_record[q][0] and chroms_obj_record[p][1] <
            chroms_obj_record[q][1]
            and chroms_obj_record[p][2] <= chroms_obj_record[q][2]):
        return 1
    else:
        return 0


# 进行快速非支配排序
def fast_non_dominate_sorting(popsize, chromos_obj_record):
    s, n = {}, {}
    front, rank = [], []
    front[0] = []
    iter_range = popsize  # 如果不是亲子代混合，就是100+x个需要比较
    for p in range(iter_range):
        s[p] = []
        n[p] = 0
        for q in range(iter_range):
            if dominate(p, q, chromos_obj_record):  # if p dominates q for three objectives
                if q not in s[p]:
                    s[p].append(q)  # s[p] is the set of solutions dominated by p
            elif dominate(q, p, chromos_obj_record):  # if q dominates p for three objectives
                n[p] = n[p] + 1  # n[p] is the set of solutions dominating p, 3 obj
        if n[p] == 0:
            rank[p] = 0  # p belongs to front 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []  # Used to store the members of the next front
        for p in front[i]:
            for q in s[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front[i] = Q

    del front[len(front) - 1]
    return front, rank


# TODO
def crowding_distance_calculation(front, chroms_obj_record):
    distance = {m: 0 for m in front}  # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for o in range(3):  # dual objective, we have three obj, so this should be adapted
        obj = {m: chroms_obj_record[m][o] for m in front}
        sorted_keys = sorted(obj, key=obj.get)
        distance[sorted_keys[0]] = distance[sorted_keys[len(front) - 1]] = 999999999999
        for i in range(1, len(front) - 1):
            if len(set(obj.values())) == 1:
                distance[sorted_keys[i]] = distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]] = distance[sorted_keys[i]] + (
                        obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) / (
                                                   obj[sorted_keys[len(front) - 1]] - obj[sorted_keys[0]])
    # The overall crowding distance is the sum of distance corresponding to each objective
    return distance


# 进行


def crossover(population_list, crossover_rate):
    pa_list_cpoy = copy.deepcopy(population_list)
    pa_list = []
    of_list = []
    pa_index = [i for i in range(len(population_list))]
    random.shuffle(pa_index)  # 将父代顺序打乱
    for i in range(len(population_list)):
        pa_list.append(pa_list_cpoy[pa_index[i]])  # 打乱后的父代
    for i in range(int(len(pa_list) / 2)):
        parent1 = pa_list[2 * i][:]
        parent2 = pa_list[2 * i + 1][:]
        # # 对工序编码进行POX交叉
        if np.random.random() <= crossover_rate:
            s = random.randint(1, 8)  # 1~8之间取一数s, 0~s的索引代表的工序不做改变，s~9进行顺序交换，同时交换其机器编码
            judge_list = []
            for j in range(s, max(parent1[0])+1):
                judge_list.append(j)
            p2 = 0
            for p1 in range(len(parent1[0])):
                if parent1[0][p1] not in judge_list:
                    continue
                while parent2[0][p2] not in judge_list:
                    p2 += 1
                if parent2[0][p2] in judge_list:
                    parent1[0][p1], parent2[0][p2] = parent2[0][p2], parent1[0][p1]
                    parent1[1][p1], parent2[1][p2] = parent2[1][p2], parent1[1][p1]
                    p2 += 1
        # 对机器编码进行均匀交叉
        indexp2 = sorted(range(len(parent2[0])), key=lambda k: parent2[0][k])
        indexp1 = sorted(range(len(parent1[0])), key=lambda k: parent1[0][k])
        for j in range(len(parent1[0])):
            if np.random.random() <= crossover_rate:
                parent1[1][indexp1[j]], parent2[1][indexp2[j]] = parent2[1][indexp2[j]], parent1[1][indexp1[j]]

        of_list.extend((parent1, parent2))
    return of_list  # 时间复杂度O（n**2)


def mutation(offspring_list, mutation_rate, pt_index):
    pa_list = copy.deepcopy(offspring_list)
    for i in range(len(pa_list)):
        if np.random.random() <= mutation_rate:
            m = random.choice(list(range(len(pa_list[0][0]))))  # 处于m位置的机器编码变异
            p_index = -1  # 工序索引
            job = pa_list[i][0][m]
            for j in range(m+1):
                if pa_list[i][0][j] == job:
                    p_index += 1
            if len(pt_index[job][p_index][:]) > 1:
                choice_ma = random.choice(pt_index[job][p_index][:])  # 随机选择工序中可加工的机器
                while pa_list[i][1][m] == choice_ma:  # 判断是否重复，重复就重新挑选，直至
                    choice_ma = random.choice(pt_index[job][p_index][:])
                pa_list[i][1][m] = choice_ma
                break
    return pa_list


# TODO
def elite_selection(combined_list, combined_front, combined_obj_record, popsize):
    # N = 0  # elist preservation
    # new_pop = []  # new——pop记录原种群里的index
    # total_size = len(combined_list)
    # while N < total_size:
    #     for i in range(len(combined_front)):
    #         N = N + len(combined_front[i])
    #         if N > total_size:
    #             distance = crowding_distance_calculation(combined_front[i], combined_obj_record)
    #             sorted_cdf = sorted(distance, key=distance.get)
    #             sorted_cdf.reverse()
    #             for j in sorted_cdf:
    #                 if len(new_pop) == total_size:
    #                     break
    #                 new_pop.append(j)
    #             break
    #
    #         else:
    #             new_pop.extend(combined_front[i])
    # pop_list = []
    # for n in new_pop:
    #     while len(pop_list) < popsize:
    #         pop_list.append(combined_list[n])
    #         continue
    #     break
    # return pop_list, new_pop
    pop_list = []
    new_pop = []
    for i in range(len(combined_front)):
        for j in combined_front[i]:
            if len(pop_list) >= popsize:
                break
            pop_list.append(combined_list[j])
            new_pop.append(j)
    return pop_list, new_pop


# 创建参考点集,返回点集以及参考点个数N，M为目标函数个数
def uniformpoint(popsize, M):
    H1 = 1
    while (comb(H1 + M - 1, M - 1) <= popsize):
        H1 = H1 + 1
    H1 = H1 - 1
    W = np.array(list(combinations(range(H1 + M - 1), M - 1))) - np.tile(np.array(list(range(M - 1))),
                                                                         (int(comb(H1 + M - 1, M - 1)), 1))
    W = (np.hstack((W, H1 + np.zeros((W.shape[0], 1)))) - np.hstack((np.zeros((W.shape[0], 1)), W))) / H1
    if H1 < M:
        H2 = 0
        while comb(H1 + M - 1, M - 1) + comb(H2 + M - 1, M - 1) <= popsize:
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


def pdist(x, y):
    x0 = x.shape[0]
    y0 = y.shape[0]
    xmy = np.dot(x, y.T)  # x乘以y
    xm = np.array(np.sqrt(np.sum(x ** 2, 1))).reshape(x0, 1)
    ym = np.array(np.sqrt(np.sum(y ** 2, 1))).reshape(1, y0)
    xmmym = np.dot(xm, ym)
    cos = xmy / xmmym
    return cos


def last_selection(chromos_obj_record, K, Zmin, Z, last, front):
    popfun = np.array([chromos_obj_record[i] for i in range(sum([len(front[j]) for j in range(last+1)]))])
    N, M = popfun.shape[0], popfun.shape[1]
    popfun1 = popfun[:(popsize - K), :]
    popfun2 = popfun[(popsize - K):, :]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]


    extreme = np\
        .zeros(M)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun / (np.tile(w[i,:], (N, 1))), 1))  # 1代表取每一行最大值

    extreme = extreme.astype(int)
    temp = np.linalg.pinv(np.mat(popfun[extreme, :]))  # 求逆矩阵
    hyprtplane = np.array(np.dot(temp, np.ones((M, 1))))  # np.dot代表矩阵乘积
    a = 1 / hyprtplane
    if np.sum(a == math.nan) != 0:
        a = np.max(popfun, 0)  # 按照列最大值取,得到一个一维列表
    np.array(a).reshape(M, 1)  # 将一维列表reshape成二维数组
    a = a.T
    popfun = popfun / (np.tile(a, (N, 1)))

    # 联系每一个解和对应向量
    # 计算每一个解最近的参考线的距离
    cos = pdist(popfun, Z)
    distance = np.tile(np.array(np.sqrt(np.sum(popfun ** 2, 1))).reshape(N, 1), (1, NZ)) * np.sqrt(1 - cos ** 2)
    # 联系每一个解和对应的向量
    d = np.min(distance.T, 0)
    pi = np.argmin(distance.T, 0)

    # 计算z关联的个数
    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)

    # 选出剩余的K个
    choose = np.zeros(N2)
    choose = choose.astype(bool)
    zchoose = np.ones(NZ)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < K:
        # 选择最不拥挤的参考点
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
        #        I = np.ravel(np.array(np.where(choose == False)))
        #        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
        I = np.ravel(np.array(np.where(pi[N1:] == j)))
        I = I[choose[I] == False]
        if I.shape[0] != 0:
            if rho[j] == 0:
                s = np.argmin(d[N1 + I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            rho[j] = rho[j] + 1
        else:
            zchoose[j] = False
    return choose


# 选择操作
def envselect(chromos_obj_record, Zmin, popsize, M, Z, N, Combined_front):
    front, rank = fast_non_dominate_sorting(popsize, chromos_obj_record)
    count = 0
    last = -1  # 计算最后需要进行计算的front层
    for i in range(len(front)):
        count += len(front[i])
        last += 1
        if count >= popsize:
            break
    last_front = np.array(front[last])
    K = N - sum([len(front[i]) for i in range(last)])  # 代表最后一层还需要选择的个体数
    choose = last_selection(chromos_obj_record, K, Zmin, Z, last, Combined_front)  # 返回选择好的个体索引

    return 0


def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# 种群自适应归一化
def normalization(Z, chromos_obj_record):



def gerenate_chromo(Job_processnum, pi, pt, pt_index):
    Chromo_job = []
    m = 0
    for i in Job_processnum:
        c1 = [m for j in range(i)]
        Chromo_job.extend(c1)
        m += 1
    np.random.shuffle(Chromo_job)
    judge_index = [np.random.random() for i in range(len(Chromo_job))]  # 55个随机数（0,1）
    Chromo_machine = [0 for i in range(len(Chromo_job))]  # 55，用于记录Chromo对应的加工机器的编号
    index_1 = [-1 for i in range(len(Job_processnum))]  # [-1, -1, -1,......]
    for i in range(len(judge_index)):
        index_1[Chromo_job[i]] += 1
        if judge_index[i] <= pi:
            Chromo_machine[i] = index_of(min(pt[Chromo_job[i]][index_1[Chromo_job[i]]][:]),
                                         pt[Chromo_job[i]][index_1[Chromo_job[i]]][:])
        else:
            Chromo_machine[i] = random.choice(pt_index[Chromo_job[i]][index_1[Chromo_job[i]]][:])
    return [Chromo_job, Chromo_machine]


'''--------------------------------main---------------------------------------'''
if __name__ == "__main__":
    # 读取数据
    path = r'D:\cangmu\Documents\CODE\learn_python\NSGA-II\NSGA-II_PFSP' + '\\FJSP-data\\' + 'MK01' + '.xlsx'
    para_tmp = pd.read_excel(path, index_col=[0])
    Job_processnum = list(map(int, para_tmp.iloc[:, 0]))
    Job_index = list(map(int, para_tmp.index))
    Job_num = len(Job_index)
    Machine = list(range(6))
    popsize = 100  # 设置种群数量
    crossover_rate = 0.8  # 设置交叉概率
    mutation_rate = 0.1  # 设置变异概率
    num_generation = 200  # 设置迭代次数
    M = 3  # 目标函数个数

    pt = []
    for j in range(Job_num):
        p = Job_processnum[j]
        ptarr = [[math.inf for i in range(len(Machine))] for i in range(p)]
        ptlist = list(map(int, para_tmp.iloc[j][1:].dropna()))
        process = 0
        while ptlist != []:
            num = ptlist[0]  # 2
            for i in range(num):
                ptarr[process][ptlist[2 * i + 1] - 1] = ptlist[2 * i + 2]
            del ptlist[0:num * 2 + 1]
            process += 1
        pt.append(ptarr)

    pt_index = [[[] for j in range(Job_processnum[i])] for i in range(Job_num)]
    pt_worktime = [[[] for j in range(Job_processnum[i])] for i in range(Job_num)]
    for i in range(Job_num):
        for j in range(Job_processnum[i]):
            for k in range(len(Machine)):
                if pt[i][j][k] != math.inf:
                    pt_index[i][j].append(k)
                    pt_worktime[i][j].append(pt[i][j][k])

    # 开始
    Population_list = []
    Chromos_obj_record = []
    # 生成初始种群
    for i in range(popsize):
        Population_list.append(gerenate_chromo(Job_processnum, 0.5, pt, pt_index))
    # # 对初始种群进行解码，生成适应度值
    # Chromos_obj_record = np.array(fintness_calculation(Population_list))
    # # 求出初始种群的理想点,理想点即每个目标函数的最小值
    # Zmin = Chromos_obj_record.min(0)
    Zmin = np.array([])  # 理想点
    Z, N = uniformpoint(popsize, M)  # Z是参考点
    gen = 1
    total_time_start = time.time()
    while gen < num_generation:
        best_obj = []
        time_start = time.time()
        Parent_list = copy.deepcopy(Population_list)
        # 交叉
        Offspring_list = crossover(Parent_list, crossover_rate)
        # 变异
        Offspring_list = mutation(Offspring_list, mutation_rate, pt_index)

        # 父子代合并
        Combined_list = copy.deepcopy(Parent_list) + copy.deepcopy(Offspring_list)
        # 计算合并后的适应度函数
        Combined_obj_record = fintness_calculation(Combined_list)
        # 更新理想点
        Zmin = np.array(Combined_list).min(0)
        # 进行非支配排序
        Combined_front, Combined_rank = fast_non_dominate_sorting(len(Combined_list), Combined_obj_record)

        # 进行选择操作,MZN分别是目标函数个数，参考点，参考点个数
        Population_list = envselect(Chromos_obj_record, Zmin, popsize, M, Z, N, Combined_front)

        # Population_list, new_pop = elite_selection(Combined_list, Combined_front, Combined_obj_record, popsize)
        time_end = time.time()
        time_cost = time_end - time_start
        print("第 %d 次循环用时%ds" % (gen, time_cost))

        new_pop_obj = [Combined_obj_record[k] for k in new_pop]
        print("群体的目标函数为", new_pop_obj)
        gen += 1
    total_time_cost = time.time()
    print("总用时为：%ds" % (total_time_cost-total_time_start))

