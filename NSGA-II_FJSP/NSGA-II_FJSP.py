import math
import pandas as pd
import numpy as np
import time
import copy
import itertools
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
        self.p = []  # 记录机器加工的工序列表，'str'-O_ij


    def update(self, s, e, on, t, p):  # 类中的函数，更新状态
        self.start.append(s)  # 记录工序的开始时间列表
        self.end.append(e)  # 记录工序的结束时间列表
        self._on.append(on)  # 对于jobs更新机器列表，对于machines，更新加工工件编号
        self.p.append(p)  # 对于jobs更新机器列表，对于machines，更新加工工件和工序编号
        self.T.append(t)  # 加工的时间序列
        self.last_ot = e  # 最后的时间为end
        self.L += t  # 时间累加，即机器的负载时间

    def get_on(self):
        return self._on

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
        for i in range(len(Chromo_Job)):
            j = Chromo_Job[i]  # 工件编号
            jobcount[j] += 1
            process_time = \
                self.pt[j][jobcount[Chromo_Job[i]]][Chromo_machine[i]]
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
            self.Jobs[Chromo_Job[i]].update(begintime, endtime, Chromo_machine[i], loadtime, r'M_' + str(Chromo_machine[i]))
            self.Machine_list[Chromo_machine[i]].update(begintime, endtime, j, loadtime, r'O_' + str(j+1) + str(jobcount[j]+1))

        # 计算makespan、负载、能耗

        self.makespan = max([self.Jobs[i].end[-1] for i in range(self.Job_num)])
        self.maxload = max([self.Machine_list[i].L for i in range(len(self.M))])
        for i in range(len(self.M)):
            if not self.Machine_list[i].end:  # 当该机器未排产时，赋值0
                eneryconsumption = 0
            else:
                eneryconsumption = self.Machine_list[i].L * self.load_power[i] + \
                                   (self.Machine_list[i].end[-1] - self.Machine_list[i].start[0] - self.Machine_list[
                                       i].L) \
                                   * self.empty_power[i]
            self.energycost += eneryconsumption
        return [self.makespan, self.maxload, self.energycost]

    def gant(self):
        fig = plt.figure()
        colors = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle',
                  'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod',
                  'mediumslateblue', 'navajowhite', 'navy', 'sandybrown']
        M_num = 0
        for i in range(len(self.M)):
            for j in range(len(self.Machine_list[i]._on)):
                start_time = self.Machine_list[i].start[j]
                end_time = self.Machine_list[i].end[j]
                job = self.Machine_list[i]._on[j]  # 机器加工的工件编号
                p = self.Machine_list[i].p[j]  # 字符串，机器加工的O_jp
                plt.barh(M_num, width=end_time - start_time, height=0.8, left=start_time, color=colors[job],
                         edgecolor='black')
                plt.text(x=start_time + ((end_time - start_time) / 2 - 0.25), y=M_num - 0.2, s=p, size=5,
                         fontproperties='Time New Roman')
            M_num += 1
        # plt.yticks(np.arange(M_num + 1), np.arange(1, M_num + 1), size=20, fontproperties='Times New Roman')

        plt.ylabel("机器", size=20, fontproperties='SimSun')
        plt.xlabel("时间", size=20, fontproperties='SimSun')
        plt.tick_params(labelsize=20)
        plt.tick_params(direction='in')
        plt.show()


def fintness_calculation(poplation_list):
    chromo_obj_record = []
    for chromo in poplation_list:
        schedule = Scheduling(job_processnum=Job_processnum, machine=Machine, pt=pt)
        chromo_obj_record.append(schedule.decode(Chromo_Job=chromo[0],
                                                 Chromo_machine=chromo[1]))
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
    front, rank = [[]], [-1 for i in range(popsize)]
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
        front.append(Q)

    del front[len(front) - 1]
    return front, rank


def crowding_distance_calculation(front, chroms_obj_record):
    if np.array(front).ndim != 1:
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
    elif np.array(front).ndim == 1:
        distance = np.zeros(len(front))
        objs = np.array([chroms_obj_record[front[i]] for i in range(len(front))])
        for o in range(3):  # dual objective, we have three obj, so this should be adapted
            obj = {m: objs[m][o] for m in range(len(front))}
            sorted_keys = sorted(obj, key=obj.get)
            distance[sorted_keys[0]] = distance[sorted_keys[-1]] = 999999999999
            if len(distance) == 1:
                break
            else:
                for i in range(1, len(front) - 1):
                    distance[sorted_keys[i]] = distance[sorted_keys[i]] + (
                            obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) / (
                                                       obj[sorted_keys[len(front) - 1]] - obj[sorted_keys[0]])
        return objs[np.argmin(distance)], front[np.argmin(distance)]


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
            for j in range(s, max(parent1[0]) + 1):
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
            for j in range(m + 1):
                if pa_list[i][0][j] == job:
                    p_index += 1
            if len(pt_index[job][p_index][:]) > 1:
                choice_ma = random.choice(pt_index[job][p_index][:])  # 随机选择工序中可加工的机器
                while pa_list[i][1][m] == choice_ma:  # 判断是否重复，重复就重新挑选，直至
                    choice_ma = random.choice(pt_index[job][p_index][:])
                pa_list[i][1][m] = choice_ma
                break
    return pa_list


# FIXME
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


def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


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
    path = r'D:\cangmu\Documents\CODE\learn_python\Algorithm\heuristic\NSGA-II\NSGA-II_FJSP' + '\\FJSP-data\\' + 'MK01' + '.xlsx'
    para_tmp = pd.read_excel(path, index_col=[0])
    Job_processnum = list(map(int, para_tmp.iloc[:, 0]))
    Job_index = list(map(int, para_tmp.index))
    Job_num = len(Job_index)
    Machine = list(range(6))
    popsize = 500  # 设置种群数量
    crossover_rate = 0.8  # 设置交叉概率
    mutation_rate = 0.1  # 设置变异概率
    num_generation = 200  # 设置迭代次数

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
    # 产生子种群
    Population_list = []
    Chromos_obj_record = []
    for i in range(popsize):
        Population_list.append(gerenate_chromo(Job_processnum, 0.5, pt, pt_index))
    # 解码计算每个个体的适应度，添加至列表
    gen = 0
    total_time_start = time.time()
    best_record = []
    while gen < num_generation:
        time_start = time.time()
        Parent_list = copy.deepcopy(Population_list)
        Offspring_list = crossover(Parent_list, crossover_rate)
        Offspring_list = mutation(Offspring_list, mutation_rate, pt_index)

        # print(Chromos_obj_record)
        Combined_list = copy.deepcopy(Parent_list) + copy.deepcopy(Offspring_list)
        Combined_obj_record = fintness_calculation(Combined_list)
        Combined_front, Combined_rank = fast_non_dominate_sorting(len(Combined_list), Combined_obj_record)
        # 进行选择操作
        Population_list, new_pop = elite_selection(Combined_list, Combined_front, Combined_obj_record, popsize)
        time_end = time.time()
        time_cost = time_end - time_start
        print("第 %d 次循环用时%ds" % (gen, time_cost))
        new_pop_obj = [Combined_obj_record[k] for k in new_pop]
        print("群体的目标函数为", new_pop_obj)
        gen += 1
        # 对子代的front[0]进行拥挤度运算，算出拥挤度最小的个体作为本次迭代的最优解
        best, best_front = crowding_distance_calculation(Combined_front[0], Combined_obj_record)  # 得出最佳个体的目标函数和其索引
        best_record.append(best)
        if gen >= num_generation:  # 如果最后一次迭代，将最优个体甘特图画出来
            schedule = Scheduling(Job_processnum, Machine, pt)
            schedule.decode(Combined_list[best_front][0], Combined_list[best_front][1])
            schedule.gant()

    total_time_cost = time.time()
    print("总用时为：%ds" % (total_time_cost - total_time_start))
    # 画图,输出三个目标函数迭代进化曲线，最后一次迭代画出最佳个体的甘特图
    # fig = plt.figure()
    ax = plt.axes()
    x = np.array([i for i in range(num_generation)])
    y_makespan = np.array([best_record[i][0] for i in range(num_generation)])
    y_maxload = np.array([best_record[i][1] for i in range(num_generation)])
    y_enerycost = np.array([best_record[i][2] for i in range(num_generation)])
    ax.plot(x, y_makespan, '-r', label='makespan')
    ax.plot(x, y_maxload, '-g', label='maxload')
    ax.plot(x, y_enerycost, '-b', label='enerycost')
    plt.legend(loc='right')
    plt.xlabel("number of generation")
    plt.show()
