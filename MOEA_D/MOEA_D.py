# * Zhou Hong
# * hongzhou_ie@163.com
# * Nanjing University of Aeronautics and Astronautics

# from gurobipy import *
import random
import copy
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


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


def fitness_calculation(person):
    schedule = Scheduling(job_processnum=Job_processnum, machine=Machine, pt=pt)
    return schedule.decode(Chromo_Job=person[0], Chromo_machine=person[1])


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


def crossover(parent_k, parent_l):
    parent1 = parent_k
    parent2 = parent_l
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
    return parent1, parent2


def mutation(p1, p2):
    list_ = [p1, p2]
    for i in range(len(list_)):
        if np.random.random() <= mutation_rate:
            m = random.choice(list(range(len(p1[0]))))  # 处于m位置的机器编码变异
            p_index = -1  # 工序索引
            job = list_[i][0][m]
            for j in range(m + 1):
                if list_[i][0][j] == job:
                    p_index += 1
            if len(pt_index[job][p_index][:]) > 1:
                choice_ma = random.choice(pt_index[job][p_index][:])  # 随机选择工序中可加工的机器
                while list_[i][1][m] == choice_ma:  # 判断是否重复，重复就重新挑选，直至
                    choice_ma = random.choice(pt_index[job][p_index][:])
                list_[i][1][m] = choice_ma
                break
    return random.choice(list_)


def cross_and_mutation(parent_k, parent_l):
    p1, p2 = crossover(parent_k, parent_l)
    off = mutation(p1, p2)
    return off


def crowding_distance_calculation(ep):
    distance = np.zeros(len(ep))
    objs = np.array([ep[i] for i in range(len(ep))])
    for o in range(3):  # dual objective, we have three obj, so this should be adapted
        obj = {m: objs[m][o] for m in range(len(ep))}
        sorted_keys = sorted(obj, key=obj.get)
        distance[sorted_keys[0]] = distance[sorted_keys[-1]] = 999999999999
        if len(distance) == 1:
            break
        else:
            for i in range(1, len(ep) - 1):
                distance[sorted_keys[i]] = distance[sorted_keys[i]] + (
                        obj[sorted_keys[i + 1]] - obj[sorted_keys[i - 1]]) / (
                                                   obj[sorted_keys[len(ep) - 1]] - obj[sorted_keys[0]])
    return objs[np.argmin(distance)]


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


def generate_B_i(t, index, e_distance):
    B_sort = np.argsort(e_distance[index])
    B_i = []
    for i in range(t):
        B_i.append(B_sort[i])
    return B_i


"===========================================main code========================================"
if __name__ == "__main__":
    # 读取数据
    path = r'D:\cangmu\Documents\CODE\learn_python\Algorithm\heuristic\NSGA-II\NSGA-II_FJSP' + '\\FJSP-data\\' + 'MK01' + '.xlsx'
    para_tmp = pd.read_excel(path, index_col=[0])
    Job_processnum = list(map(int, para_tmp.iloc[:, 0]))
    Job_index = list(map(int, para_tmp.index))
    Job_num = len(Job_index)
    Machine = list(range(6))
    crossover_rate = 0.8  # 设置交叉概率
    mutation_rate = 0.1  # 设置变异概率

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
    # 生成权重向量，一共生成496个权重向量
    N = 30
    # 每个维度上的间隔数。加上边缘点就是N+1个点
    m = 3  # 目标维度
    T = 4  # 领域中邻居的数量
    vectors = get_mean_vectors(N + m, m)
    vectors = (np.array(vectors) - 1) / N
    Eular_distance = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            Eular_distance[i, j] = ((vectors[i] - vectors[j]) ** 2).sum()

    # 产生种群，对应产生496个个体，
    Population_list = []
    Chromos_obj_record = []
    for i in range(len(vectors)):
        Population_list.append(gerenate_chromo(Job_processnum, 0.5, pt, pt_index))
        Chromos_obj_record.append(fitness_calculation(Population_list[i]))
    EP = []  # 存储非支配解
    EP_fit = []  # 存储非支配解的目标函数值
    best_list = []  # 存储每一次迭代中的最优的个体
    z = list(np.array(Chromos_obj_record).min(axis=0))  # 选取理想点
    gen = 1
    while gen <= 100:  # 设置迭代次数
        for i in range(len(vectors)):  # 每一次迭代中，都对所有个体进行领域搜索操作
            B_i = generate_B_i(T, i, Eular_distance)  # 取个体的领域索引
            k, l = random.sample(B_i, 2)  # 在个体的领域中随意选择两个个体
            y_ = cross_and_mutation(Population_list[k], Population_list[l])  # 对这两个个体进行交叉、变异操作，产生新个体
            FV = fitness_calculation(y_)  # 计算新个体的适应度
            for ch in range(len(FV)):  # z与新个体y_比较，更新z
                if z[ch] > FV[ch]:
                    z[ch] = FV[ch]
            B_fit = []  # 存储领域中所有点的适应度
            for j in range(len(B_i)):
                B_fit.append(Chromos_obj_record[B_i[j]])
            B_fit = np.array(B_fit)  # 转为np数组方便计算
            B_fit_ = B_fit - z
            FV_ = np.array(FV) - z
            for j in range(len(B_i)):  # 比较g(x|lambda_j,z), 取其中的最优值
                if B_fit_[j].max() > FV_.max():  # 如果邻域中的个体不如Y_，则将这些个体更新为y_
                    Population_list[B_i[j]] = copy.deepcopy(y_)
                    Chromos_obj_record[B_i[j]] = copy.deepcopy(FV)
            if EP == []:  # 第一次Y_直接加入EP
                EP.append(y_)
                EP_fit.append(FV)
            else:
                num = 0
                EP_c = copy.deepcopy(EP)
                EP_fit_c = copy.deepcopy(EP_fit)
                while num != len(EP):  # 进行比较，删去EP中劣于y_的个体
                    # 如果有个体支配y_,则ep中不可能有个体被y_支配
                    if ((np.array(FV) - np.array(EP_fit[num])) >= 0).all() == True:
                        break
                    elif ((np.array(EP_fit[num]) - np.array(FV)) >= 0).all() == True:
                        EP_c.remove(EP[num])
                        EP_fit_c.remove((EP_fit[num]))
                    if num == len(EP) - 1:
                        EP_c.append(y_)
                        EP_fit_c.append(FV)
                    num += 1
                EP = EP_c
                EP_fit = EP_fit_c
        if len(EP) > 1:
            best_list.append(crowding_distance_calculation(EP_fit))
        else:
            best_list.append(EP[0])
        print("第%d次迭代" % gen)
        print("结果是：", EP_fit)
        gen += 1
    ax = plt.axes()
    x = np.array([i for i in range(len(best_list))])
    y_makespan = np.array([best_list[i][0] for i in range(len(best_list))])
    y_maxload = np.array([best_list[i][1] for i in range(len(best_list))])
    y_enerycost = np.array([best_list[i][2] for i in range(len(best_list))])
    ax.plot(x, y_makespan, '-r', label='makespan')
    ax.plot(x, y_maxload, '-g', label='maxload')
    ax.plot(x, y_enerycost, '-b', label='enerycost')
    plt.legend(loc='right')
    plt.xlabel("number of generation")
    plt.show()





