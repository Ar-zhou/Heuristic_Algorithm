# * Zhou Hong
# * hongzhou_ie@163.com
# * Nanjing University of Aeronautics and Astronautics

# from gurobipy import *
import copy
import math
import numpy as np
import pandas as pd
import time
import matplotlib as plt


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


def generate_weight_vector():

