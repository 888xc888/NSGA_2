import numpy as np
import math
import os
import time
# 用于绘图，输出迭代过程
import matplotlib.pyplot as plt
# 项目函数导入
import load
import calculate


if __name__ == '__main__':
    # 文件路径
    item_path = os.getcwd()
    config_path = item_path + '/config.yaml'
    description_path = item_path + '/description.yaml'
    indicator_path = item_path + '/indicator.yaml'
    # ########################### 存储在config.yaml中的 ###########################
    # 加载NSGA参数 （依次是：群体大小，遗传算法终止进化代数，交叉概率，变异概率）
    population_size, termination_condition, \
        crossover_probability, mutation_probability = load.load_NSGA_config(config_path)
    # ########################### 存储在description.yaml中的 ###########################
    # 加载优先级矩阵
    prioritization_matrix = load.load_priority_matrix(description_path)  # 加载进来是个矩阵（二维列表）
    print('********** 优先级矩阵 **********')
    print(prioritization_matrix)
    # 加载识别结果
    recognition_results = load.load_recognition_results(description_path)  # 加载进来是个向量（一维列表）
    print('********** 识别结果 **********')
    print(recognition_results)
    # ########################### 存储在indicator.yaml中的 ###########################
    # 加载拆卸复杂性评价指标
    indicator_disassembly_complexity = load.load_disassembly_complexity(indicator_path)  # 加载进来是个字典
    print('********** 拆卸复杂性评价指标 **********')
    print(indicator_disassembly_complexity)
    # 加载人体工程学评价指标
    indicator_ergonomics = load.load_ergonomics(indicator_path)  # 加载进来是个字典
    print('********** 人体工程学评价指标 **********')
    print(indicator_ergonomics)

    # 获取零件个数
    part_number = len(recognition_results)
    # 种群初始化
    initia_population = calculate.population_initialization(part_number, population_size, prioritization_matrix)
    print('********** 初始种群 **********')
    print(initia_population)
    population = initia_population

    temp_list_1 = []
    temp_list_2 = []
    # 开始循环
    for i in range(termination_condition):
        print('********** 第{}次循环 **********'.format(i + 1))
        # 交叉+变异生成原来的+新生成的
        old = calculate.crossover_mutation(population, crossover_probability, mutation_probability, prioritization_matrix)
        population = calculate.select_elites(old, indicator_disassembly_complexity, indicator_ergonomics, recognition_results)
        # 打印结果
        i = 1
        for individual in population:
            print('方案{}:'.format(i))
            print(individual[:part_number])
            print(individual[part_number:])
            i = i + 1













