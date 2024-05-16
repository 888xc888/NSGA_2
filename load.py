"""
本文件定义常见的加载函数
1.加载NSGA配置
2.加载描述-优先级矩阵
3.加载描述-识别结果
4.加载评价-拆卸复杂度
5.加载评价-人体工程学
"""
import numpy as np
import yaml


# ############### 导入NSGA配置函数 ################
def load_NSGA_config(config_path):
    # 打开并加载yaml文件
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
    NSGA_config = data['NSGA_config']
    population_size = NSGA_config['population_size']
    termination_condition = NSGA_config['termination_condition']
    crossover_probability = NSGA_config['mutation_probability']
    mutation_probability = NSGA_config['mutation_probability']
    return population_size, termination_condition, crossover_probability, mutation_probability


# ############### 导入优先级矩阵约束 ################
def load_priority_matrix(description_path):
    # 打开并加载yaml文件
    with open(description_path, 'r') as file:
        data = yaml.safe_load(file)
    # 获取优先级矩阵部分
    priority_matrix = data['prioritization_matrix']
    # 获取part_number和data子项
    matrix_data = np.array(priority_matrix['data'])  # 转化成numpy数组
    return matrix_data


# ############### 识别结果加载函数 ################
def load_recognition_results(description_path):
    # 打开并加载yaml文件
    with open(description_path, 'r') as file:
        data = yaml.safe_load(file)
    # 获取识别结果向量部分
    vector = data['recognition_results_vector']
    vector_data = np.array(vector['data'])
    return vector_data


# ############### 加载拆卸复杂度评价 ################
def load_disassembly_complexity(indicator_path):
    # 打开并加载yaml文件
    with open(indicator_path, 'r') as file:
        data = yaml.safe_load(file)
    disassembly_complexity = data['disassembly_complexity']
    return disassembly_complexity


# ############### 加载拆人体工程学评价 ################
def load_ergonomics(indicator_path):
    # 打开并加载yaml文件
    with open(indicator_path, 'r') as file:
        data = yaml.safe_load(file)
    ergonomics = data['ergonomics']
    return ergonomics
