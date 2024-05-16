"""
1.定义种群中个体类Individual
2.个体拆卸复杂度计算
3.个体人体工程学参数计算
4.实现拆卸序列是否满足优先级矩阵函数if_prioritization_matrix
5.NSGA种群初始化函数population_initialization
6.快速非支配排序函数
7.拥挤度计算函数
8.交叉算子+变异算子
9.精英选择算子
"""
from collections import defaultdict
import random
import copy

import numpy as np


# ######################################### 定义个体类 #########################################

class Individual(object):  # 定义个体类
    def __init__(self, part_number):

        # 拆卸属性
        self.part_number = part_number
        self.disassembly_sequence = np.zeros(part_number)  # 初始化全为0
        self.human_robot_tasking_sequence = -1 * np.ones(shape=(part_number,))  # 1表示操作工拆卸，0表示机械臂拆卸  # 初始化全为-1

        # 评价指标打分
        self.disassembly_complexity = 0.0
        self.ergonomics = 0.0

        # 关于NSGA的参数
        self.n = 0  # 解p被几个解支配
        self.rank = 0  # 解p所在层数
        self.S = []  # 解p支配解的集合
        self.distance = 0  # 拥挤度距离

    # 与两外一个实例个体比较，看是否支配
    def comparison(self, other):  # other是比较的个体
        # 数值越小越好
        if self.ergonomics < other.ergonomics and self.disassembly_complexity < other.disassembly_complexity:
            return 1  # 当前个体支配other个体
        elif self.ergonomics > other.ergonomics and self.disassembly_complexity > other.disassembly_complexity:
            return 2  # other个体支配当前个体
        return 0  # 两者不互相支配


# ######################################### 个体拆卸复杂度计算 ##########################################

def calculate_disassembly_complexity(individual, indicator_disassembly_complexity, recognition_results):
    """
    :param individual: 传入一个个体
    :param indicator_disassembly_complexity:传入拆卸复杂度评鉴指标，是字典，里面有两个键值对
    :param  recognition_results: 传入识别结果列表list
    :return disassembly_complexity: 拆卸复杂度值
    """
    disassembly_complexity = 0  # 这个数值直接输出最后的重心比较值，不再用三角模糊数了
    for i in range(individual.part_number):
        ordinal = individual.disassembly_sequence[i]  # 找出第i个要拆的零件是什么（取出其零件编号，注意，这里的编号是从1开始的）
        condition_number = recognition_results[ordinal - 1]  # 找出该中拆卸零件的情况种类编号
        if individual.human_robot_tasking_sequence[i] == 1:  # 判断是是否是操作工拆卸
            j = 0  # 这里就是做一个标记，后面为了索引找到对应的行
        else:
            j = 1
        # 0：正常；1：中等滑丝；2：严重滑丝；3：中等生锈；4：严重生锈
        # 三角模糊数的加法采用A(a,b,c),B(e,f,g);A+B=(a+e,b+f,c+g)
        # 三角模糊数的比较采用重心数比较：A(a,b,c),B(e,f,g);如果 (a+2b+c)/4 大于 (e+2f+g)/4 那么 A 大于 B
        if condition_number == 0:
            condition = 'screw_normal'
            # 三角模糊数求和
            result = [sum(x) for x in zip(indicator_disassembly_complexity[condition][j][0],
                                          indicator_disassembly_complexity[condition][j][1],
                                          indicator_disassembly_complexity[condition][j][2])]
            # 三角模糊求重心
            disassembly_complexity = (result[0] + 2 * result[1] + result[2]) / 4
        elif condition_number == 1 or condition_number == 2:
            condition = 'screw_slippage'
            if condition_number == 1:  # 中等滑丝
                # 三角模糊数求和
                result = [sum(x) for x in zip(indicator_disassembly_complexity[condition][j][0],
                                              indicator_disassembly_complexity[condition][j][1],
                                              indicator_disassembly_complexity[condition][j][2])]
                # 三角模糊求重心
                disassembly_complexity = (result[0] + 2 * result[1] + result[2]) / 4
            else:  # 严重滑丝
                # 三角模糊数求和
                result = [sum(x) for x in zip(indicator_disassembly_complexity[condition][j][3],
                                              indicator_disassembly_complexity[condition][j][4],
                                              indicator_disassembly_complexity[condition][j][5])]
                # 三角模糊求重心
                disassembly_complexity = (result[0] + 2 * result[1] + result[2]) / 4
        else:
            condition = 'screw_rust'
            if condition_number == 3:  # 中等生锈
                # 三角模糊数求和
                result = [sum(x) for x in zip(indicator_disassembly_complexity[condition][j][0],
                                              indicator_disassembly_complexity[condition][j][1],
                                              indicator_disassembly_complexity[condition][j][2])]
                # 三角模糊求重心
                disassembly_complexity = (result[0] + 2 * result[1] + result[2]) / 4
            else:  # 严重生锈
                # 三角模糊数求和
                result = [sum(x) for x in zip(indicator_disassembly_complexity[condition][j][3],
                                              indicator_disassembly_complexity[condition][j][4],
                                              indicator_disassembly_complexity[condition][j][5])]
                # 三角模糊求重心
                disassembly_complexity = (result[0] + 2 * result[1] + result[2]) / 4

    return disassembly_complexity


# ######################################### 个体人体工程学参数计算 ##########################################

def calculate_ergonomics(individual, indicator_ergonomics, recognition_results):
    """
    :param individual: 传入一个个体
    :param indicator_ergonomics: 传入人体工程学评鉴指标,是一个字典
    :param  recognition_results: 传入识别结果列表
    :return ergonomics: 人体工程学评价值
    """
    ergonomics = 0
    for i in range(individual.part_number):
        if individual.human_robot_tasking_sequence[i] == 1:  # 1表示操作工拆卸，0表示机械臂拆卸
            ordinal = individual.disassembly_sequence[i]  # 找出第i个要拆的零件是什么（取出其零件编号，注意，这里的编号是从1开始的）
            condition_number = recognition_results[ordinal - 1]  # 找出该中拆卸零件的情况种类编号
            # 0：正常；1：中等滑丝；2：严重滑丝；3：中等生锈；4：严重生锈
            if condition_number == 0:
                condition = 'screw_normal'
                for k in range(4):
                    ergonomics = ergonomics + indicator_ergonomics[condition][k][0]
            elif condition_number == 1 or condition_number == 2:
                condition = 'screw_slippage'
                if condition_number == 1:  # 中等滑丝
                    for k in range(4):
                        ergonomics = ergonomics + indicator_ergonomics[condition][k][0]
                else:  # 严重滑丝
                    for k in range(4):
                        ergonomics = ergonomics + indicator_ergonomics[condition][k][1]
            else:
                condition = 'screw_rust'
                if condition_number == 3:  # 中等生锈
                    for k in range(4):
                        ergonomics = ergonomics + indicator_ergonomics[condition][k][0]
                else:  # 严重生锈
                    for k in range(4):
                        ergonomics = ergonomics + indicator_ergonomics[condition][k][1]

    return ergonomics


# ######################################### 个体是否满足优先级约束检查 ##########################################

def if_prioritization_matrix(disassembly_sequence, prioritization_matrix):
    """
    :param disassembly_sequence: 传入拆卸序列 这个是一个一维ndarray
    :param prioritization_matrix: 传入优先级矩阵 这个是一个二维list
    :return: flag: True表示满足优先级约束，False表示不满足
    """
    flag = True  # 是否合法的标记
    length = len(disassembly_sequence)
    '''
    依次序列从第1个（从0开始）到倒数第二个拆卸对象开
    始检查，每一个拆卸对象是否符合拆卸优先级矩阵
    '''
    for i in range(length - 1):  # 最后一个不需要检查
        # di为某个拆卸对象的序号（从1开始标号）
        di = disassembly_sequence[i]
        # 查找该零件之后的零件是否对当前拆卸零件有约束，有约束则拆卸序列不合法
        for j in range(i + 1, length):
            # 检查零件dj对di是否有约束
            dj = disassembly_sequence[j]
            if prioritization_matrix[dj - 1][di - 1] == 1:  # 等于1表示dj对di有约束
                flag = False
                return flag

    return flag


# ######################################### 初始随即种群生成函数 #########################################

# 种群以矩阵的形式存储
# 生成矩阵的一行：[5,1,6,4,3,2,7,0,1,0,1,0,1,1]前7位为拆卸序列，后7位为人机分配情况
def population_initialization(part_number, population_size, prioritization_matrix):
    """
    :param part_number: 零件数量
    :param population_size: 种群大小
    :param prioritization_matrix: 优先级矩阵，用于判断产生的个体是否满足拆卸优先级关系
    :return: 返回初始化种群
    """
    # 定义可能的数字范围,(零件个数)
    numbers = np.arange(1, part_number + 1)
    # 创建一个空种群
    population = []
    while len(population) < population_size:
        # 生成拆卸序列，前半部分
        disassembly_sequence = np.random.choice(numbers, size=part_number, replace=False)
        # 判断拆卸序列是否满足优先级矩阵，不满足则重新生成拆卸序类
        while not if_prioritization_matrix(disassembly_sequence, prioritization_matrix):
            disassembly_sequence = np.random.choice(numbers, size=part_number, replace=False)
        # 生成人机任务分配序列，后半部分
        human_robot_tasking_sequence = np.random.randint(0, 2, size=part_number)
        # 上面两个序列拼起来生成单个个体
        individual = np.append(disassembly_sequence, human_robot_tasking_sequence)
        # 判断是否重复,不重复就加入
        if not any(np.all(individual == pop_item) for pop_item in population):
            population.append(individual)

    return population


# ######################################### 快速非支配排序函数 #########################################

def fast_non_dominated_sort(P):
    """
    非支配排序
    :param P: 种群P,以list的形似存储个体实例
    :return: F：分层结果，返回值类型为dict，键为层号，值为list（该层中的个体）
    """
    F = defaultdict(list)
    '''
    F是这样的数据形式
                  defaultdict(<class 'list'>,
                          {'key1': [1, 2],
                           'key2': ['apple'],
                           'key3': []})
                           
    '''

    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            # 数值越小越好
            tempt = p.comparison(q)  # 判断支配关系
            if tempt == 1:  # p支配q
                p.S.append(q)
            elif tempt == 2:  # q支配p
                p.n += 1
        if p.n == 0:
            p.rank = 1
            F[1].append(p)  # p属于第一层，最好
    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i += 1
        F[i] = Q
    return F


# ######################################### 拥挤度计算函数 #########################################
# 计算L列表中所有个体的拥挤距离，直接在L列表中修改
def crowding_distance_assignment(L):
    """
    计算拥挤度
    :param L: F[i]，是个list，为第i层的节点集合
    :return: 无  在python中，传入函数的参数如果是可变类型（如列表、字典、集合等），函数内部对这些参数的修改会影响到原始对象。
    """
    length = len(L)
    # 初始化距离
    for i in range(length):
        L[i].distance = 0
    # 遍历每个目标方向（有几个优化目标，就有几个目标方向，现在有'disassembly_complexity', 'ergonomics'）
    for objective in ['disassembly_complexity', 'ergonomics']:
        L.sort(key=lambda x: getattr(x, objective))  # 使用objective属性值排序 # sort默认升序
        # 最大最小的拥挤距离赋值为无穷大
        # 多个解在同一个目标函数维度上达到最大值或最小值，只需要给第一个最大值或最后一个最小值的个体赋值无穷大即可
        L[0].distance = float('inf')
        L[length - 1].distance = float('inf')
        # 找出最大最小两个数值，为了后面归一化求拥挤距离
        f_max = getattr(L[length - 1], objective)
        f_min = getattr(L[0], objective)
        # 当最大值和最小值相等时（后面归一化会出现除0错误），我们可以跳过该目标方向的拥挤距离计算
        if f_max == f_min:
            # print(f"目标方向 '{objective}' 上的最大值和最小值相等，均为 {f_max}，跳过该目标方向的拥挤距离计算")
            continue
        # 计算拥挤距离
        for i in range(1, length - 1):
            L[i].distance = L[i].distance + (getattr(L[i + 1], objective) - getattr(L[i - 1], objective)) / (
                    f_max - f_min)


# ######################################### 交叉算子+变异算子 #########################################

def crossover_mutation(parent, crossover_probability, mutation_probability, prioritization_matrix):
    """
    :param parent: 一个二维矩阵，外层是一个列表，内层是population_size个这样的个体，类型为ndarray
    :param crossover_probability: 交叉概率
    :param mutation_probability: 变异概率
    :param prioritization_matrix: 优先级矩阵，判断交叉变异后的拆卸序列是否满足优先级矩阵
    :return result: 父代和子代和并的结果，用于后面的精英选择策略
    """
    # 某个体[4，6，5，3，2，1，7，0，0，1，0，1，0，1，] 前7位表示拆卸序列的顺序，后七位表示依次对应的人或机器人的任务分配情况
    '''
    拆卸序列部分：[4，6，5，3，2，1，7]
                    交叉方式：在进化算法的交叉环节中，无论是一点交叉还是两点交叉，基因重组后产生的后代可能出现编码重复的情况，
                            那么就需要我们对产生的子代进行修订，常见修订算法有部分匹配交叉（PMX），顺序交叉（OX），循环交叉（CX）等
                            这里用顺序交叉（Order Crossover，OX）
                    变异方式：通过随机交换序列中的两个元素来实现变异。
    任务分配序列部分：[0，0，1，0，1，0，1，]
                    交叉方式：多点交叉：交换两个父代的n个位置的任务分配，这里的n我们先取2。
                    变异方式：通过反转某个位置上的二进制位来实现变异。  
    '''

    population_size = len(parent)  # 计算种群个数
    part_number = int(len(parent[0]) / 2)  # 计算零件个数 # 这样计算出来是浮点数，转化为整数才能索引

    # 拆卸序列部分，顺序交叉（Order Crossover，OX）
    def OX_crossover(parent1, parent2, prioritization_matrix):
        """
        :param parent1: 父代1
        :param parent2: 父代2
        :param prioritization_matrix: 优先级矩阵
        :return offspring1, offspring2: 两个子代
        """
        if not len(parent1) == len(parent2):
            return False  # 判度输入两个父代是否长度一样，不一样直接返回False
        length = len(parent1)

        def crossover(parent1, parent2):  # 再定义一个函数是为了代码的复用，当优先级检测不通过的时候再次运行

            index1 = random.randint(0, length - 1)
            index2 = random.randint(index1, length - 1)  # 从index1起（包括）往后挑一个index2(index1可以与index2一样)
            # print(index1, index2)

            # 保留的部分单独拿出来
            temp_gene1 = parent1[index1:index2 + 1]
            temp_gene2 = parent2[index1:index2 + 1]

            # parent1中剔除temp_gene2中有的元素，按照原来的顺序再次排好。另一个同理。
            temp1 = []
            temp2 = []
            for i in range(length):
                if parent2[i] not in temp_gene1:
                    temp1.append(parent2[i])
                if parent1[i] not in temp_gene2:
                    temp2.append(parent1[i])

            # offspring全部初始化-1，长度为length
            offspring1 = [-1 for _ in range(length)]
            offspring2 = [-1 for _ in range(length)]

            # 先把xndex1~index2从temp_gene填到offspring中
            for j in range(index1, index2 + 1):
                offspring1[j] = temp_gene1[j - index1]
                offspring2[j] = temp_gene2[j - index1]

            # 再把temp填到offspring中
            for k in range(0, length):
                if offspring1[k] == -1:
                    offspring1[k] = temp1.pop(0)  # 从temp中依次取出，并且从temp中剔除
                    offspring2[k] = temp2.pop(0)  # offspring1和offspring2是完全一样的所以这里直接写了
            return offspring1, offspring2

        offspring1, offspring2 = crossover(parent1, parent2)
        # 判断两个offspring是否满足prioritization_matrix
        while not (if_prioritization_matrix(offspring1, prioritization_matrix) and \
                   if_prioritization_matrix(offspring2, prioritization_matrix)):
            # 不满足优先级矩阵重新生成子代
            offspring1, offspring2 = crossover(parent1, parent2)

        return offspring1, offspring2

    # 拆卸序列部分，交换变异
    def switch_mutation(parent1, prioritization_matrix):
        """
        :param parent1: 父代
        :param prioritization_matrix: 优先级举证
        :return offspring1: 子代
        """
        seq_length = len(parent1)
        selected_numbers = random.sample(list(range(seq_length)), 2)  # 随机取出两个不同的整数
        # 拷贝一下 原来的parent1不去改变
        offspring1 = parent1.copy()
        # 交换两个位置的数值，直接通过索引赋值的方式一步到位完成，无需借助临时变量
        offspring1[selected_numbers[0]], offspring1[selected_numbers[1]] = offspring1[selected_numbers[1]], \
            offspring1[selected_numbers[0]]
        # 判断是否满足拆卸优先级矩阵，不满足重新来
        while not if_prioritization_matrix(offspring1, prioritization_matrix):
            selected_numbers = random.sample(list(range(seq_length)), 2)  # 随机取出两个不同的整数
            # 拷贝一下 原来的parent1不去改变
            offspring1 = parent1.copy()
            # 交换两个位置的数值，直接通过索引赋值的方式一步到位完成，无需借助临时变量
            offspring1[selected_numbers[0]], offspring1[selected_numbers[1]] = offspring1[selected_numbers[1]], \
                offspring1[selected_numbers[0]]
        return offspring1

    # 任务分配序列部分,多点交叉(默认两个父代随机交叉两个位置的任务分配结果)
    def multi_point_crossover(parent1, parent2, n_points=2):
        if not len(parent1) == len(parent2):
            return False  # 判度输入两个父代是否长度一样，不一样直接返回False
        seq_length = len(parent1)
        crossover_points = random.sample(list(range(seq_length)), n_points)  # 随机取出两个不同的整数
        # 拷贝一下 原来的parent不去改变
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        for point in crossover_points:
            # 交换对应位置的值
            temp = offspring1[point]
            offspring1[point] = offspring2[point]
            offspring2[point] = temp
        return offspring1, offspring2

    # 任务分配序列部分,随机挑选一个点变异
    def flip_bit_mutation(parent1):
        # 拷贝一下
        offspring1 = parent1.copy()
        seq_length = len(parent1)
        mutation_point = random.randint(0,
                                        seq_length - 1)  # 随机产生一个整数  # 这里的random.randint(0, seq_length)，seq_length可以取到
        offspring1[mutation_point] = 1 - offspring1[mutation_point]  # 实现0，1翻转
        return offspring1

    # 建立一个空offspring列表,里面每个元素是ndarray代表一个个体
    offspring = []

    # 首先进行交叉操作
    parent_temp = copy.deepcopy(parent)  # 原来的parent不动
    random.shuffle(parent_temp)  # 打乱列表顺序(只打乱最外层)，模拟随机取出
    pairs = [(parent_temp[i], parent_temp[i + 1]) for i in range(0, len(parent_temp), 2)]  # 两两取出准备交叉
    for pairs_i in pairs:
        if np.random.rand() < crossover_probability:  # s是否进行交叉
            # 先拆卸序列部分交叉
            temp1 = pairs_i[0][:part_number]
            temp2 = pairs_i[1][:part_number]
            offspring1_1, offspring2_1 = OX_crossover(temp1, temp2, prioritization_matrix)
            # 再人机任务分配部分交叉
            temp3 = pairs_i[0][part_number:]
            temp4 = pairs_i[1][part_number:]
            offspring1_2, offspring2_2 = multi_point_crossover(temp3, temp4, 2)
            offspring.append(np.concatenate((offspring1_1, offspring1_2)))
            offspring.append(np.concatenate((offspring2_1, offspring2_2)))
        else:  # 不交叉直接赋值
            offspring.append(pairs_i[0])
            offspring.append(pairs_i[1])

        # 再进行变异操作
        for offspring_i in offspring:
            if np.random.rand() < mutation_probability:
                # 先拆卸序列部分变异
                offspring_i[:part_number] = switch_mutation(offspring_i[:part_number], prioritization_matrix)
                # 再人机任务分配部分变异
                offspring_i[part_number:] = flip_bit_mutation(offspring_i[part_number:])

    # 父代子代合并（为了保留优秀个体），为了之后的精英选择策略
    result = parent + offspring  # 外层是一个list，内层是一个ndarray

    return result


# ######################################### 精英选择算子 #########################################
def select_elites(old, indicator_disassembly_complexity, indicator_ergonomics, recognition_results):
    """
    :param old: old是父代子代合并结果外层list，内层一个个体用一个ndarray
    :param indicator_disassembly_complexity: 传入拆卸复杂度评鉴指标，是字典
    :param indicator_ergonomics: 传入人体工程学评鉴指标,是一个字典
    :param recognition_results: 传入识别结果列表
    :return next_generation:
    """
    # 计算old中个体数量
    individual_number = len(old)
    # 种群数量
    population_size = individual_number / 2
    # 计算零件数量
    part_number = int(len(old[0]) / 2)  # 要转化为int后面才能索引

    # 将old列表全部转化为实例列表,并计算复杂度和人体工程学评价指标
    instance_list = [Individual(part_number) for _ in range(individual_number)]  # 空实例列表
    for i in range(individual_number):
        # individual_number是ndarray；old[i]也是ndarray
        # 每个实例disassembly_sequence赋值
        instance_list[i].disassembly_sequence = np.copy(old[i][:part_number])
        # 每个实例human_robot_tasking_sequence赋值
        instance_list[i].human_robot_tasking_sequence = np.copy(old[i][part_number:])
        # 计算每个个体实例的拆卸复杂度
        instance_list[i].disassembly_complexity = calculate_disassembly_complexity(instance_list[i], \
                                                                                   indicator_disassembly_complexity, \
                                                                                   recognition_results)
        # 计算每个个体实例的人体工程学参数
        instance_list[i].ergonomics = calculate_ergonomics(instance_list[i], indicator_ergonomics, recognition_results)

    # sort_list是一个defaultdict(list)
    sort_list = fast_non_dominated_sort(instance_list)
    # print([i.disassembly_sequence for i in sort_list[1]])

    # 下一代的实例列表
    next_generation_list = []
    for key, value in sort_list.items():
        # key是第几个支配层；value是这一层的实例列表
        if len(next_generation_list) < population_size:  # 可以往next_generation_list加入个体
            if len(next_generation_list) + len(value) <= population_size:
                # 表示这一支配层的个体可以整体全都加入next_generation_list
                next_generation_list.extend(value)
                if len(next_generation_list) == population_size:  # 刚好填满直接跳出
                    break
            else:  # 这一层要进行拥挤距离排序，根据拥挤距离排序加入next_generation_list
                # 对当前层级value列表中的每个个体的拥挤距离
                crowding_distance_assignment(value)
                value.sort(key=lambda x: x.distance, reverse=True)  # 使用distance属性值排序 # reverse = True为降序
                # 计算取出个数
                temp = int(population_size - len(next_generation_list))  # 转化为int才能给后面索引
                next_generation_list.extend(value[:temp])
                # 取出结束直接跳出
                break

    # 初始化新一代种群为空列表
    next_generation = []

    # 将next_generation_list转化到next_generation
    for next_generation_individual in next_generation_list:
        next_generation.append(np.concatenate((next_generation_individual.disassembly_sequence, \
                                               next_generation_individual.human_robot_tasking_sequence)))

    return next_generation
