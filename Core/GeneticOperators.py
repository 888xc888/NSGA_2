"""
GeneticOperators.py
遗传算子模块 - 包含交叉和变异操作

Author: Refactored Code
Date: 2025-06-23
"""

import random
import numpy as np
from typing import List, Tuple
from Models.Individual import Individual
from Utils.ValidationUtils import ValidationUtils


class GeneticOperators:
    """遗传算子类"""
    
    @staticmethod
    def order_crossover(parent1: np.ndarray, parent2: np.ndarray, 
                       prioritization_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        顺序交叉（Order Crossover, OX）用于拆卸序列
        
        Args:
            parent1: 父代1的拆卸序列
            parent2: 父代2的拆卸序列
            prioritization_matrix: 优先级矩阵
            
        Returns:
            两个子代的拆卸序列
        """
        def single_crossover():
            length = len(parent1)
            
            # 随机选择交叉点
            index1 = random.randint(0, length - 1)
            index2 = random.randint(index1, length - 1)
            
            # 保留的基因段
            temp_gene1 = parent1[index1:index2 + 1]
            temp_gene2 = parent2[index1:index2 + 1]
            
            # 构建剩余基因序列
            temp1 = [x for x in parent2 if x not in temp_gene1]
            temp2 = [x for x in parent1 if x not in temp_gene2]
            
            # 构建子代
            offspring1 = [-1] * length
            offspring2 = [-1] * length
            
            # 填入保留的基因段
            for i in range(index1, index2 + 1):
                offspring1[i] = temp_gene1[i - index1]
                offspring2[i] = temp_gene2[i - index1]
            
            # 填入剩余基因
            for i in range(length):
                if offspring1[i] == -1:
                    offspring1[i] = temp1.pop(0)
                if offspring2[i] == -1:
                    offspring2[i] = temp2.pop(0)
            
            return np.array(offspring1), np.array(offspring2)
        
        # 重复生成直到满足优先级约束
        max_attempts = 100
        for _ in range(max_attempts):
            offspring1, offspring2 = single_crossover()
            if (ValidationUtils.validate_prioritization_matrix(offspring1, prioritization_matrix) and
                ValidationUtils.validate_prioritization_matrix(offspring2, prioritization_matrix)):
                return offspring1, offspring2
        
        # 如果无法生成有效子代，返回原父代
        return parent1.copy(), parent2.copy()
    
    @staticmethod
    def multi_point_crossover(parent1: np.ndarray, parent2: np.ndarray, 
                            n_points: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        多点交叉用于人机任务分配序列
        
        Args:
            parent1: 父代1的人机任务分配序列
            parent2: 父代2的人机任务分配序列
            n_points: 交叉点数量
            
        Returns:
            两个子代的人机任务分配序列
        """
        length = len(parent1)
        
        # 选择交叉点
        if n_points >= length:
            crossover_points = list(range(length))
        else:
            crossover_points = random.sample(range(length), n_points)
        
        # 复制父代
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # 在交叉点处交换基因
        for point in crossover_points:
            offspring1[point], offspring2[point] = offspring2[point], offspring1[point]
        
        return offspring1, offspring2
    
    @staticmethod
    def swap_mutation(sequence: np.ndarray, 
                     prioritization_matrix: np.ndarray) -> np.ndarray:
        """
        交换变异用于拆卸序列
        
        Args:
            sequence: 原序列
            prioritization_matrix: 优先级矩阵
            
        Returns:
            变异后的序列
        """
        max_attempts = 100
        
        for _ in range(max_attempts):
            # 复制原序列
            mutated = sequence.copy()
            
            # 随机选择两个位置进行交换
            length = len(sequence)
            pos1, pos2 = random.sample(range(length), 2)
            
            # 交换
            mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
            
            # 检查是否满足优先级约束
            if ValidationUtils.validate_prioritization_matrix(mutated, prioritization_matrix):
                return mutated
        
        # 如果无法生成有效变异，返回原序列
        return sequence.copy()
    
    @staticmethod
    def flip_mutation(sequence: np.ndarray) -> np.ndarray:
        """
        位翻转变异用于人机任务分配序列
        
        Args:
            sequence: 原序列
            
        Returns:
            变异后的序列
        """
        mutated = sequence.copy()
        
        # 随机选择一个位置进行翻转
        mutation_point = random.randint(0, len(sequence) - 1)
        mutated[mutation_point] = 1 - mutated[mutation_point]
        
        return mutated
    
    @staticmethod
    def crossover_individuals(parent1: Individual, parent2: Individual,
                            prioritization_matrix: np.ndarray) -> Tuple[Individual, Individual]:
        """
        个体级别的交叉操作
        
        Args:
            parent1: 父代个体1
            parent2: 父代个体2
            prioritization_matrix: 优先级矩阵
            
        Returns:
            两个子代个体
        """
        # 拆卸序列交叉
        offspring1_disassembly, offspring2_disassembly = GeneticOperators.order_crossover(
            parent1.disassembly_sequence, parent2.disassembly_sequence, prioritization_matrix
        )
        
        # 人机任务分配序列交叉
        offspring1_tasking, offspring2_tasking = GeneticOperators.multi_point_crossover(
            parent1.human_robot_tasking_sequence, parent2.human_robot_tasking_sequence
        )
        
        # 创建子代个体
        offspring1 = Individual(parent1.part_number)
        offspring1.disassembly_sequence = offspring1_disassembly
        offspring1.human_robot_tasking_sequence = offspring1_tasking
        
        offspring2 = Individual(parent2.part_number)
        offspring2.disassembly_sequence = offspring2_disassembly
        offspring2.human_robot_tasking_sequence = offspring2_tasking
        
        return offspring1, offspring2
    
    @staticmethod
    def mutate_individual(individual: Individual, 
                         prioritization_matrix: np.ndarray) -> Individual:
        """
        个体级别的变异操作
        
        Args:
            individual: 原个体
            prioritization_matrix: 优先级矩阵
            
        Returns:
            变异后的个体
        """
        mutated = individual.copy()
        
        # 拆卸序列变异
        mutated.disassembly_sequence = GeneticOperators.swap_mutation(
            individual.disassembly_sequence, prioritization_matrix
        )
        
        # 人机任务分配序列变异
        mutated.human_robot_tasking_sequence = GeneticOperators.flip_mutation(
            individual.human_robot_tasking_sequence
        )
        
        return mutated
