"""
Individual.py
个体类定义模块 - 定义NSGA-II算法中的个体结构

Author: Refactored Code
Date: 2025-06-23
"""

import numpy as np
from typing import List, Optional


class Individual:
    """
    NSGA-II算法中的个体类
    包含拆卸序列、人机任务分配、评价指标和NSGA-II相关参数
    """
    
    def __init__(self, part_number: int):
        """
        初始化个体
        
        Args:
            part_number: 零件数量
        """
        # 基本属性
        self.part_number = part_number
        
        # 拆卸属性
        self.disassembly_sequence = np.zeros(part_number, dtype=int)  # 拆卸序列
        self.human_robot_tasking_sequence = -1 * np.ones(part_number, dtype=int)  # 人机任务分配序列
        
        # 评价指标
        self.disassembly_complexity = 0.0  # 拆卸复杂度
        self.ergonomics = 0.0  # 人体工程学评价
        
        # NSGA-II算法参数
        self.n = 0  # 被支配的解的数量
        self.rank = 0  # 非支配层级
        self.S: List['Individual'] = []  # 该个体支配的解集合
        self.distance = 0.0  # 拥挤度距离
    
    def comparison(self, other: 'Individual') -> int:
        """
        与另一个个体比较支配关系
        
        Args:
            other: 比较的个体
            
        Returns:
            1: 当前个体支配other个体
            2: other个体支配当前个体
            0: 两者不互相支配
        """
        # 数值越小越好
        if (self.ergonomics < other.ergonomics and 
            self.disassembly_complexity < other.disassembly_complexity):
            return 1  # 当前个体支配other个体
        elif (self.ergonomics > other.ergonomics and 
              self.disassembly_complexity > other.disassembly_complexity):
            return 2  # other个体支配当前个体
        return 0  # 两者不互相支配
    
    def to_chromosome(self) -> np.ndarray:
        """
        将个体转换为染色体表示（用于遗传操作）
        
        Returns:
            numpy数组，前半部分为拆卸序列，后半部分为人机任务分配
        """
        return np.concatenate([self.disassembly_sequence, self.human_robot_tasking_sequence])
    
    def from_chromosome(self, chromosome: np.ndarray) -> None:
        """
        从染色体表示恢复个体
        
        Args:
            chromosome: 染色体数组
        """
        mid_point = len(chromosome) // 2
        self.disassembly_sequence = chromosome[:mid_point].copy()
        self.human_robot_tasking_sequence = chromosome[mid_point:].copy()
    
    def copy(self) -> 'Individual':
        """
        创建个体的深拷贝
        
        Returns:
            复制的个体
        """
        new_individual = Individual(self.part_number)
        new_individual.disassembly_sequence = self.disassembly_sequence.copy()
        new_individual.human_robot_tasking_sequence = self.human_robot_tasking_sequence.copy()
        new_individual.disassembly_complexity = self.disassembly_complexity
        new_individual.ergonomics = self.ergonomics
        new_individual.n = self.n
        new_individual.rank = self.rank
        new_individual.S = self.S.copy()
        new_individual.distance = self.distance
        return new_individual
    
    def __str__(self) -> str:
        """
        个体的字符串表示
        """
        return (f"Individual(rank={self.rank}, "
                f"complexity={self.disassembly_complexity:.3f}, "
                f"ergonomics={self.ergonomics:.3f}, "
                f"distance={self.distance:.3f})")
