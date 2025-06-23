"""
ValidationUtils.py
验证工具模块 - 包含各种验证函数

Author: Refactored Code
Date: 2025-06-23
"""

import numpy as np
from typing import List


class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def validate_prioritization_matrix(disassembly_sequence: np.ndarray, 
                                     prioritization_matrix: np.ndarray) -> bool:
        """
        验证拆卸序列是否满足优先级矩阵约束
        
        Args:
            disassembly_sequence: 拆卸序列（1-based索引）
            prioritization_matrix: 优先级矩阵
            
        Returns:
            True表示满足优先级约束，False表示不满足
        """
        length = len(disassembly_sequence)
        
        # 依次检查每个拆卸对象是否符合拆卸优先级矩阵
        for i in range(length - 1):  # 最后一个不需要检查
            # di为某个拆卸对象的序号（从1开始标号）
            di = disassembly_sequence[i]
            
            # 查找该零件之后的零件是否对当前拆卸零件有约束
            for j in range(i + 1, length):
                # 检查零件dj对di是否有约束
                dj = disassembly_sequence[j]
                if prioritization_matrix[dj - 1][di - 1] == 1:  # 等于1表示dj对di有约束
                    return False
        
        return True
    
    @staticmethod
    def validate_sequence_integrity(sequence: np.ndarray, part_number: int) -> bool:
        """
        验证序列的完整性（是否包含所有零件且无重复）
        
        Args:
            sequence: 拆卸序列
            part_number: 零件总数
            
        Returns:
            True表示序列完整，False表示不完整
        """
        if len(sequence) != part_number:
            return False
        
        # 检查是否包含1到part_number的所有数字
        expected_set = set(range(1, part_number + 1))
        actual_set = set(sequence)
        
        return expected_set == actual_set
    
    @staticmethod
    def validate_human_robot_sequence(sequence: np.ndarray) -> bool:
        """
        验证人机任务分配序列的有效性
        
        Args:
            sequence: 人机任务分配序列（应该只包含0和1）
            
        Returns:
            True表示序列有效，False表示无效
        """
        return np.all(np.isin(sequence, [0, 1]))
    
    @staticmethod
    def validate_individual_chromosome(chromosome: np.ndarray, part_number: int, 
                                     prioritization_matrix: np.ndarray) -> bool:
        """
        验证个体染色体的有效性
        
        Args:
            chromosome: 个体染色体
            part_number: 零件数量
            prioritization_matrix: 优先级矩阵
            
        Returns:
            True表示染色体有效，False表示无效
        """
        if len(chromosome) != 2 * part_number:
            return False
        
        disassembly_seq = chromosome[:part_number]
        human_robot_seq = chromosome[part_number:]
        
        # 验证拆卸序列完整性
        if not ValidationUtils.validate_sequence_integrity(disassembly_seq, part_number):
            return False
        
        # 验证人机任务分配序列
        if not ValidationUtils.validate_human_robot_sequence(human_robot_seq):
            return False
        
        # 验证优先级约束
        if not ValidationUtils.validate_prioritization_matrix(disassembly_seq, prioritization_matrix):
            return False
        
        return True
