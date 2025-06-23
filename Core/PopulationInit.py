"""
PopulationInitializer.py
种群初始化模块 - 负责生成初始种群

Author: Refactored Code
Date: 2025-06-23
"""

import random
import numpy as np
from typing import List
from Models.Individual import Individual
from Utils.ValidationUtils import ValidationUtils


class PopulationInitializer:
    """种群初始化器"""
    
    @staticmethod
    def initialize_population(population_size: int, part_number: int, 
                            prioritization_matrix: np.ndarray) -> List[Individual]:
        """
        初始化种群
        
        Args:
            population_size: 种群大小
            part_number: 零件数量
            prioritization_matrix: 优先级矩阵
            
        Returns:
            初始化的种群
        """
        population = []
        numbers = np.arange(1, part_number + 1)
        max_attempts = 1000  # 最大尝试次数
        
        print(f"正在生成初始种群（目标大小: {population_size}）...")
        
        while len(population) < population_size:
            attempts = 0
            individual = None
            
            # 尝试生成有效个体
            while attempts < max_attempts:
                attempts += 1
                
                # 生成拆卸序列
                disassembly_sequence = np.random.choice(numbers, size=part_number, replace=False)
                
                # 验证是否满足优先级约束
                if ValidationUtils.validate_prioritization_matrix(disassembly_sequence, prioritization_matrix):
                    # 生成人机任务分配序列
                    human_robot_sequence = np.random.randint(0, 2, size=part_number)
                    
                    # 创建个体
                    individual = Individual(part_number)
                    individual.disassembly_sequence = disassembly_sequence
                    individual.human_robot_tasking_sequence = human_robot_sequence
                    
                    # 检查是否与现有个体重复
                    if not PopulationInitializer._is_duplicate(individual, population):
                        break
            
            if individual is not None:
                population.append(individual)
                if len(population) % 5 == 0:  # 每生成5个个体显示一次进度
                    print(f"已生成 {len(population)}/{population_size} 个个体")
            else:
                print(f"警告：无法生成第 {len(population) + 1} 个有效个体，跳过")
                # 如果无法生成有效个体，可以考虑降低约束或使用修复策略
                individual = PopulationInitializer._generate_fallback_individual(
                    part_number, prioritization_matrix, population
                )
                if individual is not None:
                    population.append(individual)
        
        print(f"初始种群生成完成，共 {len(population)} 个个体")
        return population
    
    @staticmethod
    def _is_duplicate(individual: Individual, population: List[Individual]) -> bool:
        """检查个体是否重复"""
        for existing in population:
            if (np.array_equal(individual.disassembly_sequence, existing.disassembly_sequence) and
                np.array_equal(individual.human_robot_tasking_sequence, existing.human_robot_tasking_sequence)):
                return True
        return False
    
    @staticmethod
    def _generate_fallback_individual(part_number: int, prioritization_matrix: np.ndarray,
                                    existing_population: List[Individual]) -> Individual:
        """生成后备个体"""
        # 使用简单的顺序策略
        individual = Individual(part_number)
        individual.disassembly_sequence = np.arange(1, part_number + 1)
        individual.human_robot_tasking_sequence = np.random.randint(0, 2, size=part_number)
        return individual
