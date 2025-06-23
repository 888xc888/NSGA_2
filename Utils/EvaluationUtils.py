"""
EvaluationUtils.py
评价工具模块 - 包含个体评价相关的工具函数

Author: Refactored Code
Date: 2025-06-23
"""

import numpy as np
from typing import Dict, Any
from Models.Individual import Individual


class EvaluationUtils:
    """评价工具类"""
    
    @staticmethod
    def calculate_disassembly_complexity(individual: Individual, 
                                       indicator_disassembly_complexity: Dict[str, Any],
                                       recognition_results: np.ndarray) -> float:
        """
        计算个体的拆卸复杂度
        
        Args:
            individual: 个体对象
            indicator_disassembly_complexity: 拆卸复杂度评价指标
            recognition_results: 识别结果列表
            
        Returns:
            拆卸复杂度值
        """
        disassembly_complexity = 0.0
        
        for i in range(individual.part_number):
            # 找出第i个要拆的零件编号（从1开始）
            ordinal = individual.disassembly_sequence[i]
            # 找出该零件的情况种类编号
            condition_number = recognition_results[ordinal - 1]
            
            # 判断是操作工拆卸(1)还是机械臂拆卸(0)
            j = 0 if individual.human_robot_tasking_sequence[i] == 1 else 1
            
            # 根据不同的零件状况计算复杂度
            complexity_value = EvaluationUtils._calculate_single_part_complexity(
                condition_number, j, indicator_disassembly_complexity
            )
            disassembly_complexity += complexity_value
        
        return disassembly_complexity
    
    @staticmethod
    def _calculate_single_part_complexity(condition_number: int, operator_type: int,
                                        indicator_disassembly_complexity: Dict[str, Any]) -> float:
        """
        计算单个零件的拆卸复杂度
        
        Args:
            condition_number: 零件状况编号（0：正常；1：中等滑丝；2：严重滑丝；3：中等生锈；4：严重生锈）
            operator_type: 操作者类型（0：操作工，1：机械臂）
            indicator_disassembly_complexity: 拆卸复杂度评价指标
            
        Returns:
            单个零件的拆卸复杂度
        """
        if condition_number == 0:
            condition = 'screw_normal'
            result = [sum(x) for x in zip(
                indicator_disassembly_complexity[condition]['data'][operator_type][0],
                indicator_disassembly_complexity[condition]['data'][operator_type][1],
                indicator_disassembly_complexity[condition]['data'][operator_type][2]
            )]
        elif condition_number in [1, 2]:
            condition = 'screw_slippage'
            if condition_number == 1:  # 中等滑丝
                result = [sum(x) for x in zip(
                    indicator_disassembly_complexity[condition]['data'][operator_type][0],
                    indicator_disassembly_complexity[condition]['data'][operator_type][1],
                    indicator_disassembly_complexity[condition]['data'][operator_type][2]
                )]
            else:  # 严重滑丝
                result = [sum(x) for x in zip(
                    indicator_disassembly_complexity[condition]['data'][operator_type][3],
                    indicator_disassembly_complexity[condition]['data'][operator_type][4],
                    indicator_disassembly_complexity[condition]['data'][operator_type][5]
                )]
        else:  # condition_number in [3, 4]
            condition = 'screw_rust'
            if condition_number == 3:  # 中等生锈
                result = [sum(x) for x in zip(
                    indicator_disassembly_complexity[condition]['data'][operator_type][0],
                    indicator_disassembly_complexity[condition]['data'][operator_type][1],
                    indicator_disassembly_complexity[condition]['data'][operator_type][2]
                )]
            else:  # 严重生锈
                result = [sum(x) for x in zip(
                    indicator_disassembly_complexity[condition]['data'][operator_type][3],
                    indicator_disassembly_complexity[condition]['data'][operator_type][4],
                    indicator_disassembly_complexity[condition]['data'][operator_type][5]
                )]
        
        # 三角模糊数重心计算
        return (result[0] + 2 * result[1] + result[2]) / 4
    
    @staticmethod
    def calculate_ergonomics(individual: Individual,
                           indicator_ergonomics: Dict[str, Any],
                           recognition_results: np.ndarray) -> float:
        """
        计算个体的人体工程学评价
        
        Args:
            individual: 个体对象
            indicator_ergonomics: 人体工程学评价指标
            recognition_results: 识别结果列表
            
        Returns:
            人体工程学评价值
        """
        ergonomics = 0.0
        
        for i in range(individual.part_number):
            # 只有操作工拆卸才计算人体工程学评价
            if individual.human_robot_tasking_sequence[i] == 1:
                # 找出第i个要拆的零件编号（从1开始）
                ordinal = individual.disassembly_sequence[i]
                # 找出该零件的情况种类编号
                condition_number = recognition_results[ordinal - 1]
                
                ergonomics_value = EvaluationUtils._calculate_single_part_ergonomics(
                    condition_number, indicator_ergonomics
                )
                ergonomics += ergonomics_value
        
        return ergonomics
    
    @staticmethod
    def _calculate_single_part_ergonomics(condition_number: int,
                                        indicator_ergonomics: Dict[str, Any]) -> float:
        """
        计算单个零件的人体工程学评价
        
        Args:
            condition_number: 零件状况编号
            indicator_ergonomics: 人体工程学评价指标
            
        Returns:
            单个零件的人体工程学评价值
        """
        ergonomics_value = 0.0
        
        if condition_number == 0:
            condition = 'screw_normal'
            for k in range(4):
                ergonomics_value += indicator_ergonomics[condition]['data'][k][0]
        elif condition_number in [1, 2]:
            condition = 'screw_slippage'
            if condition_number == 1:  # 中等滑丝
                for k in range(4):
                    ergonomics_value += indicator_ergonomics[condition]['data'][k][0]
            else:  # 严重滑丝
                for k in range(4):
                    ergonomics_value += indicator_ergonomics[condition]['data'][k][1]
        else:  # condition_number in [3, 4]
            condition = 'screw_rust'
            if condition_number == 3:  # 中等生锈
                for k in range(4):
                    ergonomics_value += indicator_ergonomics[condition]['data'][k][0]
            else:  # 严重生锈
                for k in range(4):
                    ergonomics_value += indicator_ergonomics[condition]['data'][k][1]
        
        return ergonomics_value
