"""
NSGACore.py
NSGA-II算法核心模块 - 包含非支配排序和拥挤度计算

Author: Refactored Code
Date: 2025-06-23
"""

import random
from collections import defaultdict
from typing import List, Dict
from Models.Individual import Individual


class NSGACore:
    """NSGA-II算法核心类"""
    
    @staticmethod
    def fast_non_dominated_sort(population: List[Individual]) -> Dict[int, List[Individual]]:
        """
        快速非支配排序
        
        Args:
            population: 种群列表
            
        Returns:
            分层结果字典，键为层号，值为该层的个体列表
        """
        F = defaultdict(list)
        
        # 初始化每个个体的参数
        for p in population:
            p.S = []
            p.n = 0
            
            # 与其他个体比较支配关系
            for q in population:
                comparison_result = p.comparison(q)
                if comparison_result == 1:  # p支配q
                    p.S.append(q)
                elif comparison_result == 2:  # q支配p
                    p.n += 1
            
            # 如果没有被任何个体支配，则属于第一层
            if p.n == 0:
                p.rank = 1
                F[1].append(p)
        
        # 逐层构建非支配前沿
        i = 1
        while F[i]:
            Q = []
            for p in F[i]:
                for q in p.S:
                    q.n -= 1
                    if q.n == 0:
                        q.rank = i + 1
                        Q.append(q)
            i += 1
            F[i] = Q
        
        return F
    
    @staticmethod
    def crowding_distance_assignment(population: List[Individual]) -> None:
        """
        拥挤度距离分配
        
        Args:
            population: 同一层级的个体列表
        """
        length = len(population)
        if length <= 2:
            # 如果个体数量少于等于2，直接设为无穷大
            for individual in population:
                individual.distance = float('inf')
            return
        
        # 初始化距离
        for individual in population:
            individual.distance = 0.0
        
        # 遍历每个目标函数
        for objective in ['disassembly_complexity', 'ergonomics']:
            # 按目标函数值排序
            population.sort(key=lambda x: getattr(x, objective))
            
            # 边界个体设为无穷大
            population[0].distance = float('inf')
            population[length - 1].distance = float('inf')
            
            # 获取目标函数的最大值和最小值
            f_max = getattr(population[length - 1], objective)
            f_min = getattr(population[0], objective)
            
            # 如果最大值等于最小值，跳过该目标函数
            if f_max == f_min:
                continue
            
            # 计算中间个体的拥挤度距离
            for i in range(1, length - 1):
                if population[i].distance != float('inf'):
                    distance_increment = (
                        getattr(population[i + 1], objective) - 
                        getattr(population[i - 1], objective)
                    ) / (f_max - f_min)
                    population[i].distance += distance_increment
    
    @staticmethod
    def tournament_selection(population: List[Individual], tournament_size: int = 2) -> Individual:
        """
        锦标赛选择
        
        Args:
            population: 种群
            tournament_size: 锦标赛大小
            
        Returns:
            选中的个体
        """
        import random
        
        # 随机选择tournament_size个个体
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # 选择最优个体（rank最小，如果rank相同则distance最大）
        best = tournament[0]
        for individual in tournament[1:]:
            if (individual.rank < best.rank or 
                (individual.rank == best.rank and individual.distance > best.distance)):
                best = individual
        
        return best
    
    @staticmethod
    def environmental_selection(population: List[Individual], 
                              target_size: int) -> List[Individual]:
        """
        环境选择（保留最优个体）
        
        Args:
            population: 当前种群
            target_size: 目标种群大小
            
        Returns:
            选择后的种群
        """
        if len(population) <= target_size:
            return population
        
        # 非支配排序
        fronts = NSGACore.fast_non_dominated_sort(population)
        
        next_generation = []
        
        # 按层级添加个体
        for rank in sorted(fronts.keys()):
            if not fronts[rank]:  # 空层级
                continue
                
            if len(next_generation) + len(fronts[rank]) <= target_size:
                # 整层可以加入
                next_generation.extend(fronts[rank])
            else:
                # 需要从当前层级中选择部分个体
                remaining_slots = target_size - len(next_generation)
                if remaining_slots > 0:
                    # 计算拥挤度距离
                    NSGACore.crowding_distance_assignment(fronts[rank])
                    # 按拥挤度距离降序排序
                    fronts[rank].sort(key=lambda x: x.distance, reverse=True)
                    # 选择前remaining_slots个个体
                    next_generation.extend(fronts[rank][:remaining_slots])
                break
        
        return next_generation
