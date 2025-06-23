"""
NSGAOptimizer.py
NSGA-II优化器主类 - 整合所有模块的主算法实现

Author: Refactored Code
Date: 2025-06-23
"""

import time
import random
import numpy as np
from typing import List, Dict, Any, Optional
from Models.Individual import Individual
from Core.NSGACore import NSGACore
from Core.GeneticOperators import GeneticOperators
from Core.PopulationInit import PopulationInitializer
from Config.ConfigManager import ConfigManager, NSGAConfig
from Utils.EvaluationUtils import EvaluationUtils
from Output.OutputManager import OutputManager


class NSGAOptimizer:
    """
    NSGA-II优化器主类
    整合所有模块，提供完整的多目标优化功能
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化NSGA-II优化器
        
        Args:
            config_manager: 配置管理器
        """
        self.config_manager = config_manager
        self.nsga_config: NSGAConfig = None
        self.output_manager: OutputManager = None
        
        # 问题相关数据
        self.prioritization_matrix: np.ndarray = None
        self.recognition_results: np.ndarray = None
        self.disassembly_complexity_indicators: Dict[str, Any] = None
        self.ergonomics_indicators: Dict[str, Any] = None
        self.part_number: int = 0
        
        # 进化历史记录
        self.evolution_history: List[Dict[str, Any]] = []
        
        # 初始化
        self._initialize()
    
    def _initialize(self) -> None:
        """初始化优化器"""
        # 加载配置
        self.nsga_config = self.config_manager.load_nsga_config()
        system_config = self.config_manager.load_system_config()
        
        # 初始化输出管理器
        self.output_manager = OutputManager(system_config.output_directory)
        
        # 加载问题数据
        self.prioritization_matrix = self.config_manager.load_prioritization_matrix()
        self.recognition_results = self.config_manager.load_recognition_results()
        self.disassembly_complexity_indicators = self.config_manager.load_disassembly_complexity_indicators()
        self.ergonomics_indicators = self.config_manager.load_ergonomics_indicators()
        
        # 计算零件数量
        self.part_number = len(self.recognition_results)
        
        print("NSGA-II优化器初始化完成")
        print(f"零件数量: {self.part_number}")
        print(f"种群大小: {self.nsga_config.population_size}")
        print(f"最大代数: {self.nsga_config.termination_condition}")
    
    def _evaluate_individual(self, individual: Individual) -> None:
        """
        评价个体的目标函数值
        
        Args:
            individual: 待评价的个体
        """
        # 计算拆卸复杂度
        individual.disassembly_complexity = EvaluationUtils.calculate_disassembly_complexity(
            individual, self.disassembly_complexity_indicators, self.recognition_results
        )
        
        # 计算人体工程学评价
        individual.ergonomics = EvaluationUtils.calculate_ergonomics(
            individual, self.ergonomics_indicators, self.recognition_results
        )
    
    def _evaluate_population(self, population: List[Individual]) -> None:
        """
        评价整个种群
        
        Args:
            population: 种群
        """
        for individual in population:
            self._evaluate_individual(individual)
    
    def _create_offspring(self, population: List[Individual]) -> List[Individual]:
        """
        创建子代种群
        
        Args:
            population: 父代种群
            
        Returns:
            子代种群
        """
        offspring = []
        population_size = len(population)
        
        # 使用锦标赛选择和遗传操作创建子代
        while len(offspring) < population_size:
            # 选择父代
            parent1 = NSGACore.tournament_selection(population)
            parent2 = NSGACore.tournament_selection(population)
            
            # 交叉操作
            if random.random() < self.nsga_config.crossover_probability:
                child1, child2 = GeneticOperators.crossover_individuals(
                    parent1, parent2, self.prioritization_matrix
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异操作
            if random.random() < self.nsga_config.mutation_probability:
                child1 = GeneticOperators.mutate_individual(child1, self.prioritization_matrix)
            
            if random.random() < self.nsga_config.mutation_probability:
                child2 = GeneticOperators.mutate_individual(child2, self.prioritization_matrix)
            
            offspring.extend([child1, child2])
        
        # 如果子代数量超过目标数量，随机移除多余的个体
        if len(offspring) > population_size:
            offspring = offspring[:population_size]
        
        return offspring
    
    def _record_generation_statistics(self, generation: int, population: List[Individual]) -> None:
        """
        记录代数统计信息
        
        Args:
            generation: 代数
            population: 当前种群
        """
        if not population:
            return
        
        complexity_values = [ind.disassembly_complexity for ind in population]
        ergonomics_values = [ind.ergonomics for ind in population]
        
        stats = {
            'generation': generation,
            'population_size': len(population),
            'best_complexity': min(complexity_values),
            'avg_complexity': np.mean(complexity_values),
            'worst_complexity': max(complexity_values),
            'best_ergonomics': min(ergonomics_values),
            'avg_ergonomics': np.mean(ergonomics_values),
            'worst_ergonomics': max(ergonomics_values),
            'pareto_front_size': len([ind for ind in population if ind.rank == 1])
        }
        
        self.evolution_history.append(stats)
    
    def optimize(self, verbose: bool = True, plot_interval: int = 50) -> List[Individual]:
        """
        执行NSGA-II优化
        
        Args:
            verbose: 是否显示详细输出
            plot_interval: 绘图间隔
            
        Returns:
            最终的帕累托最优解集
        """
        start_time = time.time()
        
        if verbose:
            print("\n开始NSGA-II优化...")
        
        # 1. 初始化种群
        if verbose:
            print("正在初始化种群...")
        
        population = PopulationInitializer.initialize_population(
            self.nsga_config.population_size,
            self.part_number,
            self.prioritization_matrix
        )
        
        # 2. 评价初始种群
        self._evaluate_population(population)
        
        # 3. 主进化循环
        for generation in range(1, self.nsga_config.termination_condition + 1):
            if verbose and generation % 10 == 0:
                print(f"正在处理第 {generation} 代...")
            
            # 创建子代
            offspring = self._create_offspring(population)
            
            # 评价子代
            self._evaluate_population(offspring)
            
            # 合并父代和子代
            combined_population = population + offspring
            
            # 环境选择（保留最优个体）
            population = NSGACore.environmental_selection(
                combined_population, self.nsga_config.population_size
            )
            
            # 记录统计信息
            self._record_generation_statistics(generation, population)
            
            # 定期输出进度和绘图
            if verbose and generation % 50 == 0:
                self.output_manager.print_generation_summary(generation, population)
            
            if generation % plot_interval == 0:
                self.output_manager.plot_pareto_front(population, generation)
        
        # 4. 优化完成
        end_time = time.time()
        execution_time = end_time - start_time
        
        if verbose:
            self.output_manager.print_final_summary(
                population, self.nsga_config.termination_condition, execution_time
            )
        
        # 5. 保存结果
        self._save_results(population, execution_time)
        
        # 返回帕累托最优解
        pareto_optimal = [ind for ind in population if ind.rank == 1]
        return pareto_optimal
    
    def _save_results(self, final_population: List[Individual], execution_time: float) -> None:
        """
        保存优化结果
        
        Args:
            final_population: 最终种群
            execution_time: 执行时间
        """
        print("\n正在保存结果...")
        
        # 保存最终结果
        result_file = self.output_manager.save_final_results(
            final_population, self.nsga_config.termination_condition
        )
        print(f"最终结果已保存到: {result_file}")
        
        # 保存进化历史
        history_file = self.output_manager.save_evolution_history(self.evolution_history)
        print(f"进化历史已保存到: {history_file}")
        
        # 绘制收敛曲线
        convergence_plot = self.output_manager.plot_convergence_curve(self.evolution_history)
        if convergence_plot:
            print(f"收敛曲线已保存到: {convergence_plot}")
        
        # 绘制最终帕累托前沿
        pareto_plot = self.output_manager.plot_pareto_front(
            final_population, self.nsga_config.termination_condition
        )
        if pareto_plot:
            print(f"帕累托前沿图已保存到: {pareto_plot}")
        
        # 导出最优解详细信息
        best_solutions_file = self.output_manager.export_best_solutions(final_population)
        print(f"最优解详细信息已保存到: {best_solutions_file}")
        
        print("所有结果保存完成！")
    
    def get_best_solutions(self, population: Optional[List[Individual]] = None, 
                          top_n: int = 5) -> List[Individual]:
        """
        获取最优解
        
        Args:
            population: 种群（如果为None，则使用当前最优解）
            top_n: 返回前N个解
            
        Returns:
            最优解列表
        """
        if population is None:
            # 如果没有提供种群，需要先运行优化
            raise ValueError("请先运行optimize()方法或提供种群")
        
        # 选择第一层级的解
        pareto_optimal = [ind for ind in population if ind.rank == 1]
        
        if len(pareto_optimal) > top_n:
            # 按拥挤度距离排序
            pareto_optimal.sort(key=lambda x: x.distance, reverse=True)
            pareto_optimal = pareto_optimal[:top_n]
        
        return pareto_optimal
    
    def analyze_solutions(self, solutions: List[Individual]) -> Dict[str, Any]:
        """
        分析解的特征
        
        Args:
            solutions: 解列表
            
        Returns:
            分析结果字典
        """
        if not solutions:
            return {}
        
        complexity_values = [sol.disassembly_complexity for sol in solutions]
        ergonomics_values = [sol.ergonomics for sol in solutions]
        
        # 统计人机任务分配
        total_tasks = 0
        human_tasks = 0
        for sol in solutions:
            total_tasks += len(sol.human_robot_tasking_sequence)
            human_tasks += np.sum(sol.human_robot_tasking_sequence)
        
        analysis = {
            'solution_count': len(solutions),
            'complexity_stats': {
                'min': min(complexity_values),
                'max': max(complexity_values),
                'mean': np.mean(complexity_values),
                'std': np.std(complexity_values)
            },
            'ergonomics_stats': {
                'min': min(ergonomics_values),
                'max': max(ergonomics_values),
                'mean': np.mean(ergonomics_values),
                'std': np.std(ergonomics_values)
            },
            'task_allocation': {
                'human_ratio': human_tasks / total_tasks if total_tasks > 0 else 0,
                'robot_ratio': (total_tasks - human_tasks) / total_tasks if total_tasks > 0 else 0
            }
        }
        
        return analysis
