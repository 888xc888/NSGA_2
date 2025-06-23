"""
OutputManager.py
输出管理模块 - 负责结果输出、可视化和日志记录

Author: Refactored Code
Date: 2025-06-23
"""

import os
import json
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
from Models.Individual import Individual


class OutputManager:
    """输出管理器"""
    
    def __init__(self, output_dir: str = "Output"):
        """
        初始化输出管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def ensure_output_directory(self) -> None:
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def save_final_results(self, final_population: List[Individual], 
                          generation: int, format_type: str = "json") -> str:
        """
        保存最终结果
        
        Args:
            final_population: 最终种群
            generation: 最终代数
            format_type: 保存格式 (json, csv, excel)
            
        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备数据
        results_data = []
        for i, individual in enumerate(final_population):
            result = {
                "solution_id": i + 1,
                "rank": individual.rank,
                "disassembly_complexity": individual.disassembly_complexity,
                "ergonomics": individual.ergonomics,
                "crowding_distance": individual.distance,
                "disassembly_sequence": individual.disassembly_sequence.tolist(),
                "human_robot_tasking": individual.human_robot_tasking_sequence.tolist()
            }
            results_data.append(result)
        
        # 保存元数据
        metadata = {
            "algorithm": "NSGA-II",
            "problem": "Disassembly Sequence Optimization",
            "generation": generation,
            "population_size": len(final_population),
            "timestamp": timestamp,
            "objectives": ["disassembly_complexity", "ergonomics"]
        }
        
        if format_type.lower() == "json":
            return self._save_json_results(results_data, metadata, timestamp)
        elif format_type.lower() == "csv":
            return self._save_csv_results(results_data, metadata, timestamp)
        elif format_type.lower() == "excel":
            return self._save_excel_results(results_data, metadata, timestamp)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _save_json_results(self, results_data: List[Dict], metadata: Dict, 
                          timestamp: str) -> str:
        """保存JSON格式结果"""
        filename = f"nsga_results_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        output_data = {
            "metadata": metadata,
            "results": results_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_csv_results(self, results_data: List[Dict], metadata: Dict, 
                         timestamp: str) -> str:
        """保存CSV格式结果"""
        filename = f"nsga_results_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # 转换为DataFrame
        df = pd.DataFrame(results_data)
        
        # 保存CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        # 保存元数据
        metadata_file = f"metadata_{timestamp}.json"
        metadata_path = os.path.join(self.output_dir, metadata_file)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_excel_results(self, results_data: List[Dict], metadata: Dict, 
                           timestamp: str) -> str:
        """保存Excel格式结果"""
        filename = f"nsga_results_{timestamp}.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 保存结果数据
            df = pd.DataFrame(results_data)
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # 保存元数据
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        return filepath
    
    def save_evolution_history(self, history: List[Dict[str, Any]], 
                             timestamp: Optional[str] = None) -> str:
        """
        保存进化历史
        
        Args:
            history: 进化历史数据
            timestamp: 时间戳
            
        Returns:
            保存的文件路径
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"evolution_history_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def plot_pareto_front(self, population: List[Individual], 
                         generation: int, save_plot: bool = True) -> Optional[str]:
        """
        绘制帕累托前沿
        
        Args:
            population: 种群
            generation: 代数
            save_plot: 是否保存图片
            
        Returns:
            保存的图片路径（如果保存）
        """
        # 提取目标函数值
        complexity_values = [ind.disassembly_complexity for ind in population]
        ergonomics_values = [ind.ergonomics for ind in population]
        ranks = [ind.rank for ind in population]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 按层级着色
        unique_ranks = sorted(set(ranks))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_ranks)))
        
        for i, rank in enumerate(unique_ranks):
            rank_indices = [j for j, r in enumerate(ranks) if r == rank]
            rank_complexity = [complexity_values[j] for j in rank_indices]
            rank_ergonomics = [ergonomics_values[j] for j in rank_indices]
            
            plt.scatter(rank_complexity, rank_ergonomics, 
                       c=[colors[i]], label=f'Rank {rank}', alpha=0.7, s=50)
        
        plt.xlabel('拆卸复杂度 (Disassembly Complexity)')
        plt.ylabel('人体工程学评价 (Ergonomics)')
        plt.title(f'帕累托前沿 - 第{generation}代 (Pareto Front - Generation {generation})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pareto_front_gen_{generation}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def plot_convergence_curve(self, history: List[Dict[str, Any]], 
                             save_plot: bool = True) -> Optional[str]:
        """
        绘制收敛曲线
        
        Args:
            history: 进化历史
            save_plot: 是否保存图片
            
        Returns:
            保存的图片路径（如果保存）
        """
        generations = [h['generation'] for h in history]
        best_complexity = [h['best_complexity'] for h in history]
        best_ergonomics = [h['best_ergonomics'] for h in history]
        avg_complexity = [h['avg_complexity'] for h in history]
        avg_ergonomics = [h['avg_ergonomics'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 拆卸复杂度收敛曲线
        ax1.plot(generations, best_complexity, 'b-', label='最优值', linewidth=2)
        ax1.plot(generations, avg_complexity, 'r--', label='平均值', linewidth=1)
        ax1.set_xlabel('代数 (Generation)')
        ax1.set_ylabel('拆卸复杂度 (Disassembly Complexity)')
        ax1.set_title('拆卸复杂度收敛曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 人体工程学评价收敛曲线
        ax2.plot(generations, best_ergonomics, 'b-', label='最优值', linewidth=2)
        ax2.plot(generations, avg_ergonomics, 'r--', label='平均值', linewidth=1)
        ax2.set_xlabel('代数 (Generation)')
        ax2.set_ylabel('人体工程学评价 (Ergonomics)')
        ax2.set_title('人体工程学评价收敛曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"convergence_curve_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return None
    
    def print_generation_summary(self, generation: int, population: List[Individual]) -> None:
        """
        打印代数摘要信息
        
        Args:
            generation: 代数
            population: 当前种群
        """
        if not population:
            return
        
        complexity_values = [ind.disassembly_complexity for ind in population]
        ergonomics_values = [ind.ergonomics for ind in population]
        
        print(f"\n========== 第 {generation} 代摘要 ==========")
        print(f"种群大小: {len(population)}")
        print(f"拆卸复杂度 - 最优: {min(complexity_values):.4f}, "
              f"平均: {np.mean(complexity_values):.4f}, "
              f"最差: {max(complexity_values):.4f}")
        print(f"人体工程学 - 最优: {min(ergonomics_values):.4f}, "
              f"平均: {np.mean(ergonomics_values):.4f}, "
              f"最差: {max(ergonomics_values):.4f}")
        
        # 统计各层级个体数量
        ranks = [ind.rank for ind in population]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        print(f"层级分布: {dict(sorted(rank_counts.items()))}")
    
    def print_final_summary(self, final_population: List[Individual], 
                           total_generations: int, execution_time: float) -> None:
        """
        打印最终摘要
        
        Args:
            final_population: 最终种群
            total_generations: 总代数
            execution_time: 执行时间（秒）
        """
        print(f"\n{'='*50}")
        print(f"NSGA-II算法执行完成")
        print(f"{'='*50}")
        print(f"总代数: {total_generations}")
        print(f"执行时间: {execution_time:.2f} 秒")
        print(f"最终种群大小: {len(final_population)}")
        
        # 第一层级（帕累托最优解）统计
        pareto_optimal = [ind for ind in final_population if ind.rank == 1]
        print(f"帕累托最优解数量: {len(pareto_optimal)}")
        
        if pareto_optimal:
            complexity_values = [ind.disassembly_complexity for ind in pareto_optimal]
            ergonomics_values = [ind.ergonomics for ind in pareto_optimal]
            
            print(f"\n帕累托最优解统计:")
            print(f"拆卸复杂度范围: [{min(complexity_values):.4f}, {max(complexity_values):.4f}]")
            print(f"人体工程学评价范围: [{min(ergonomics_values):.4f}, {max(ergonomics_values):.4f}]")
        
        print(f"{'='*50}")
    
    def export_best_solutions(self, population: List[Individual], 
                            top_n: int = 5) -> str:
        """
        导出最优解详细信息
        
        Args:
            population: 种群
            top_n: 导出前N个最优解
            
        Returns:
            导出文件路径
        """
        # 选择第一层级的解
        pareto_optimal = [ind for ind in population if ind.rank == 1]
        
        if len(pareto_optimal) > top_n:
            # 按拥挤度距离排序，选择前top_n个
            pareto_optimal.sort(key=lambda x: x.distance, reverse=True)
            pareto_optimal = pareto_optimal[:top_n]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_solutions_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("NSGA-II最优解详细信息\n")
            f.write("="*50 + "\n\n")
            
            for i, individual in enumerate(pareto_optimal, 1):
                f.write(f"解 {i}:\n")
                f.write(f"  拆卸复杂度: {individual.disassembly_complexity:.6f}\n")
                f.write(f"  人体工程学评价: {individual.ergonomics:.6f}\n")
                f.write(f"  拥挤度距离: {individual.distance:.6f}\n")
                f.write(f"  拆卸序列: {individual.disassembly_sequence.tolist()}\n")
                f.write(f"  人机任务分配: {individual.human_robot_tasking_sequence.tolist()}\n")
                f.write(f"  (0=机器人, 1=操作工)\n")
                f.write("-" * 40 + "\n")
        
        return filepath
