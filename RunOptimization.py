"""
RunOptimization.py
NSGA-II拆卸序列优化系统主运行程序

运行此文件开始NSGA-II多目标优化
"""

import os
import sys
import time

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Config.ConfigManager import ConfigManager
from NSGAOptimizer import NSGAOptimizer
from Utils.ResultsManager import ResultsManager


def main():
    """主运行函数"""
    print("=" * 60)
    print("   NSGA-II拆卸序列优化系统")
    print("   Multi-Objective Disassembly Sequence Optimization")
    print("=" * 60)
    
    try:
        # 初始化配置管理器
        print("正在初始化系统...")
        config_manager = ConfigManager(current_dir)
        
        # 加载配置信息并显示
        nsga_config = config_manager.load_nsga_config()
        system_config = config_manager.load_system_config()
        recognition_results = config_manager.load_recognition_results()
        
        print("\n系统配置信息:")
        print("-" * 30)
        print(f"零件数量: {len(recognition_results)}")
        print(f"种群大小: {nsga_config.population_size}")
        print(f"最大进化代数: {nsga_config.termination_condition}")
        print(f"交叉概率: {nsga_config.crossover_probability}")
        print(f"变异概率: {nsga_config.mutation_probability}")
        print(f"输出目录: {system_config.output_directory}")
        
        # 初始化NSGA-II优化器
        print("\n正在初始化NSGA-II优化器...")
        optimizer = NSGAOptimizer(config_manager)
        
        print("系统初始化完成！")
        
        # 开始优化
        print("\n开始运行NSGA-II优化...")
        start_time = time.time()
        
        pareto_solutions = optimizer.optimize(verbose=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 显示结果
        print("\n" + "=" * 50)
        print("优化完成！")
        print("=" * 50)
        print(f"执行时间: {execution_time:.2f} 秒")
        print(f"帕累托最优解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            print("\n前3个最优解:")
            print("-" * 40)
            
            # 按拥挤度距离排序
            sorted_solutions = sorted(pareto_solutions, 
                                     key=lambda x: x.distance, reverse=True)
            
            for i, solution in enumerate(sorted_solutions[:3], 1):
                print(f"解 {i}:")
                print(f"  拆卸复杂度: {solution.disassembly_complexity:.6f}")
                print(f"  人体工程学评价: {solution.ergonomics:.6f}")
                print(f"  拆卸序列: {solution.disassembly_sequence.tolist()}")
                
                # 统计人机任务分配
                human_tasks = sum(solution.human_robot_tasking_sequence)
                robot_tasks = len(solution.human_robot_tasking_sequence) - human_tasks
                print(f"  任务分配: 操作工({human_tasks}), 机器人({robot_tasks})")
                print()
        
        # 初始化结果管理器并保存本次运行结果
        print("\n正在保存运行结果...")
        results_manager = ResultsManager()
        
        # 生成运行描述
        run_description = f"NSGA-II优化 - {len(pareto_solutions)}个帕累托解 - {execution_time:.1f}秒"
        
        # 保存结果到编号文件夹
        results_manager.save_current_run(description=run_description)
        
        print("优化完成！")
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n按Enter键退出...")
    input()


if __name__ == "__main__":
    main()
