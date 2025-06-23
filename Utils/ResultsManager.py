"""
ResultsManager.py
结果管理器 - 专门用于管理和保存NSGA-II优化结果

功能：
1. 自动创建编号文件夹（000Run, 001Run, 002Run...）
2. 将所有结果文件保存到对应的运行文件夹中
3. 提供清理功能删除测试数据
"""

import os
import shutil
import glob
from datetime import datetime
from typing import Optional, List
import json


class ResultsManager:
    """结果管理器类"""
    
    def __init__(self, base_output_dir: str = "Output"):
        """
        初始化结果管理器
        
        Args:
            base_output_dir: 基础输出目录
        """
        self.base_output_dir = base_output_dir
        self.results_dir = os.path.join(base_output_dir, "Results")
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_next_run_number(self) -> int:
        """
        获取下一个运行编号
        
        Returns:
            下一个运行编号
        """
        # 查找所有现有的运行文件夹
        existing_runs = glob.glob(os.path.join(self.results_dir, "*Run"))
        
        if not existing_runs:
            return 0
        
        # 提取编号并找到最大值
        run_numbers = []
        for run_path in existing_runs:
            folder_name = os.path.basename(run_path)
            if folder_name.endswith("Run"):
                try:
                    number = int(folder_name[:-3])  # 移除"Run"后缀
                    run_numbers.append(number)
                except ValueError:
                    continue
        
        return max(run_numbers) + 1 if run_numbers else 0
    
    def create_new_run_folder(self) -> str:
        """
        创建新的运行文件夹
        
        Returns:
            新创建的运行文件夹路径
        """
        run_number = self.get_next_run_number()
        folder_name = f"{run_number:03d}Run"  # 格式：000Run, 001Run, 002Run...
        run_folder_path = os.path.join(self.results_dir, folder_name)
        
        os.makedirs(run_folder_path, exist_ok=True)
        
        # 创建运行信息文件
        run_info = {
            "run_number": run_number,
            "folder_name": folder_name,
            "creation_time": datetime.now().isoformat(),
            "description": f"NSGA-II优化运行 #{run_number:03d}"
        }
        
        info_file_path = os.path.join(run_folder_path, "run_info.json")
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)
        
        return run_folder_path
    
    def move_files_to_run_folder(self, run_folder_path: str, file_patterns: Optional[List[str]] = None) -> List[str]:
        """
        将文件移动到运行文件夹
        
        Args:
            run_folder_path: 运行文件夹路径
            file_patterns: 要移动的文件模式列表，如果为None则移动所有结果文件
            
        Returns:
            移动的文件列表
        """
        if file_patterns is None:
            # 默认的文件模式
            file_patterns = [
                "*.txt",
                "*.png", 
                "*.json",
                "*.csv"
            ]
        
        moved_files = []
        
        for pattern in file_patterns:
            # 在基础输出目录中查找匹配的文件
            files = glob.glob(os.path.join(self.base_output_dir, pattern))
            
            for file_path in files:
                # 跳过子目录中的文件和特殊文件
                if os.path.dirname(file_path) != self.base_output_dir:
                    continue
                if os.path.basename(file_path) in ["__init__.py", "OutputManager.py"]:
                    continue
                
                # 移动文件到运行文件夹
                file_name = os.path.basename(file_path)
                destination = os.path.join(run_folder_path, file_name)
                
                try:
                    shutil.move(file_path, destination)
                    moved_files.append(file_name)
                    print(f"移动文件: {file_name} -> {os.path.basename(run_folder_path)}")
                except Exception as e:
                    print(f"移动文件失败 {file_name}: {e}")
        
        return moved_files
    
    def save_current_run(self, description: Optional[str] = None) -> str:
        """
        保存当前运行的结果
        
        Args:
            description: 运行描述
            
        Returns:
            创建的运行文件夹路径
        """
        # 创建新的运行文件夹
        run_folder_path = self.create_new_run_folder()
        
        # 移动结果文件
        moved_files = self.move_files_to_run_folder(run_folder_path)
        
        # 更新运行信息
        info_file_path = os.path.join(run_folder_path, "run_info.json")
        with open(info_file_path, 'r', encoding='utf-8') as f:
            run_info = json.load(f)
        
        run_info["moved_files"] = moved_files
        run_info["file_count"] = len(moved_files)
        if description:
            run_info["description"] = description
        
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)
        
        folder_name = os.path.basename(run_folder_path)
        print(f"\n✅ 运行结果已保存到: {folder_name}")
        print(f"📁 文件夹路径: {run_folder_path}")
        print(f"📄 保存文件数量: {len(moved_files)}")
        
        return run_folder_path
    
    def clear_test_data(self, confirm: bool = False) -> None:
        """
        清理测试数据
        
        Args:
            confirm: 是否确认删除
        """
        if not confirm:
            print("⚠️  警告：此操作将删除Output目录中的所有测试文件！")
            print("如要确认删除，请调用 clear_test_data(confirm=True)")
            return
        
        # 要删除的文件模式
        file_patterns = [
            "*.txt",
            "*.png", 
            "*.json",
            "*.csv"
        ]
        
        deleted_files = []
        
        for pattern in file_patterns:
            files = glob.glob(os.path.join(self.base_output_dir, pattern))
            
            for file_path in files:
                # 跳过子目录中的文件和特殊文件
                if os.path.dirname(file_path) != self.base_output_dir:
                    continue
                if os.path.basename(file_path) in ["__init__.py", "OutputManager.py"]:
                    continue
                
                try:
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
                    print(f"删除文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"删除文件失败 {os.path.basename(file_path)}: {e}")
        
        print(f"\n🗑️  清理完成，共删除 {len(deleted_files)} 个文件")
    
    def list_runs(self) -> List[dict]:
        """
        列出所有运行记录
        
        Returns:
            运行记录列表
        """
        runs = []
        run_folders = glob.glob(os.path.join(self.results_dir, "*Run"))
        
        for run_folder in sorted(run_folders):
            info_file = os.path.join(run_folder, "run_info.json")
            
            if os.path.exists(info_file):
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        run_info = json.load(f)
                    runs.append(run_info)
                except Exception as e:
                    print(f"读取运行信息失败 {os.path.basename(run_folder)}: {e}")
        
        return runs
    
    def print_runs_summary(self) -> None:
        """打印运行摘要"""
        runs = self.list_runs()
        
        if not runs:
            print("📭 暂无保存的运行记录")
            return
        
        print(f"\n📊 运行记录摘要 (共{len(runs)}个运行)")
        print("=" * 80)
        print(f"{'编号':<6} {'文件夹名':<10} {'创建时间':<20} {'文件数':<6} {'描述'}")
        print("-" * 80)
        
        for run in runs:
            run_number = run.get('run_number', 'N/A')
            folder_name = run.get('folder_name', 'N/A')
            creation_time = run.get('creation_time', 'N/A')[:19]  # 只显示日期时间部分
            file_count = run.get('file_count', 0)
            description = run.get('description', 'N/A')
            
            print(f"{run_number:<6} {folder_name:<10} {creation_time:<20} {file_count:<6} {description}")


# 便捷函数
def create_results_manager(base_output_dir: str = "Output") -> ResultsManager:
    """
    创建结果管理器实例
    
    Args:
        base_output_dir: 基础输出目录
        
    Returns:
        ResultsManager实例
    """
    return ResultsManager(base_output_dir)


def save_run_results(description: Optional[str] = None, base_output_dir: str = "Output") -> str:
    """
    快速保存运行结果
    
    Args:
        description: 运行描述
        base_output_dir: 基础输出目录
        
    Returns:
        创建的运行文件夹路径
    """
    manager = ResultsManager(base_output_dir)
    return manager.save_current_run(description)


def clear_all_test_data(base_output_dir: str = "Output") -> None:
    """
    清理所有测试数据
    
    Args:
        base_output_dir: 基础输出目录
    """
    manager = ResultsManager(base_output_dir)
    manager.clear_test_data(confirm=True)


if __name__ == "__main__":
    # 测试代码
    manager = ResultsManager()
    manager.print_runs_summary()
