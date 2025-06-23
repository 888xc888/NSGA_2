"""
ConfigManager.py
配置管理模块 - 负责加载和管理所有配置参数

Author: Refactored Code
Date: 2025-06-23
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class NSGAConfig:
    """NSGA-II算法配置参数"""
    population_size: int
    termination_condition: int
    crossover_probability: float
    mutation_probability: float


@dataclass
class SystemConfig:
    """系统配置参数"""
    save_frequency: int = 10
    enable_plotting: bool = True
    output_directory: str = "Output"


class ConfigManager:
    """
    配置管理器 - 负责加载和管理所有配置文件
    """
    
    def __init__(self, config_dir: str = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为当前目录
        """
        self.config_dir = config_dir or os.getcwd()
        self._nsga_config: NSGAConfig = None
        self._system_config: SystemConfig = None
        
    def load_nsga_config(self, config_path: str = None) -> NSGAConfig:
        """
        加载NSGA-II算法配置参数
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            NSGA配置对象
        """
        if config_path is None:
            config_path = os.path.join(self.config_dir, 'Config', 'AlgorithmConfig.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        nsga_data = data['NSGA_config']
        self._nsga_config = NSGAConfig(
            population_size=nsga_data['population_size'],
            termination_condition=nsga_data['termination_condition'],
            crossover_probability=nsga_data['crossover_probability'],
            mutation_probability=nsga_data['mutation_probability']
        )
        
        return self._nsga_config
    
    def load_system_config(self, config_path: str = None) -> SystemConfig:
        """
        加载系统配置参数
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            系统配置对象
        """
        if config_path is None:
            config_path = os.path.join(self.config_dir, 'Config', 'SystemConfig.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            system_data = data.get('system_config', {})
            self._system_config = SystemConfig(
                save_frequency=system_data.get('save_frequency', 10),
                enable_plotting=system_data.get('enable_plotting', True),
                output_directory=system_data.get('output_directory', 'Output')
            )
        except FileNotFoundError:
            # 如果配置文件不存在，使用默认配置
            self._system_config = SystemConfig()
        
        return self._system_config
    
    def load_prioritization_matrix(self, description_path: str = None) -> np.ndarray:
        """
        加载优先级矩阵
        
        Args:
            description_path: 描述文件路径
            
        Returns:
            优先级矩阵的numpy数组
        """
        if description_path is None:
            description_path = os.path.join(self.config_dir, 'Data', 'ProblemDescription.yaml')
        
        with open(description_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        priority_matrix = data['prioritization_matrix']
        return np.array(priority_matrix['data'])
    
    def load_recognition_results(self, description_path: str = None) -> np.ndarray:
        """
        加载识别结果向量
        
        Args:
            description_path: 描述文件路径
            
        Returns:
            识别结果的numpy数组
        """
        if description_path is None:
            description_path = os.path.join(self.config_dir, 'Data', 'ProblemDescription.yaml')
        
        with open(description_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        vector = data['recognition_results_vector']
        return np.array(vector['data'])
    
    def load_disassembly_complexity_indicators(self, indicator_path: str = None) -> Dict[str, Any]:
        """
        加载拆卸复杂度评价指标
        
        Args:
            indicator_path: 指标文件路径
            
        Returns:
            拆卸复杂度指标字典
        """
        if indicator_path is None:
            indicator_path = os.path.join(self.config_dir, 'Data', 'EvaluationIndicators.yaml')
        
        with open(indicator_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        return data['disassembly_complexity']
    
    def load_ergonomics_indicators(self, indicator_path: str = None) -> Dict[str, Any]:
        """
        加载人体工程学评价指标
        
        Args:
            indicator_path: 指标文件路径
            
        Returns:
            人体工程学指标字典
        """
        if indicator_path is None:
            indicator_path = os.path.join(self.config_dir, 'Data', 'EvaluationIndicators.yaml')
        
        with open(indicator_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        return data['ergonomics']
    
    @property
    def nsga_config(self) -> NSGAConfig:
        """获取NSGA配置"""
        if self._nsga_config is None:
            self.load_nsga_config()
        return self._nsga_config
    
    @property
    def system_config(self) -> SystemConfig:
        """获取系统配置"""
        if self._system_config is None:
            self.load_system_config()
        return self._system_config
