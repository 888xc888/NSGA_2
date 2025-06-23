# NSGA-II拆卸序列优化系统

基于NSGA-II算法的多目标拆卸序列优化解决方案，具备完整的结果管理和可视化功能。

## 🌟 项目概述

本项目是一个完全重构的NSGA-II（Non-dominated Sorting Genetic Algorithm II）实现，专门用于解决拆卸序列优化问题。系统采用模块化设计，集成了智能结果管理系统，提供了优秀的用户体验和强大的分析功能。

## ✨ 主要特性

### 🎯 核心算法功能
- **多目标优化**: 同时优化拆卸复杂度和人体工程学评价
- **约束处理**: 支持零件拆卸优先级约束和人机任务分配
- **智能初始化**: 多样化的种群初始化策略
- **高效遗传算子**: 针对拆卸序列优化的专用交叉变异算子

### 📊 结果管理与可视化
- **自动结果保存**: 每次运行自动保存到编号文件夹（000Run, 001Run...）
- **丰富的可视化**: 帕累托前沿图、收敛曲线、进化历史
- **多格式输出**: JSON、TXT、PNG等多种格式
- **运行记录追踪**: 完整的运行历史和统计信息

### 🛠️ 用户友好功能
- **交互式工具**: 查看运行记录和清理测试数据
- **详细文档**: 完整的使用说明和API文档
- **模块化设计**: 清晰的代码结构，易于扩展和维护

## 📁 项目结构

```
NSGA/
├── 📂 Config/                    # 配置管理模块
│   ├── ConfigManager.py         # 配置管理器
│   ├── AlgorithmConfig.yaml     # 算法参数配置
│   └── SystemConfig.yaml        # 系统配置
├── 📂 Core/                      # 核心算法模块
│   ├── NSGACore.py              # NSGA-II核心算法
│   ├── GeneticOperators.py      # 遗传算子实现
│   └── PopulationInit.py        # 种群初始化策略
├── 📂 Data/                      # 数据与评价指标
│   ├── ProblemDescription.yaml  # 问题描述
│   └── EvaluationIndicators.yaml # 评价指标数据
├── 📂 Models/                    # 数据模型
│   └── Individual.py            # 个体类定义
├── 📂 Utils/                     # 工具模块
│   ├── EvaluationUtils.py       # 评价计算工具
│   ├── ValidationUtils.py       # 验证工具
│   └── ResultsManager.py        # 🆕 结果管理器
├── 📂 Output/                    # 输出管理
│   ├── OutputManager.py         # 输出管理器
│   └── 📂 Results/               # 🆕 结果存储目录
│       ├── 000Run/              # 第一次运行结果
│       ├── 001Run/              # 第二次运行结果
│       └── ...                  # 更多运行结果
├── 🚀 RunOptimization.py        # 主程序入口
├── 🔧 NSGAOptimizer.py          # NSGA-II优化器
├── 👀 view_runs.py              # 查看运行记录工具
├── 🗑️ clear_test_data.py        # 清理测试数据工具
└── 📖 Results_Management_Guide.md # 结果管理使用说明
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install numpy pandas matplotlib pyyaml openpyxl
```

### 2. 运行优化
```bash
python RunOptimization.py
```

### 3. 查看结果
```bash
python view_runs.py
```

### 4. 清理测试数据（可选）
```bash
python clear_test_data.py
```

## 📖 使用方法

### 🎯 基本使用

**运行优化算法：**
```bash
python RunOptimization.py
```
系统会自动：
- 运行NSGA-II多目标优化
- 创建新的编号文件夹（如002Run）
- 保存所有结果文件和图表
- 显示最优解摘要

**查看历史运行：**
```bash
python view_runs.py
```
显示所有历史运行记录和详细信息。

### 🔧 高级使用

**编程接口：**
```python
from Config.ConfigManager import ConfigManager
from NSGAOptimizer import NSGAOptimizer
from Utils.ResultsManager import ResultsManager

# 初始化系统
config_manager = ConfigManager()
optimizer = NSGAOptimizer(config_manager)

# 运行优化
pareto_solutions = optimizer.optimize(verbose=True)

# 保存结果
results_manager = ResultsManager()
results_manager.save_current_run("自定义描述")
```

**结果管理：**
```python
from Utils.ResultsManager import ResultsManager

manager = ResultsManager()

# 查看运行记录
manager.print_runs_summary()

# 清理临时文件
manager.clear_test_data(confirm=True)
```

## ⚙️ 配置说明

### 算法配置 (Config/AlgorithmConfig.yaml)

```yaml
NSGA_config:
  population_size: 20        # 种群大小
  termination_condition: 250 # 最大进化代数
  crossover_probability: 0.5 # 交叉概率
  mutation_probability: 0.01 # 变异概率
```

### 系统配置 (Config/SystemConfig.yaml)
```yaml
system_config:
  output_directory: "Output"  # 输出目录
  save_plots: true           # 是否保存图表
  plot_format: "png"         # 图表格式
  verbose: true              # 详细输出
```

## 📊 输出结果说明

### 自动保存的文件
每次运行会自动保存到编号文件夹中：

```
Output/Results/00XRun/
├── 📄 run_info.json                    # 运行信息记录
├── 📄 best_solutions_*.txt             # 最优解详细信息
├── 📄 nsga_results_*.json              # 完整结果数据
├── 📄 evolution_history_*.json         # 进化历史数据
├── 📈 convergence_curve_*.png          # 收敛曲线图
├── 📈 pareto_front_gen_50_*.png        # 第50代帕累托前沿
├── 📈 pareto_front_gen_100_*.png       # 第100代帕累托前沿
├── 📈 pareto_front_gen_150_*.png       # 第150代帕累托前沿
├── 📈 pareto_front_gen_200_*.png       # 第200代帕累托前沿
└── 📈 pareto_front_gen_250_*.png       # 最终帕累托前沿
```

### 结果文件格式

**最优解文件 (best_solutions_*.txt):**
```
NSGA-II最优解详细信息
==================================================

解 1:
  拆卸复杂度: 1079.000000
  人体工程学评价: 0.000000
  拥挤度距离: inf
  拆卸序列: [2, 7, 4, 3, 5, 9, 10, 8, 6, 1]
  人机任务分配: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  (0=机器人, 1=操作工)
```

**运行信息文件 (run_info.json):**
```json
{
  "run_number": 0,
  "folder_name": "000Run",
  "creation_time": "2025-06-23T23:14:20.110241",
  "description": "NSGA-II优化 - 20个帕累托解 - 4.2秒",
  "moved_files": ["file1.txt", "file2.png", ...],
  "file_count": 10
}
```

## 🛠️ 工具和脚本

### 查看运行记录
```bash
python view_runs.py
```
显示所有历史运行的摘要信息，包括运行时间、文件数量、描述等。

### 清理测试数据
```bash
python clear_test_data.py
```
交互式清理Output目录下的临时文件，不会影响已保存的运行结果。

### 结果管理API
```python
from Utils.ResultsManager import ResultsManager

# 创建管理器
manager = ResultsManager()

# 保存当前运行
manager.save_current_run("运行描述")

# 查看运行列表
runs = manager.list_runs()
manager.print_runs_summary()

# 清理测试数据
manager.clear_test_data(confirm=True)
```

## 🧬 算法特点

### NSGA-II核心功能
- **非支配排序**: 高效的帕累托排序算法
- **拥挤度距离**: 保持解的多样性
- **精英选择**: 确保优秀个体的传承
- **约束处理**: 支持拆卸优先级约束

### 专用遗传算子
- **序列交叉**: 针对拆卸序列优化的PMX交叉
- **智能变异**: 保持约束的变异操作
- **多样化初始化**: 确保初始种群的多样性

### 多目标优化
- **目标1**: 拆卸复杂度最小化
- **目标2**: 人体工程学评价最小化
- **约束**: 拆卸优先级和人机任务分配

## 📈 性能与可扩展性

### 计算性能
- 支持大规模零件数量（测试过100+零件）
- 高效的约束检查和评价计算
- 优化的内存使用

### 可扩展性
- 模块化设计，易于添加新的目标函数
- 支持自定义遗传算子
- 灵活的配置系统

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 [Issue](../../issues)
- 发送邮件至项目维护者

## 🙏 致谢

感谢所有为本项目做出贡献的研究者和开发者。

---

⭐ 如果这个项目对您有帮助，请给它一个星标！

```yaml
system_config:
  save_frequency: 10         # 保存频率
  enable_plotting: true      # 是否启用绘图
  output_directory: "Output" # 输出目录
  log_level: "INFO"         # 日志级别
```

### 问题描述 (Data/ProblemDescription.yaml)

- 优先级矩阵：定义零件拆卸的约束关系
- 识别结果：描述零件的状态（正常、滑丝、生锈等）

### 评价指标 (Data/EvaluationIndicators.yaml)

- 拆卸复杂度：基于三角模糊数的复杂度评价
- 人体工程学：操作工拆卸时的工程学评价

## 算法特点

### NSGA-II核心功能

1. **快速非支配排序**: 高效的帕累托层级划分
2. **拥挤度距离**: 保持解的多样性
3. **精英选择**: 保留优秀个体
4. **锦标赛选择**: 平衡选择压力和多样性

### 遗传算子

1. **顺序交叉(OX)**: 适用于排列问题的交叉算子
2. **多点交叉**: 适用于二进制编码的交叉
3. **交换变异**: 保持排列有效性的变异算子
4. **位翻转变异**: 二进制编码的变异算子

### 约束处理

- **优先级约束验证**: 确保拆卸序列满足优先级关系
- **修复策略**: 自动修复无效的个体
- **启发式初始化**: 基于问题特征的智能初始化

## 输出结果

### 文件输出

- `nsga_results_*.json`: 最终优化结果
- `evolution_history_*.json`: 进化历史数据
- `best_solutions_*.txt`: 最优解详细信息

### 图形输出

- `pareto_front_*.png`: 帕累托前沿图
- `convergence_curve_*.png`: 收敛曲线图

### 统计分析

- 目标函数统计信息
- 任务分配比例分析
- 解的多样性分析

## 扩展指南

### 添加新的目标函数

1. 在`Individual.py`中添加新的目标函数属性
2. 在`EvaluationUtils.py`中实现计算逻辑
3. 在`NSGACore.py`中更新支配关系比较
4. 在`OutputManager.py`中更新输出格式

### 添加新的约束

1. 在`ValidationUtils.py`中添加约束验证函数
2. 在遗传算子中集成约束检查
3. 在种群初始化中考虑约束

### 自定义遗传算子

1. 在`GeneticOperators.py`中实现新算子
2. 在`NSGAOptimizer.py`中集成新算子
3. 在配置文件中添加相关参数

## 性能优化建议

1. **并行化**: 可以并行评价个体适应度
2. **缓存**: 缓存重复计算的结果
3. **早停**: 基于收敛判据提前终止
4. **自适应参数**: 动态调整遗传算子参数

## 注意事项

1. 确保优先级矩阵的一致性
2. 合理设置种群大小和进化代数
3. 根据问题规模调整遗传算子参数
4. 定期备份重要的配置和结果文件

## 版本信息

- **版本**: 2.0.0 (重构版本)
---

⭐ 如果这个项目对您有帮助，请给它一个星标！