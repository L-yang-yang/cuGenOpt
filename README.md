# GenSolver - 通用CUDA启发式求解器

> **一个问题无关的GPU加速组合优化求解框架**

## 项目定位

GenSolver 是一个基于CUDA的通用启发式求解框架。它不感知具体问题类型（TSP、VRP、指派、调度等），而是提供：

- **通用的解编码结构**：二维数组表示
- **通用的搜索算子**：swap、reverse、insert、relocate等
- **灵活的多目标比较**：加权和、字典序、可行性优先等
- **高效的GPU并行调度**：种群级 + 邻域级并行

用户只需定义：
1. **问题数据**（距离矩阵、约束等）
2. **评估函数**（数组 → 目标值）

框架负责所有GPU并行化、种群管理、搜索调度。

## 目录结构

```
generic_solver/
├── README.md                    # 本文件
├── design/                      # 设计文档
│   ├── 00_overview.md          # 总体设计概览
│   ├── 01_solution_encoding.md # 解编码设计
│   ├── 02_objective_system.md  # 目标函数系统设计
│   ├── 03_comparison.md        # 解比较机制设计
│   ├── 04_operators.md         # 搜索算子设计
│   ├── 05_memory_layout.md     # GPU内存层次设计
│   ├── 06_population.md        # 种群管理设计
│   ├── 07_parallel_strategy.md # 并行化策略设计
│   ├── 08_user_interface.md    # 用户接口设计
│   ├── 09_roadmap.md           # 开发路线图
│   ├── 10_cpu_gpu_boundary.md  # CPU/GPU职责边界设计
│   └── 11_evaluation_abstraction.md # 评估函数抽象（用户不需要CUDA）
├── src/                         # 源代码（后续）
├── include/                     # 公共头文件（后续）
├── examples/                    # 使用示例（后续）
│   ├── tsp/                    # TSP问题示例
│   ├── vrp/                    # VRP问题示例
│   └── assignment/             # 指派问题示例
└── tests/                       # 测试（后续）
```

## 核心理念

```
框架看到的：一堆数字数组 + 一组目标值
用户看到的：TSP路径 / VRP路线 / 排班方案 / ...
```

## 设计文档索引

- [00 总体设计概览](design/00_overview.md) ← **从这里开始**
- [01 解编码设计](design/01_solution_encoding.md)
- [02 目标函数系统](design/02_objective_system.md)
- [03 解比较机制](design/03_comparison.md)
- [04 搜索算子](design/04_operators.md)
- [05 GPU内存层次](design/05_memory_layout.md)
- [06 种群管理](design/06_population.md)
- [07 并行化策略](design/07_parallel_strategy.md)
- [08 用户接口](design/08_user_interface.md)
- [09 开发路线图](design/09_roadmap.md)
- [10 CPU/GPU职责边界](design/10_cpu_gpu_boundary.md)
- [11 评估函数抽象](design/11_evaluation_abstraction.md) -- 用户不需要写CUDA

## 状态

- [x] 设计文档
- [x] 核心框架实现 (prototype/)
- [x] 12 种问题示例 (TSP/VRP/Knapsack/QAP/JSP/...)
- [x] 完整 benchmark 对比 (GPU vs OR-Tools)
- [x] 自适应种群 + 自适应岛屿模型
- [ ] **大规模实例优化** (N≥150 质量问题待解决)

## 最新进展 (2026-02-22)

### ✅ 已完成
1. **自适应岛屿模型**: `num_islands=0` 根据种群规模自动决定岛屿数
2. **完整 benchmark**: 
   - GPU (T4): 670 条数据，6 个实验组
   - OR-Tools: 170 条对比数据
3. **通用性验证**: 12 种问题类型全部测试通过


### 🔧 待优化
1. 诊断大规模实例的实际配置 (pop_size, 算子使用)
2. 优化初始解生成策略 (启发式构造)
3. 调整 LNS 参数适配大规模实例
4. 考虑实现问题特定的高效算子

详见: `design/21_features_and_experiments_summary.md` 和 `prototype/benchmark/results/final_20260222/`
