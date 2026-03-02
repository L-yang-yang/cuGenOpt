# Benchmark 汇总分析报告

**日期**: 2026-02-22  
**GPU**: Tesla T4 (40 SM, 16GB VRAM)  
**配置**: 自适应种群 + 自适应岛屿 + SA + AOS + CX 10%

---

## 一、数据概览

### 1.1 GPU Benchmark (GenSolver)
- **文件**: `gensolver_all_20260222_175333.csv`
- **数据量**: 670 条（含 header = 671 行）
- **运行时间**: ~110 分钟
- **覆盖实验**:
  - Exp1: 通用性验证 (12 问题类型)
  - Exp2: TSP 质量 (6 实例 × 5 时间预算)
  - Exp3: VRP 质量 (1 实例 × 4 时间预算)
  - Exp4: 可扩展性 (6 实例 × 3 时间预算)
  - Exp5: 消融实验
  - Exp6: 时间-质量曲线

### 1.2 OR-Tools Benchmark (CPU Baseline)
- **文件**: `ortools_20260222_182532.csv`
- **数据量**: 170 条（含 header = 171 行）
- **运行时间**: ~57 分钟
- **覆盖实验**:
  - TSP: 6 实例 × 5 时间预算 (1s, 5s, 10s, 30s, 60s)
  - VRP: 1 实例 × 4 时间预算 (1s, 5s, 10s, 30s)

---

## 二、核心发现

### 2.1 小规模实例 (N ≤ 100): GPU 表现优秀

| 实例 | 规模 | 1s 预算 | 5s 预算 | 30s 预算 | 结论 |
|------|------|---------|---------|----------|------|
| **eil51** | N=51 | GPU **0.18%** vs OR 1.50% | GPU 0.05% vs OR 0.00% | GPU 0.05% vs OR 0.00% | 1s 预算下 GPU 更优 |
| **kroA100** | N=100 | GPU **0.16%** vs OR 0.46% | GPU 0.12% vs OR 0.00% | GPU 0.03% vs OR 0.00% | 1s 预算下 GPU 更优 |
| **A-n32-k5** (VRP) | N=31 | GPU **0.00%** vs OR 0.31% | GPU 0.00% vs OR 0.00% | GPU 0.00% vs OR 0.00% | GPU 持平或更优 |

**结论**: 在小规模实例上，GPU 在短时间预算（1s）下表现更好，长时间预算下与 OR-Tools 持平。

---

### 2.2 大规模实例 (N ≥ 150): **GPU 质量显著落后**

| 实例 | 规模 | 1s 预算 | 5s 预算 | 30s 预算 | 60s 预算 |
|------|------|---------|---------|----------|----------|
| **ch150** | N=150 | GPU 7.48% vs OR **2.27%** | GPU 2.33% vs OR **1.29%** | GPU 2.30% vs OR **0.54%** | GPU 2.15% vs OR **0.54%** |
| **tsp225** | N=225 | GPU 67.94% vs OR **1.92%** | GPU 3.33% vs OR **1.35%** | GPU 3.03% vs OR **-0.61%** | GPU 3.00% vs OR **-0.61%** |
| **lin318** | N=318 | GPU 163.07% vs OR **4.04%** | GPU 16.31% vs OR **3.62%** | GPU 5.57% vs OR **2.87%** | GPU 5.57% vs OR **2.85%** |
| **pcb442** | N=442 | GPU 324.78% vs OR **6.36%** | GPU 95.26% vs OR **2.24%** | GPU 6.92% vs OR **1.87%** | GPU 5.89% vs OR **1.66%** |

**关键问题**:
1. **1s 预算下 GPU 质量崩溃**: N≥225 时，GPU 的 gap 达到 67%-324%，完全不可用
2. **5s 预算下仍有巨大差距**: `pcb442` 的 gap 高达 95%，`lin318` 为 16%
3. **30s/60s 预算下差距收窄但仍落后**: GPU 需要 30-60s 才能达到 OR-Tools 1s 的质量

---

## 三、根因分析

### 3.1 可能的原因

#### 原因 1: 初始解质量差
- **现象**: 1s 预算下 gap 极高，说明初始解可能很差，需要很长时间才能收敛
- **验证**: 检查 `init_oversample` 是否生效，初始解是否足够多样化

#### 原因 2: 算子效率低
- **现象**: 大规模实例下，每代的改进幅度可能不如 OR-Tools 的 GLS
- **验证**: 对比每秒评估次数 (generations/time) 和每代平均改进

#### 原因 3: 种群规模不足
- **现象**: 自适应种群可能在大规模实例上分配的种群太小
- **验证**: 检查 `pcb442` 等实例的实际 `pop_size`

#### 原因 4: LNS 参数不当
- **现象**: LNS 的 destroy 比例或 per-op cap 可能不适合大规模实例
- **验证**: 对比有/无 LNS 的结果

---

## 四、实验数据统计

### 4.1 通用性验证 (Exp1)

| 问题类型 | 实例 | 最优解 | GPU avg gap | 达到最优率 |
|---------|------|--------|-------------|-----------|
| TSP | TSP5 | 19 | 0.00% | 5/5 |
| Knapsack | Knapsack100 | 1514 | 0.00% | 5/5 |
| Assignment | Assign4 | 13 | -7.14% | 5/5 |
| Schedule | Schedule3x4 | 21 | 0.00% | 5/5 |
| VRP | CVRP10 | 200 | 0.00% | 5/5 |
| Load Balance | LoadBal8 | 14 | 0.00% | 5/5 |
| Graph Coloring | GraphColor10 | 0 | 0.00% | 5/5 |
| Bin Packing | BinPack8 | 4 | 0.00% | 5/5 |
| QAP | QAP5 | 50 | 16.00% | 0/5 |
| VRPTW | VRPTW8 | 162 | 0.00% | 5/5 |
| JSP (Int) | JSP3x3_Int | 12 | 0.00% | 5/5 |
| JSP (Perm) | JSP3x3_Perm | 12 | 0.00% | 5/5 |

**结论**: 在小规模实例上，求解器对 12 种问题类型都能找到最优解（除 QAP5 外）。

---

## 五、关键结论与建议

### 5.1 当前状态评估

**优势**:
- ✓ 通用性强：12 种问题类型，零建模工作量
- ✓ 小规模实例质量好：N≤100 时与 OR-Tools 持平或更优
- ✓ 短时间预算优势：1s 内在小规模上超越 OR-Tools

**劣势**:
- ✗ **大规模实例质量严重不足**: N≥150 时，需要 30-60s 才能达到 OR-Tools 1s 的质量
- ✗ **初始解收敛慢**: 1s 预算下大规模实例完全不可用
- ✗ **扩展性问题**: 随着 N 增大，质量下降非常明显

### 5.2 紧急待办

**优先级 P0** (阻碍论文发表):
1. **诊断大规模实例质量问题**:
   - 检查 `pcb442` 等实例的实际 `pop_size`、`num_islands`、每代评估次数
   - 对比 verbose 输出，看是否存在配置异常
   
2. **优化初始解生成**:
   - 当前 `init_oversample=4` 可能不够
   - 考虑增加到 8 或 16，或使用启发式初始解（如贪心构造）

3. **调优 LNS 参数**:
   - 大规模实例可能需要更大的 destroy 比例
   - 检查 per-op cap 是否限制了 LNS 的探索能力

**优先级 P1** (增强说服力):
4. **增加 VRP 对比实例**: 当前只有 1 个 VRP 实例，数据不够充分
5. **补充消融实验**: 验证各模块（AOS、岛屿、LNS）对大规模实例的贡献

---

## 六、论文叙事

基于当前数据，**不建议**直接声称"GPU 比 CPU 快"或"质量更好"。

**推荐叙事角度**:

1. **通用性优势**:
   > "Our framework supports 12 problem types with zero modeling effort, while OR-Tools requires problem-specific implementations (Routing for TSP/VRP, CP-SAT for scheduling, etc.)."

2. **小规模实例优势**:
   > "For small instances (N≤100), GenSolver achieves comparable or better solution quality within 1-5 seconds, demonstrating the effectiveness of GPU-accelerated metaheuristics."

3. **坦诚承认大规模挑战**:
   > "For large instances (N≥200), the current implementation requires longer convergence time compared to specialized solvers like OR-Tools GLS. This suggests opportunities for future optimization in initial solution generation and operator tuning for large-scale problems."

4. **强调设计目标**:
   > "Our design prioritizes ease of use and generality over absolute performance. Users can define a new problem in ~50 lines of code, compared to hundreds of lines required for MIP formulations or specialized solver APIs."

---

## 七、后续行动

### 立即行动
1. **诊断 verbose 输出**: 检查 `pcb442` 在 1s 预算下的实际运行参数
2. **对比评估次数**: 计算 GPU vs OR-Tools 的每秒评估次数，看是否存在效率问题
3. **测试初始解质量**: 单独测试初始解的 gap，确认是否是收敛慢的根源

### 中期优化
4. **调优大规模实例参数**: 针对 N≥200 的实例，调整 `init_oversample`、LNS destroy 比例、种群规模
5. **实现启发式初始解**: 为 TSP 添加贪心构造（如 nearest neighbor），提升初始解质量
6. **增加 VRP 对比数据**: 补充更多 VRP 实例的对比

---

## 八、数据文件清单

```
final_20260222/
├── gensolver_all_20260222_175333.csv   (GPU benchmark, 671 行)
├── ortools_20260222_182532.csv         (OR-Tools baseline, 171 行)
├── log_all_20260222_175333.txt         (GPU 运行日志)
├── ortools_20260222_182532.log         (OR-Tools 运行日志)
└── ANALYSIS_SUMMARY.md                 (本报告)
```

---

**结论**: 当前数据揭示了 GPU 求解器在大规模实例上的严重质量问题，需要紧急优化才能支撑论文发表。小规模实例的表现是可接受的，但大规模实例的 gap 差距（5-10 倍）无法用"通用性"来辩护。
