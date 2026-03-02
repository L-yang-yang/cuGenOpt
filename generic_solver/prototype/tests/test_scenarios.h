/**
 * test_scenarios.h - 通用性验证场景集
 * 
 * 目的：防止实现过程中过度特化于某个问题类型
 * 用法：每个设计决策/代码修改后，检查是否对所有场景都work
 * 
 * 场景设计原则：
 *   - 每个实例足够小，可以手算验证最优解
 *   - 覆盖不同的编码范式（排列、0-1、整数）
 *   - 覆盖不同的维度使用方式（一维、二维）
 *   - 覆盖单目标和多目标
 *   - 覆盖有约束和无约束
 */

#pragma once
#include <cstdio>

// ============================================================
// 统一的解表示：data[MAX_D1][MAX_D2]
// 所有场景共用同一个结构
// ============================================================
#define MAX_D1 16
#define MAX_D2 32
#define MAX_OBJ 4

struct Solution {
    int data[MAX_D1][MAX_D2];   // 解的原始数据
    int dim1;                    // 实际使用的第一维大小
    int dim2_sizes[MAX_D1];      // 每行的实际大小
    double objectives[MAX_OBJ];  // 目标值
    int num_objectives;          // 实际目标数
    bool is_feasible;            // 可行性
};


// ============================================================
// 场景1: TSP (5个城市)
// 编码: 排列, data[0][0..4] = 城市访问顺序
// 维度: dim1=1, dim2_sizes[0]=5
// 目标: [总距离] (最小化)
// 约束: 无（排列天然可行）
// 最优: 0→1→3→4→2→0, 距离=19
// ============================================================
namespace TSP5 {
    const int N = 5;
    
    // 城市坐标: (0,0) (3,0) (6,0) (3,4) (6,4)
    //
    //     3---4        最优路径: 0-1-3-4-2-0
    //     |   |        距离: 3+5+3+4+6 = 不对，让我重算
    //     0-1-2        
    //
    // 距离矩阵(对称):
    //      0   1   2   3   4
    //  0 [ 0,  3,  6,  5,  7]
    //  1 [ 3,  0,  3,  4,  5]
    //  2 [ 6,  3,  0,  5,  4]
    //  3 [ 5,  4,  5,  0,  3]
    //  4 [ 7,  5,  4,  3,  0]
    
    const double dist[N][N] = {
        { 0, 3, 6, 5, 7},
        { 3, 0, 3, 4, 5},
        { 6, 3, 0, 5, 4},
        { 5, 4, 5, 0, 3},
        { 7, 5, 4, 3, 0}
    };
    
    // 最优路径: 0→1→2→4→3→0, 距离 = 3+3+4+3+5 = 18
    const int optimal_route[] = {0, 1, 2, 4, 3};
    const double optimal_distance = 18.0;
    
    // 编码方式
    // dim1 = 1, dim2_sizes[0] = 5
    // data[0] = [0, 1, 2, 4, 3]  (城市排列)
    
    void init_solution(Solution& sol) {
        sol.dim1 = 1;
        sol.dim2_sizes[0] = N;
        sol.num_objectives = 1;
    }
    
    // 评估: 计算总距离（环路）
    double evaluate(const int* route, int n) {
        double total = 0;
        for (int i = 0; i < n; i++) {
            total += dist[route[i]][route[(i+1) % n]];
        }
        return total;
    }
    
    void print_info() {
        printf("=== TSP5 ===\n");
        printf("编码: 排列 | 维度: [1][5] | 目标: 总距离(min)\n");
        printf("最优: 0-1-2-4-3-0 = %.0f\n\n", optimal_distance);
    }
}


// ============================================================
// 场景2: VRP (1仓库 + 4客户, 2辆车)
// 编码: 分组排列, data[0..1][*] = 两条路径
// 维度: dim1=2, dim2_sizes[k]=路径k的客户数
// 目标: [总距离, 车辆数] (最小化, 最小化)
// 约束: 每辆车容量=10, 客户需求=[0, 4, 3, 5, 6]
//        (仓库=节点0, 需求=0)
// ============================================================
namespace VRP4 {
    const int N = 5;   // 0=仓库, 1-4=客户
    const int K = 2;   // 最多2辆车
    const double capacity = 10.0;
    
    // 坐标: 仓库(0,0), 客户(2,0)(4,0)(2,3)(4,3)
    const double dist[N][N] = {
        { 0, 2, 4, 3, 5},
        { 2, 0, 2, 3, 3},
        { 4, 2, 0, 3, 3},
        { 3, 3, 3, 0, 2},
        { 5, 3, 3, 2, 0}
    };
    
    const double demand[N] = {0, 4, 3, 5, 6};
    // 客户总需求 = 4+3+5+6 = 18 > 单车容量10 → 必须用2辆车
    
    // 一个可行解:
    //   路径0: 1→2     (需求: 4+3=7 ≤ 10) 距离: 0→1 + 1→2 + 2→0 = 2+2+4 = 8
    //   路径1: 3→4     (需求: 5+6=11 > 10!) 不可行!
    //
    // 另一个可行解:
    //   路径0: 1→3     (需求: 4+5=9 ≤ 10) 距离: 0→1 + 1→3 + 3→0 = 2+3+3 = 8
    //   路径1: 2→4     (需求: 3+6=9 ≤ 10) 距离: 0→2 + 2→4 + 4→0 = 4+3+5 = 12
    //   总距离 = 20, 车辆数 = 2
    //
    // 更好的可行解:
    //   路径0: 1→2     (需求: 4+3=7 ≤ 10) 距离: 0→1 + 1→2 + 2→0 = 2+2+4 = 8
    //   路径1: 4→3     (需求: 6+5=11 > 10!) 不可行!
    //
    //   路径0: 2→1     (需求: 3+4=7 ≤ 10) 距离: 0→2 + 2→1 + 1→0 = 4+2+2 = 8
    //   路径1: 3→4     (需求: 5+6=11 > 10!) 不可行!
    //
    //   路径0: 1→4     (需求: 4+6=10 ≤ 10) 距离: 0→1 + 1→4 + 4→0 = 2+3+5 = 10
    //   路径1: 2→3     (需求: 3+5=8 ≤ 10) 距离: 0→2 + 2→3 + 3→0 = 4+3+3 = 10
    //   总距离 = 20, 车辆数 = 2, 可行!
    
    // 编码方式
    // dim1 = 2 (两条路径)
    // data[0] = [1, 4]    dim2_sizes[0] = 2
    // data[1] = [2, 3]    dim2_sizes[1] = 2
    // 注意：路径不含仓库节点0，框架默认"每条路径从仓库出发、回到仓库"由评估函数处理
    
    void init_solution(Solution& sol) {
        sol.dim1 = K;
        sol.num_objectives = 2;  // 距离 + 车辆数
    }
    
    // 评估: 计算总距离和超载量
    // route不含仓库，评估时自动加上 0→route[0]→...→route[n-1]→0
    void evaluate(const int* route, int len, double& route_dist, double& route_load) {
        route_dist = 0;
        route_load = 0;
        if (len == 0) return;
        route_dist += dist[0][route[0]];           // 仓库→第一个客户
        for (int i = 0; i < len - 1; i++) {
            route_dist += dist[route[i]][route[i+1]];
            route_load += demand[route[i]];
        }
        route_load += demand[route[len-1]];
        route_dist += dist[route[len-1]][0];        // 最后客户→仓库
    }
    
    void print_info() {
        printf("=== VRP4 ===\n");
        printf("编码: 分组排列 | 维度: [2][*] | 目标: 距离(min)+车辆数(min)\n");
        printf("约束: 容量=10, 需求=[4,3,5,6]\n");
        printf("可行解示例: [1,4]+[2,3] 距离=20\n\n");
    }
}


// ============================================================
// 场景3: 0-1背包 (6个物品)
// 编码: 0-1向量, data[0][0..5] = 是否选取
// 维度: dim1=1, dim2_sizes[0]=6
// 目标: [总价值] (最大化!)
// 约束: 总重量 ≤ 容量15
// ============================================================
namespace KNAPSACK6 {
    const int N = 6;
    const double capacity = 15.0;
    
    // 物品:       0    1    2    3    4    5
    const double weight[N] = { 2,   3,   5,   7,   4,   6};
    const double value[N]  = { 6,   5,   8,  14,   7,  10};
    // 性价比:                 3.0  1.7  1.6  2.0  1.75 1.67
    
    // 最优: 选物品 0,3,4 → 重量=2+7+4=13≤15, 价值=6+14+7=27
    // 验证: 选0,2,4 → 重量=2+5+4=11, 价值=6+8+7=21 (更差)
    //       选0,3,5 → 重量=2+7+6=15, 价值=6+14+10=30 (更好!)
    //       选3,5   → 重量=7+6=13, 价值=14+10=24
    //       选0,3,4,1→ 重量=2+7+4+3=16>15 不可行
    //       选0,3,5 → 价值=30, 这应该是最优
    
    const int optimal_selection[] = {1, 0, 0, 1, 0, 1};  // 选物品0,3,5
    const double optimal_value = 30.0;
    
    // 编码方式
    // dim1 = 1, dim2_sizes[0] = 6
    // data[0] = [1, 0, 0, 1, 0, 1]  (0-1向量)
    
    void init_solution(Solution& sol) {
        sol.dim1 = 1;
        sol.dim2_sizes[0] = N;
        sol.num_objectives = 1;  // 总价值（最大化，框架内取负变最小化）
    }
    
    double evaluate(const int* selection, int n, double& total_weight) {
        double total_value = 0;
        total_weight = 0;
        for (int i = 0; i < n; i++) {
            if (selection[i]) {
                total_value += value[i];
                total_weight += weight[i];
            }
        }
        return total_value;
    }
    
    bool is_feasible(const int* selection, int n) {
        double w = 0;
        for (int i = 0; i < n; i++)
            if (selection[i]) w += weight[i];
        return w <= capacity;
    }
    
    void print_info() {
        printf("=== KNAPSACK6 ===\n");
        printf("编码: 0-1 | 维度: [1][6] | 目标: 总价值(max)\n");
        printf("约束: 容量=15\n");
        printf("最优: 选{0,3,5} 价值=30 重量=15\n\n");
    }
}


// ============================================================
// 场景4: 指派问题 (4人 × 4任务)
// 编码: 排列, data[0][0..3] = 任务分配
//   data[0][j] = 第j个人分配到的任务编号
// 维度: dim1=1, dim2_sizes[0]=4
// 目标: [总成本] (最小化)
// 约束: 每人恰好一个任务，每任务恰好一人（排列天然满足）
// ============================================================
namespace ASSIGN4 {
    const int N = 4;
    
    // 成本矩阵: cost[人][任务]
    //           任务0 任务1 任务2 任务3
    const double cost[N][N] = {
        {  9,  2,  7,  8},   // 人0
        {  6,  4,  3,  7},   // 人1
        {  5,  8,  1,  8},   // 人2
        {  7,  6,  9,  4}    // 人3
    };
    
    // 最优（匈牙利算法可验证）:
    // 人0→任务1(2), 人1→任务2(3), 人2→任务0(5), 人3→任务3(4)
    // 排列: [1, 2, 0, 3], 总成本 = 2+3+5+4 = 14
    
    const int optimal_assign[] = {1, 2, 0, 3};
    const double optimal_cost = 14.0;
    
    // 编码方式
    // dim1 = 1, dim2_sizes[0] = 4
    // data[0] = [1, 2, 0, 3]  (人i分配到任务data[0][i])
    
    void init_solution(Solution& sol) {
        sol.dim1 = 1;
        sol.dim2_sizes[0] = N;
        sol.num_objectives = 1;
    }
    
    double evaluate(const int* assign, int n) {
        double total = 0;
        for (int i = 0; i < n; i++) {
            total += cost[i][assign[i]];
        }
        return total;
    }
    
    void print_info() {
        printf("=== ASSIGN4 ===\n");
        printf("编码: 排列 | 维度: [1][4] | 目标: 总成本(min)\n");
        printf("最优: [1,2,0,3] 成本=14\n\n");
    }
}


// ============================================================
// 场景5: 排班问题 (3天 × 4员工, 每天需2人)
// 编码: 0-1矩阵, data[day][emp] = 是否在该天上班
// 维度: dim1=3, dim2_sizes[i]=4
// 目标: [总成本, 不公平度] (最小化, 最小化)
// 约束: 每天恰好2人上班
// ============================================================
namespace SCHEDULE3x4 {
    const int DAYS = 3;
    const int EMPS = 4;
    const int REQUIRED = 2;  // 每天需要2人
    
    // 成本: cost[天][人] 表示第d天安排第e人上班的成本
    const double cost[DAYS][EMPS] = {
        { 5,  3,  8,  4},   // 周一
        { 6,  2,  7,  5},   // 周二
        { 4,  6,  3,  7}    // 周三
    };
    
    // 一个可行解:
    //   周一: 员工1,3 (成本3+4=7)
    //   周二: 员工1,3 (成本2+5=7)
    //   周三: 员工0,2 (成本4+3=7)
    //   总成本=21, 工作天数=[1,2,1,2], 不公平度=max-min=2-1=1
    //
    // 更好的:
    //   周一: 员工1,3 (成本3+4=7)
    //   周二: 员工0,1 (成本6+2=8)
    //   周三: 员工2,3 (成本3+7=10)
    //   总成本=25, 工作天数=[1,2,1,2], 不公平度=1
    //   不，总成本更高了
    //
    // 最优目标1(成本最低):
    //   周一: 员工1,3 (3+4=7)
    //   周二: 员工1,3 (2+5=7)
    //   周三: 员工2,0 (3+4=7)
    //   总成本=21
    //   工作天数=[1,2,1,2] → 不公平度=1
    
    // 编码方式
    // dim1 = 3 (天), dim2_sizes[i] = 4 (员工)
    // data[d][e] = 0或1
    
    void init_solution(Solution& sol) {
        sol.dim1 = DAYS;
        for (int d = 0; d < DAYS; d++) sol.dim2_sizes[d] = EMPS;
        sol.num_objectives = 2;  // 成本 + 不公平度
    }
    
    void evaluate(const int data[MAX_D1][MAX_D2], 
                  double& total_cost, double& unfairness, bool& feasible) {
        total_cost = 0;
        feasible = true;
        int workdays[EMPS] = {0};
        
        for (int d = 0; d < DAYS; d++) {
            int count = 0;
            for (int e = 0; e < EMPS; e++) {
                if (data[d][e]) {
                    total_cost += cost[d][e];
                    workdays[e]++;
                    count++;
                }
            }
            if (count != REQUIRED) feasible = false;  // 每天必须恰好2人
        }
        
        int max_w = 0, min_w = DAYS;
        for (int e = 0; e < EMPS; e++) {
            if (workdays[e] > max_w) max_w = workdays[e];
            if (workdays[e] < min_w) min_w = workdays[e];
        }
        unfairness = max_w - min_w;
    }
    
    void print_info() {
        printf("=== SCHEDULE3x4 ===\n");
        printf("编码: 0-1矩阵 | 维度: [3][4] | 目标: 成本(min)+不公平度(min)\n");
        printf("约束: 每天恰好2人上班\n");
        printf("参考解: 总成本=21, 不公平度=1\n\n");
    }
}


// ============================================================
// 通用性检查清单
// 每次修改代码后对照此表检查
// ============================================================
//
// ┌─────────┬──────────┬──────┬──────┬───────┬──────────┐
// │ 场景     │ 编码类型  │ dim1 │ dim2 │ 目标数 │ 有约束？  │
// ├─────────┼──────────┼──────┼──────┼───────┼──────────┤
// │ TSP5    │ 排列      │ 1    │ 5    │ 1     │ 无       │
// │ VRP4    │ 分组排列  │ 2    │ 变长 │ 2     │ 容量     │
// │ KNAPSACK│ 0-1      │ 1    │ 6    │ 1     │ 重量     │
// │ ASSIGN4 │ 排列      │ 1    │ 4    │ 1     │ 无(隐式) │
// │ SCHEDULE│ 0-1矩阵  │ 3    │ 4    │ 2     │ 人数     │
// └─────────┴──────────┴──────┴──────┴───────┴──────────┘
//
// 关键差异点（任何改动都需要对照）：
//
// 1. dim1=1 vs dim1>1
//    TSP/背包/指派用一行，VRP/排班用多行
//    → 算子是否支持单行和多行？
//
// 2. 排列 vs 0-1
//    TSP/VRP/指派是排列（元素不重复），背包/排班是0-1
//    → swap是否都合法？排列swap合法，0-1 swap可能需要翻转
//
// 3. 定长 vs 变长
//    TSP/背包/指派各行定长，VRP各路径变长
//    → dim2_sizes是否被正确使用？
//
// 4. 单目标 vs 多目标
//    TSP/背包/指派单目标，VRP/排班多目标
//    → 比较器是否处理多目标？
//
// 5. 最小化 vs 最大化
//    大部分最小化，背包最大化
//    → 比较器方向是否可配？
//
// 6. 约束
//    TSP/指派无显式约束，VRP有容量，背包有重量，排班有人数
//    → 可行性判断是否灵活？


// ============================================================
// 打印所有场景信息
// ============================================================
inline void print_all_scenarios() {
    printf("========== GenSolver 通用性验证场景 ==========\n\n");
    TSP5::print_info();
    VRP4::print_info();
    KNAPSACK6::print_info();
    ASSIGN4::print_info();
    SCHEDULE3x4::print_info();
    printf("===============================================\n");
}
