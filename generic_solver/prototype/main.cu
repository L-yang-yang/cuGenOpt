/**
 * main.cu - GenSolver 通用性验证
 * 
 * 用同一个 solve<Problem>() 依次求解 12 种不同类型/编码的问题：
 *   1.  TSP5         — 排列编码, 单目标(min), 无约束
 *   2.  Knapsack6    — 0-1编码, 单目标(max), 容量约束
 *   3.  Assign4      — 排列编码, 单目标(min), 无约束(不同评估)
 *   4.  Schedule     — 0-1矩阵, 多目标(min+min), 人数约束
 *   5.  CVRP10       — 多行排列, 分区初始化, 跨行算子, 容量约束
 *   6.  LoadBalance8 — Integer 编码, 单目标(min), 无约束
 *   7.  GraphColor   — Integer 编码, 纯目标优化(min 冲突)
 *   8.  BinPacking   — Integer 编码, 有约束(容量)
 *   9.  QAP          — 排列编码, 二次交互目标
 *  10.  VRPTW        — 多行排列, 时间窗+容量双约束
 *  11a. JSP(Int)     — Integer 多行 + RowMode::Fixed, 双约束
 *  11b. JSP(Perm)    — Permutation 多重集 + 贪心解码, 无约束
 */

#include "solver.cuh"
#include "tsp.cuh"
#include "knapsack.cuh"
#include "assignment.cuh"
#include "schedule.cuh"
#include "vrp.cuh"
#include "load_balance.cuh"
#include "graph_color.cuh"
#include "bin_packing.cuh"
#include "qap.cuh"
#include "vrptw.cuh"
#include "jsp.cuh"

// ============================================================
// 1. TSP5 — 排列, 最小化距离, 最优=18
// ============================================================
void run_tsp() {
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║  TSP5: 排列编码 | 1目标(min) | 无约束 ║\n");
    printf("║  已知最优: 18.0                      ║\n");
    printf("╚══════════════════════════════════════╝\n");
    
    const int N = 5;
    float dist[N * N] = {
        0, 3, 6, 5, 7,
        3, 0, 3, 4, 5,
        6, 3, 0, 5, 4,
        5, 4, 5, 0, 3,
        7, 5, 4, 3, 0
    };
    
    auto prob = TSPProblem::create(dist, N);
    SolverConfig cfg;
    cfg.pop_size = 64;  cfg.max_gen = 500;
    cfg.seed = 42;  cfg.print_every = 250;
    
    solve(prob, cfg);
    prob.destroy();
}

// ============================================================
// 2. Knapsack6 — 0-1, 最大化价值, 容量约束, 最优=30
// ============================================================
void run_knapsack() {
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  Knapsack6: 0-1编码 | 1目标(MAX) | 容量约束 ║\n");
    printf("║  已知最优: 价值=30, 重量=15              ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    
    const int N = 6;
    float weights[N] = {2, 3, 5, 7, 4, 6};
    float values[N]  = {6, 5, 8, 14, 7, 10};
    float capacity = 15.0f;
    
    auto prob = KnapsackProblem::create(weights, values, N, capacity);
    SolverConfig cfg;
    cfg.pop_size = 64;  cfg.max_gen = 500;
    cfg.seed = 123;  cfg.print_every = 250;
    
    solve(prob, cfg);
    prob.destroy();
}

// ============================================================
// 3. Assign4 — 排列, 最小化成本, 最优=14
// ============================================================
void run_assignment() {
    printf("\n");
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  Assign4: 排列编码 | 1目标(min) | 无约束(隐式) ║\n");
    printf("║  已知最优: [1,2,0,3] 成本=14                 ║\n");
    printf("╚══════════════════════════════════════════════╝\n");
    
    const int N = 4;
    float cost[N * N] = {
        9, 2, 7, 8,
        6, 4, 3, 7,
        5, 8, 1, 8,
        7, 6, 9, 4
    };
    
    auto prob = AssignmentProblem::create(cost, N);
    SolverConfig cfg;
    cfg.pop_size = 64;  cfg.max_gen = 500;
    cfg.seed = 456;  cfg.print_every = 250;
    
    solve(prob, cfg);
    prob.destroy();
}

// ============================================================
// 4. Schedule3x4 — 0-1矩阵, 多目标, 人数约束
//    参考解: 成本=21, 不公平度=1
// ============================================================
void run_schedule() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Schedule: 0-1矩阵 | 2目标(min+min) | 人数约束   ║\n");
    printf("║  参考解: 成本=21, 不公平度=1                     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int DAYS = 3, EMPS = 4, REQUIRED = 2;
    float cost[DAYS * EMPS] = {
        5, 3, 8, 4,
        6, 2, 7, 5,
        4, 6, 3, 7
    };
    
    auto prob = ScheduleProblem::create(cost, DAYS, EMPS, REQUIRED);
    SolverConfig cfg;
    cfg.pop_size = 128;  cfg.max_gen = 1000;
    cfg.seed = 789;  cfg.print_every = 500;
    
    solve(prob, cfg);
    prob.destroy();
}

// ============================================================
// 5. CVRP10 — 多行排列, 分区初始化, 跨行算子, 容量约束
//    10客户 4车 容量=15  参考最优≈200
// ============================================================
void run_vrp() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  CVRP10: 多行排列 | 分区 | 跨行算子 | 容量约束   ║\n");
    printf("║  10客户, 4车, 容量=15, 参考最优≈200              ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int N = 10;
    const int N_NODES = N + 1;
    
    // depot 居中，客户分 4 个方向簇
    float coords[N_NODES][2] = {
        {50, 50},   // depot
        {60, 50}, {70, 50}, {80, 50},   // East:  需求 5+4+6=15
        {50, 60}, {50, 70}, {50, 80},   // North: 需求 5+4+6=15
        {40, 50}, {30, 50},             // West:  需求 5+4=9
        {50, 40}, {50, 30},             // South: 需求 5+6=11
    };
    float demands[N] = {5, 4, 6, 5, 4, 6, 5, 4, 5, 6};
    
    float dist[N_NODES * N_NODES];
    for (int i = 0; i < N_NODES; i++)
        for (int j = 0; j < N_NODES; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * N_NODES + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
    
    auto prob = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
    
    SolverConfig cfg;
    cfg.pop_size = 256;
    cfg.max_gen = 2000;
    cfg.seed = 42;
    cfg.print_every = 1000;
    cfg.num_islands = 4;
    cfg.migrate_interval = 50;
    cfg.migrate_strategy = MigrateStrategy::Hybrid;
    cfg.use_aos = true;  // 启用自适应算子选择
    
    solve(prob, cfg);
    prob.destroy();
}

// ============================================================
// 6. LoadBalance8 — Integer 编码, 最小化 makespan
//    8任务 3机器  处理时间=[5,3,8,4,6,2,7,5]  最优 makespan≈14
//    最优分配: {5,8}=13, {3,4,6}=13, {2,7,5}=14 → makespan=14
// ============================================================
void run_load_balance() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  LoadBalance8: Integer编码 | 1目标(min) | 无约束  ║\n");
    printf("║  8任务 3机器, 参考最优 makespan≈14               ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int N = 8;
    const int M = 3;
    float proc_time[N] = {5, 3, 8, 4, 6, 2, 7, 5};
    
    auto prob = LoadBalanceProblem::create(proc_time, N, M);
    SolverConfig cfg;
    cfg.pop_size = 64;  cfg.max_gen = 1000;
    cfg.seed = 42;  cfg.print_every = 500;
    
    auto result = solve(prob, cfg);
    printf("  → makespan=%.1f\n", result.best_solution.objectives[0]);
    prob.destroy();
}

// ============================================================
// 7. GraphColor — Integer 编码, Petersen 图, 最优冲突=0
// ============================================================
void run_graph_color() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  GraphColor: Integer编码 | 1目标(min) | 无约束    ║\n");
    printf("║  Petersen图 10节点15边, k=3, 最优冲突=0           ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int N = 10;
    const int K = 3;
    // Petersen 图邻接矩阵：外环 0-4, 内星 5-9
    // 外环: 0-1, 1-2, 2-3, 3-4, 4-0
    // 内星: 5-7, 7-9, 9-6, 6-8, 8-5
    // 连接: 0-5, 1-6, 2-7, 3-8, 4-9
    int adj[N * N] = {};
    auto edge = [&](int a, int b) { adj[a * N + b] = 1; adj[b * N + a] = 1; };
    // 外环
    edge(0,1); edge(1,2); edge(2,3); edge(3,4); edge(4,0);
    // 内星
    edge(5,7); edge(7,9); edge(9,6); edge(6,8); edge(8,5);
    // 连接
    edge(0,5); edge(1,6); edge(2,7); edge(3,8); edge(4,9);
    
    auto prob = GraphColorProblem::create(adj, N, K);
    SolverConfig cfg;
    cfg.pop_size = 128;  cfg.max_gen = 1000;
    cfg.seed = 42;  cfg.print_every = 500;
    
    auto result = solve(prob, cfg);
    printf("  → conflicts=%.0f (optimal=0)\n", result.best_solution.objectives[0]);
    prob.destroy();
}

// ============================================================
// 8. BinPacking — Integer 编码, 有约束, 最优=4 箱
// ============================================================
void run_bin_packing() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  BinPacking: Integer编码 | 1目标(min) | 容量约束  ║\n");
    printf("║  8物品, 容量=10, 最优=4箱                        ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int N = 8;
    const int B = 6;  // 最多 6 个箱子
    const float C = 10.0f;
    float weights[N] = {7, 5, 3, 4, 6, 2, 8, 1};
    
    auto prob = BinPackingProblem::create(weights, N, B, C);
    SolverConfig cfg;
    cfg.pop_size = 128;  cfg.max_gen = 2000;
    cfg.seed = 42;  cfg.print_every = 1000;
    
    auto result = solve(prob, cfg);
    printf("  → bins_used=%.0f, penalty=%.1f (optimal=4, penalty=0)\n",
           result.best_solution.objectives[0], result.best_solution.penalty);
    prob.destroy();
}

// ============================================================
// 9. QAP — 排列编码, 二次交互目标, 5x5 实例
// ============================================================
void run_qap() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  QAP5: 排列编码 | 1目标(min) | 无约束             ║\n");
    printf("║  5设施5位置, 已知最优=50                          ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int N = 5;
    // 物流量矩阵（对称）
    float flow[N * N] = {
        0, 5, 2, 4, 1,
        5, 0, 3, 0, 2,
        2, 3, 0, 0, 0,
        4, 0, 0, 0, 5,
        1, 2, 0, 5, 0
    };
    // 距离矩阵（对称）
    float dist[N * N] = {
        0, 1, 2, 3, 4,
        1, 0, 1, 2, 3,
        2, 1, 0, 1, 2,
        3, 2, 1, 0, 1,
        4, 3, 2, 1, 0
    };
    // 最优解：perm=[0,3,4,1,2] → cost=50
    // 验证: flow[0][1]*dist[0][3] + flow[0][2]*dist[0][4] + ... = 50
    
    auto prob = QAPProblem::create(flow, dist, N);
    SolverConfig cfg;
    cfg.pop_size = 128;  cfg.max_gen = 2000;
    cfg.seed = 42;  cfg.print_every = 1000;
    
    auto result = solve(prob, cfg);
    printf("  → cost=%.0f (optimal=50)\n", result.best_solution.objectives[0]);
    prob.destroy();
}

// ============================================================
// 10. VRPTW — 多行排列, 时间窗+容量双约束
//     8客户 3车 容量=15
// ============================================================
void run_vrptw() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  VRPTW8: 多行排列 | 分区 | 时间窗+容量双约束      ║\n");
    printf("║  8客户, 3车, 容量=15                              ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int N = 8;
    const int NN = N + 1;  // 含 depot
    
    // depot(0) 在中心, 客户分布在周围
    float coords[NN][2] = {
        {50, 50},   // depot
        {60, 50},   // 客户 1
        {70, 50},   // 客户 2
        {50, 60},   // 客户 3
        {50, 70},   // 客户 4
        {40, 50},   // 客户 5
        {30, 50},   // 客户 6
        {50, 40},   // 客户 7
        {50, 30},   // 客户 8
    };
    float demands[N] = {3, 5, 4, 6, 3, 5, 4, 5};
    
    // 距离矩阵
    float dist[NN * NN];
    for (int i = 0; i < NN; i++)
        for (int j = 0; j < NN; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * NN + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
    
    // 时间窗：depot [0, 200], 客户各有不同窗口
    // earliest[0]=depot earliest, latest[0]=depot latest
    float earliest[NN] = {0,  0, 10,  0, 20,  0, 30,  0, 10};
    float latest[NN]   = {200, 50, 60, 50, 80, 50, 90, 50, 70};
    float service[NN]  = {0,  5,  5,  5,  5,  5,  5,  5,  5};
    
    auto prob = VRPTWProblem::create(dist, demands, earliest, latest, service,
                                      N, 15.0f, 3, 3);
    SolverConfig cfg;
    cfg.pop_size = 256;  cfg.max_gen = 3000;
    cfg.seed = 42;  cfg.print_every = 1500;
    cfg.num_islands = 4;
    cfg.migrate_interval = 50;
    cfg.use_aos = true;
    
    auto result = solve(prob, cfg);
    printf("  → distance=%.1f, penalty=%.1f\n",
           result.best_solution.objectives[0], result.best_solution.penalty);
    prob.destroy();
}

// ============================================================
// 11a. JSP — Integer 多行 + RowMode::Fixed
//      data[j][i] = 工件j第i工序的开始时间
// ============================================================
void run_jsp() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  JSP3x3(Int): Integer多行 | Fixed | makespan     ║\n");
    printf("║  3工件 3机器 3工序, 参考最优 makespan≈12          ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    // 3 工件, 每个 3 道工序
    // 工件0: M0(3) → M1(2) → M2(4)
    // 工件1: M1(2) → M0(3) → M2(3)
    // 工件2: M2(4) → M1(3) → M0(1)
    const int J = 3, O = 3, M = 3;
    int machine[J * O] = {
        0, 1, 2,   // 工件0 的工序分别在机器 0,1,2
        1, 0, 2,   // 工件1 的工序分别在机器 1,0,2
        2, 1, 0    // 工件2 的工序分别在机器 2,1,0
    };
    float duration[J * O] = {
        3, 2, 4,   // 工件0 各工序耗时
        2, 3, 3,   // 工件1 各工序耗时
        4, 3, 1    // 工件2 各工序耗时
    };
    
    int time_horizon = 30;
    
    auto prob = JSPProblem::create(machine, duration, J, O, M, time_horizon);
    SolverConfig cfg;
    cfg.pop_size = 256;  cfg.max_gen = 5000;
    cfg.seed = 42;  cfg.print_every = 2500;
    
    auto result = solve(prob, cfg);
    printf("  → makespan=%.0f, penalty=%.1f (optimal=12)\n",
           result.best_solution.objectives[0], result.best_solution.penalty);
    prob.destroy();
}

// ============================================================
// 11b. JSP — Permutation 多重集（工序排列编码 + 贪心解码）
//      data[0] = 长度 J*O 的排列，值 j 出现 O 次
// ============================================================
void run_jsp_perm() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  JSP3x3(Perm): 多重集排列 | 贪心解码 | makespan  ║\n");
    printf("║  3工件 3机器 3工序, 参考最优 makespan≈12          ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int J = 3, O = 3, M = 3;
    int machine[J * O] = {
        0, 1, 2,
        1, 0, 2,
        2, 1, 0
    };
    float duration[J * O] = {
        3, 2, 4,
        2, 3, 3,
        4, 3, 1
    };
    
    auto prob = JSPPermProblem::create(machine, duration, J, O, M);
    SolverConfig cfg;
    cfg.pop_size = 256;  cfg.max_gen = 5000;
    cfg.seed = 42;  cfg.print_every = 2500;
    
    auto result = solve(prob, cfg);
    printf("  → makespan=%.0f (optimal=12)\n",
           result.best_solution.objectives[0]);
    prob.destroy();
}

// ============================================================
// 12. 新特性验证 — time_limit / stagnation / init_solutions / SolveResult
// ============================================================
void run_feature_demo() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Feature Demo: time_limit + stagnation + result  ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    const int N = 5;
    float dist[N * N] = {
        0, 3, 6, 5, 7,
        3, 0, 3, 4, 5,
        6, 3, 0, 5, 4,
        5, 4, 5, 0, 3,
        7, 5, 4, 3, 0
    };
    
    // (a) time_limit: 限制 0.01 秒（应提前终止）
    {
        printf("\n--- (a) time_limit=0.01s ---\n");
        auto prob = TSPProblem::create(dist, N);
        SolverConfig cfg;
        cfg.pop_size = 64;  cfg.max_gen = 100000;
        cfg.seed = 42;  cfg.verbose = true;  cfg.print_every = 50000;
        cfg.time_limit_sec = 0.01f;
        auto result = solve(prob, cfg);
        printf("  → stop_reason=%s, gen=%d, obj0=%.1f\n",
               result.stop_reason == StopReason::TimeLimit ? "TimeLimit" :
               result.stop_reason == StopReason::Stagnation ? "Stagnation" : "MaxGen",
               result.generations, result.best_solution.objectives[0]);
        prob.destroy();
    }
    
    // (b) stagnation: 无 SA，收敛后提前终止
    {
        printf("\n--- (b) stagnation_limit=3 (no SA) ---\n");
        auto prob = TSPProblem::create(dist, N);
        SolverConfig cfg;
        cfg.pop_size = 64;  cfg.max_gen = 100000;
        cfg.seed = 42;  cfg.verbose = true;  cfg.print_every = 50000;
        cfg.stagnation_limit = 3;
        auto result = solve(prob, cfg);
        printf("  → stop_reason=%s, gen=%d, obj0=%.1f\n",
               result.stop_reason == StopReason::TimeLimit ? "TimeLimit" :
               result.stop_reason == StopReason::Stagnation ? "Stagnation" : "MaxGen",
               result.generations, result.best_solution.objectives[0]);
        prob.destroy();
    }
    
    // (c) user initial solution: 1 valid + 1 invalid
    {
        printf("\n--- (c) user initial solutions (1 valid + 1 invalid) ---\n");
        auto prob = TSPProblem::create(dist, N);
        
        using Sol = TSPProblem::Sol;
        Sol init_sols[2] = {};
        
        // [0] 合法：已知最优解 0 1 2 4 3 → cost=18
        init_sols[0].dim2_sizes[0] = N;
        init_sols[0].data[0][0] = 0; init_sols[0].data[0][1] = 1;
        init_sols[0].data[0][2] = 2; init_sols[0].data[0][3] = 4;
        init_sols[0].data[0][4] = 3;
        
        // [1] 非法：有重复元素
        init_sols[1].dim2_sizes[0] = N;
        init_sols[1].data[0][0] = 0; init_sols[1].data[0][1] = 0;
        init_sols[1].data[0][2] = 1; init_sols[1].data[0][3] = 2;
        init_sols[1].data[0][4] = 3;
        
        SolverConfig cfg;
        cfg.pop_size = 64;  cfg.max_gen = 100;
        cfg.seed = 42;  cfg.verbose = true;  cfg.print_every = 50;
        auto result = solve(prob, cfg, init_sols, 2);
        printf("  → obj0=%.1f (should be 18.0)\n",
               result.best_solution.objectives[0]);
        prob.destroy();
    }
}

// ============================================================
// 主入口
// ============================================================
int main() {
    print_device_info();
    
    run_tsp();
    run_knapsack();
    run_assignment();
    run_schedule();
    run_vrp();
    run_load_balance();
    run_graph_color();
    run_bin_packing();
    run_qap();
    run_vrptw();
    run_jsp();
    run_jsp_perm();
    
    printf("\n══════════════════════════════════════\n");
    printf("12 种问题/编码全部使用同一个 solve<Problem>() 完成\n");
    printf("覆盖: 排列/0-1/整数/多重集排列, 单行/多行(分区+跨行+固定行), 单/多目标, min/max\n");
    printf("      有无约束, 二次交互, 时间窗, 工序顺序+机器冲突, 贪心解码\n");
    printf("══════════════════════════════════════\n");
    
    run_feature_demo();
    
    // --- CUDA Graph 验证 ---
    {
        printf("\n╔══════════════════════════════════════════╗\n");
        printf("║  CUDA Graph: TSP5 对比验证               ║\n");
        printf("╚══════════════════════════════════════════╝\n");
        
        const int N = 5;
        float dist[N * N] = {
            0, 3, 6, 5, 7,
            3, 0, 3, 4, 5,
            6, 3, 0, 5, 4,
            5, 4, 5, 0, 3,
            7, 5, 4, 3, 0
        };
        
        printf("--- (a) without CUDA Graph ---\n");
        {
            auto prob = TSPProblem::create(dist, N);
            SolverConfig cfg;
            cfg.pop_size = 64;  cfg.max_gen = 500;
            cfg.seed = 42;  cfg.verbose = false;
            auto r = solve(prob, cfg);
            printf("  obj=%.1f  %.0fms\n", r.best_solution.objectives[0], r.elapsed_ms);
            prob.destroy();
        }
        printf("--- (b) with CUDA Graph ---\n");
        {
            auto prob = TSPProblem::create(dist, N);
            SolverConfig cfg;
            cfg.pop_size = 64;  cfg.max_gen = 500;
            cfg.seed = 42;  cfg.verbose = true;  cfg.print_every = 250;
            cfg.use_cuda_graph = true;
            auto r = solve(prob, cfg);
            printf("  obj=%.1f  %.0fms\n", r.best_solution.objectives[0], r.elapsed_ms);
            prob.destroy();
        }
    }
    
    return 0;
}
