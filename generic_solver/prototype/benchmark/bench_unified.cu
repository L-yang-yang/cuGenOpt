/**
 * bench_unified.cu — 统一 Benchmark 程序
 *
 * 所有实验通过 solve() 公开接口完成，零侵入核心代码。
 * 输出标准 CSV，便于后续分析。
 *
 * 用法:
 *   ./bench_unified              → 跑全部实验
 *   ./bench_unified exp1         → 只跑通用性验证
 *   ./bench_unified exp2         → TSP 标准实例质量
 *   ./bench_unified exp3         → VRP 标准实例
 *   ./bench_unified exp4         → 可扩展性
 *   ./bench_unified exp5         → 消融实验
 *   ./bench_unified exp6         → 时间-质量曲线（与 exp2 共用数据）
 */

#include "../core/solver.cuh"
#include "../problems/tsp.cuh"
#include "../problems/tsp_large.cuh"
#include "../problems/tsp_xlarge.cuh"
#include "../problems/knapsack.cuh"
#include "../problems/assignment.cuh"
#include "../problems/schedule.cuh"
#include "../problems/vrp.cuh"
#include "../problems/load_balance.cuh"
#include "../problems/graph_color.cuh"
#include "../problems/bin_packing.cuh"
#include "../problems/qap.cuh"
#include "../problems/vrptw.cuh"
#include "../problems/jsp.cuh"
#include "../problems/tsplib_data.h"
#include <cmath>
#include <cstdio>
#include <cstring>

// ============================================================
// 常量
// ============================================================
static const unsigned SEEDS[] = {42, 123, 456, 789, 2024};
static const int NUM_SEEDS = 5;

// ============================================================
// EUC_2D 距离计算（TSPLIB 标准）
// ============================================================
static void compute_euc2d_dist(float* dist, const float coords[][2], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
}

// ============================================================
// GPU 预热（首次 kernel 编译开销不计入实验时间）
// ============================================================
static void warmup() {
    float dd[25] = {};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            dd[i * 5 + j] = (i == j) ? 0 : 10;
    auto p = TSPProblem::create(dd, 5);
    SolverConfig c;
    c.pop_size = 64; c.max_gen = 10; c.seed = 1; c.verbose = false;
    solve(p, c);
    p.destroy();
}

// ============================================================
// 通用求解模板：对每个 seed 调 solve()，输出 CSV 行
// ============================================================
// gap 计算：minimize 问题 gap=(obj-opt)/opt, maximize 问题 gap=(opt-obj)/opt
// known_optimal > 0 表示 minimize，< 0 表示 maximize（绝对值为最优值）
// known_optimal == 0 表示不计算 gap
static float calc_gap(float obj, float known_optimal) {
    if (known_optimal == 0.0f) return 0.0f;
    if (known_optimal > 0.0f)
        return (obj - known_optimal) / known_optimal * 100.0f;
    float opt_abs = -known_optimal;
    return (opt_abs - obj) / opt_abs * 100.0f;
}

template<typename Problem>
void run_experiment(const char* instance,
                    const char* config_name,
                    Problem& prob,
                    const SolverConfig& cfg,
                    float known_optimal,
                    int num_seeds = NUM_SEEDS) {
    for (int s = 0; s < num_seeds; s++) {
        SolverConfig c = cfg;
        c.seed = SEEDS[s];
        c.verbose = false;

        auto result = solve(prob, c);

        float obj = result.best_solution.objectives[0];
        float pen = result.best_solution.penalty;
        float gap = calc_gap(obj, known_optimal);
        const char* reason = (result.stop_reason == StopReason::TimeLimit)  ? "time" :
                             (result.stop_reason == StopReason::Stagnation) ? "stag" : "gen";

        printf("%s,%s,%u,%.2f,%.2f,%.1f,%.2f,%d,%s\n",
               instance, config_name, SEEDS[s],
               obj, pen, result.elapsed_ms, gap,
               result.generations, reason);
        fflush(stdout);
    }
}

// 需要每次重建 Problem 的版本（因为 solve 后 Problem 内部状态可能变化）
template<typename CreateFn>
void run_experiment_recreate(const char* instance,
                             const char* config_name,
                             CreateFn create_fn,
                             const SolverConfig& cfg,
                             float known_optimal,
                             int num_seeds = NUM_SEEDS) {
    for (int s = 0; s < num_seeds; s++) {
        SolverConfig c = cfg;
        c.seed = SEEDS[s];
        c.verbose = false;

        auto prob = create_fn();
        auto result = solve(prob, c);

        float obj = result.best_solution.objectives[0];
        float pen = result.best_solution.penalty;
        float gap = calc_gap(obj, known_optimal);
        const char* reason = (result.stop_reason == StopReason::TimeLimit)  ? "time" :
                             (result.stop_reason == StopReason::Stagnation) ? "stag" : "gen";

        printf("%s,%s,%u,%.2f,%.2f,%.1f,%.2f,%d,%s\n",
               instance, config_name, SEEDS[s],
               obj, pen, result.elapsed_ms, gap,
               result.generations, reason);
        fflush(stdout);

        prob.destroy();
    }
}

// ============================================================
// CSV header
// ============================================================
static void print_csv_header() {
    printf("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason\n");
    fflush(stdout);
}

// ============================================================
// 默认求解配置工厂
//
// 设计原则：展示求解器的开箱即用能力，不针对特定问题调参。
// make_default_config = 唯一推荐配置，所有组件全部启用，种群自适应。
// ============================================================

// 开箱即用默认配置：auto pop + SA + Isl(auto) + CX + AOS
static SolverConfig make_default_config(int gen = 5000) {
    SolverConfig c;
    c.pop_size = 0;  // 自适应种群
    c.max_gen = gen;
    c.verbose = false;
    c.sa_temp_init = 50.0f;
    c.sa_alpha = 0.999f;
    c.num_islands = 0;  // 自适应岛屿数
    c.migrate_interval = 50;
    c.migrate_strategy = MigrateStrategy::Hybrid;
    c.crossover_rate = 0.1f;
    c.use_aos = true;
    return c;
}

// 时间限制配置：基于 default，按时间预算运行
static SolverConfig make_timed_config(float seconds) {
    SolverConfig c = make_default_config(999999);
    c.time_limit_sec = seconds;
    c.stagnation_limit = 0;
    return c;
}

// 消融专用：裸 HC（无 SA、无岛、无 CX、无 AOS）
static SolverConfig make_hc_config(int gen = 10000) {
    SolverConfig c;
    c.pop_size = 0;  // 自适应种群
    c.max_gen = gen;
    c.verbose = false;
    return c;
}

// ============================================================
//  Exp1: 通用性验证 — 12 种问题/编码
//  所有问题统一使用 make_default_config（开箱即用），不针对性调参。
//  种群自适应，代数统一 2000（小规模问题足够收敛）。
// ============================================================
static void exp1_generality() {
    fprintf(stderr, "[exp1] Generality validation — 12 problems (default config)\n");

    const int GEN = 2000;
    const char* cfg_name = "default_g2k";

    // 1. TSP5
    {
        const int N = 5;
        float dist[N * N] = {
            0,3,6,5,7, 3,0,3,4,5, 6,3,0,5,4, 5,4,5,0,3, 7,5,4,3,0
        };
        auto prob = TSPProblem::create(dist, N);
        SolverConfig c = make_default_config(GEN);
        run_experiment("TSP5", cfg_name, prob, c, 18.0f);
        prob.destroy();
    }

    // 2. Knapsack6
    {
        const int N = 6;
        float w[N] = {2,3,5,7,4,6};
        float v[N] = {6,5,8,14,7,10};
        auto prob = KnapsackProblem::create(w, v, N, 15.0f);
        SolverConfig c = make_default_config(GEN);
        run_experiment("Knapsack6", cfg_name, prob, c, -30.0f);
        prob.destroy();
    }

    // 3. Assignment4
    {
        const int N = 4;
        float cost[N * N] = {9,2,7,8, 6,4,3,7, 5,8,1,8, 7,6,9,4};
        auto prob = AssignmentProblem::create(cost, N);
        SolverConfig c = make_default_config(GEN);
        run_experiment("Assign4", cfg_name, prob, c, 14.0f);
        prob.destroy();
    }

    // 4. Schedule3x4
    {
        float cost[12] = {5,3,8,4, 6,2,7,5, 4,6,3,7};
        auto prob = ScheduleProblem::create(cost, 3, 4, 2);
        SolverConfig c = make_default_config(GEN);
        run_experiment("Schedule3x4", cfg_name, prob, c, 0.0f);
        prob.destroy();
    }

    // 5. CVRP10
    {
        const int N = 10, NN = N + 1;
        float coords[NN][2] = {
            {50,50},{60,50},{70,50},{80,50},{50,60},{50,70},{50,80},{40,50},{30,50},{50,40},{50,30}
        };
        float demands[N] = {5,4,6,5,4,6,5,4,5,6};
        float dist[NN * NN];
        for (int i = 0; i < NN; i++)
            for (int j = 0; j < NN; j++) {
                float dx = coords[i][0] - coords[j][0];
                float dy = coords[i][1] - coords[j][1];
                dist[i * NN + j] = roundf(sqrtf(dx * dx + dy * dy));
            }
        auto prob = VRPProblem::create(dist, demands, N, 15.0f, 4, 4);
        SolverConfig c = make_default_config(GEN);
        run_experiment("CVRP10", cfg_name, prob, c, 200.0f);
        prob.destroy();
    }

    // 6. LoadBalance8
    {
        float pt[8] = {5,3,8,4,6,2,7,5};
        auto prob = LoadBalanceProblem::create(pt, 8, 3);
        SolverConfig c = make_default_config(GEN);
        run_experiment("LoadBal8", cfg_name, prob, c, 14.0f);
        prob.destroy();
    }

    // 7. GraphColor10 (Petersen)
    {
        const int N = 10;
        int adj[N * N] = {};
        auto edge = [&](int a, int b) { adj[a*N+b] = 1; adj[b*N+a] = 1; };
        edge(0,1); edge(1,2); edge(2,3); edge(3,4); edge(4,0);
        edge(5,7); edge(7,9); edge(9,6); edge(6,8); edge(8,5);
        edge(0,5); edge(1,6); edge(2,7); edge(3,8); edge(4,9);
        auto prob = GraphColorProblem::create(adj, N, 3);
        SolverConfig c = make_default_config(GEN);
        run_experiment("GraphColor10", cfg_name, prob, c, 0.0f);
        prob.destroy();
    }

    // 8. BinPacking8
    {
        float w[8] = {7,5,3,4,6,2,8,1};
        auto prob = BinPackingProblem::create(w, 8, 6, 10.0f);
        SolverConfig c = make_default_config(GEN);
        run_experiment("BinPack8", cfg_name, prob, c, 4.0f);
        prob.destroy();
    }

    // 9. QAP5
    {
        const int N = 5;
        float flow[N*N] = {0,5,2,4,1, 5,0,3,0,2, 2,3,0,0,0, 4,0,0,0,5, 1,2,0,5,0};
        float dist[N*N] = {0,1,2,3,4, 1,0,1,2,3, 2,1,0,1,2, 3,2,1,0,1, 4,3,2,1,0};
        auto prob = QAPProblem::create(flow, dist, N);
        SolverConfig c = make_default_config(GEN);
        run_experiment("QAP5", cfg_name, prob, c, 50.0f);
        prob.destroy();
    }

    // 10. VRPTW8
    {
        const int N = 8, NN = N + 1;
        float coords[NN][2] = {
            {50,50},{60,50},{70,50},{50,60},{50,70},{40,50},{30,50},{50,40},{50,30}
        };
        float demands[N] = {3,5,4,6,3,5,4,5};
        float dist[NN * NN];
        for (int i = 0; i < NN; i++)
            for (int j = 0; j < NN; j++) {
                float dx = coords[i][0] - coords[j][0];
                float dy = coords[i][1] - coords[j][1];
                dist[i * NN + j] = roundf(sqrtf(dx * dx + dy * dy));
            }
        float earliest[NN] = {0, 0,10, 0,20, 0,30, 0,10};
        float latest[NN]   = {200,50,60,50,80,50,90,50,70};
        float service[NN]  = {0, 5,5,5,5,5,5,5,5};
        auto prob = VRPTWProblem::create(dist, demands, earliest, latest, service,
                                          N, 15.0f, 3, 3);
        SolverConfig c = make_default_config(GEN);
        run_experiment("VRPTW8", cfg_name, prob, c, 0.0f);
        prob.destroy();
    }

    // 11a. JSP3x3 (Integer)
    {
        int machine[9] = {0,1,2, 1,0,2, 2,1,0};
        float duration[9] = {3,2,4, 2,3,3, 4,3,1};
        auto prob = JSPProblem::create(machine, duration, 3, 3, 3, 30);
        SolverConfig c = make_default_config(GEN);
        run_experiment("JSP3x3_Int", cfg_name, prob, c, 12.0f);
        prob.destroy();
    }

    // 11b. JSP3x3 (Perm multiset)
    {
        int machine[9] = {0,1,2, 1,0,2, 2,1,0};
        float duration[9] = {3,2,4, 2,3,3, 4,3,1};
        auto prob = JSPPermProblem::create(machine, duration, 3, 3, 3);
        SolverConfig c = make_default_config(GEN);
        run_experiment("JSP3x3_Perm", cfg_name, prob, c, 12.0f);
        prob.destroy();
    }
}

// ============================================================
//  Exp2: TSP 标准实例质量（多配置 + 多时间预算）
// ============================================================

// TSP 实例数据（eil51, eil76, kroA100 内嵌在 bench_quality.cu 中，
// ch150/tsp225/lin318/pcb442 在 tsplib_data.h 中）
// 这里重新内嵌 eil51/kroA100 坐标以保持独立性

static const int EIL51_N = 51;
static const float eil51_coords[EIL51_N][2] = {
    {37,52},{49,49},{52,64},{20,26},{40,30},{21,47},{17,63},{31,62},{52,33},
    {51,21},{42,41},{31,32},{ 5,25},{12,42},{36,16},{52,41},{27,23},{17,33},
    {13,13},{57,58},{62,42},{42,57},{16,57},{ 8,52},{ 7,38},{27,68},{30,48},
    {43,67},{58,48},{58,27},{37,69},{38,46},{46,10},{61,33},{62,63},{63,69},
    {32,22},{45,35},{59,15},{ 5, 6},{10,17},{21,10},{ 5,64},{30,15},{39,10},
    {32,39},{25,32},{25,55},{48,28},{56,37},{30,40}
};

static const int KROA100_N = 100;
static const float kroA100_coords[KROA100_N][2] = {
    {1380,939},{2848,96},{3510,1671},{457,334},{3888,666},{984,965},{2721,1482},
    {1286,525},{2716,1432},{738,1325},{1251,1832},{2728,1698},{3815,169},{3683,1533},
    {1247,1945},{123,862},{1234,1946},{252,1240},{611,673},{2576,1676},{928,1700},
    {53,857},{1807,1711},{274,1420},{2574,946},{178,24},{2678,1825},{1795,962},
    {3384,1498},{3520,1079},{1256,61},{1424,1728},{3913,192},{3085,1528},{2573,1969},
    {463,1670},{3875,598},{298,1513},{3479,821},{2542,236},{3955,1743},{1323,280},
    {3447,1830},{2936,337},{1621,1830},{3373,1646},{1393,1368},{3874,1318},{938,955},
    {3022,474},{2482,1183},{3854,923},{376,825},{2519,135},{2945,1622},{953,268},
    {2628,1479},{2097,981},{890,1846},{2139,1806},{2421,1007},{2290,1810},{1115,1052},
    {2588,302},{327,265},{241,341},{1917,687},{2991,792},{2573,599},{19,674},
    {3911,1673},{872,1559},{2863,558},{929,1766},{839,620},{3893,102},{2178,1619},
    {3822,899},{378,1048},{1178,100},{2599,901},{3416,143},{2961,1605},{611,1384},
    {3113,885},{2597,1830},{2586,1286},{161,906},{1429,134},{742,1025},{1625,1651},
    {1187,706},{1787,1009},{22,987},{3640,43},{3756,882},{776,392},{1724,1642},
    {198,1810},{3950,1558}
};

struct TSPInstance {
    const char* name;
    const float (*coords)[2];
    int n;
    float optimal;
};

static void exp2_tsp_quality() {
    fprintf(stderr, "[exp2] TSP standard instances — quality + time budgets\n");

    TSPInstance instances[] = {
        {"eil51",   eil51_coords,   EIL51_N,   426.0f},
        {"kroA100", kroA100_coords, KROA100_N, 21282.0f},
        {"ch150",   CH150_coords,   CH150_N,   6528.0f},
        {"tsp225",  TSP225_coords,  TSP225_N,  3916.0f},
        {"lin318",  LIN318_coords,  LIN318_N,  42029.0f},
        {"pcb442",  PCB442_coords,  PCB442_N,  50778.0f},
    };

    float time_budgets[] = {1.0f, 5.0f, 10.0f, 30.0f, 60.0f};

    for (auto& inst : instances) {
        fprintf(stderr, "  [exp2] %s (n=%d)\n", inst.name, inst.n);

        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);

        // 选择 Problem 类型：N<=64 用 TSPProblem, N<=256 用 TSPLarge, N<=512 用 TSPXLarge
        for (float t : time_budgets) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "full_t%.0fs", t);

            SolverConfig c = make_timed_config(t);

            if (inst.n <= 64) {
                run_experiment_recreate(inst.name, cfg_name,
                    [&]() { return TSPProblem::create(dist, inst.n); },
                    c, inst.optimal);
            } else if (inst.n <= 256) {
                run_experiment_recreate(inst.name, cfg_name,
                    [&]() { return TSPLargeProblem::create(dist, inst.n); },
                    c, inst.optimal);
            } else {
                run_experiment_recreate(inst.name, cfg_name,
                    [&]() { return TSPXLargeProblem::create(dist, inst.n); },
                    c, inst.optimal);
            }
        }

        delete[] dist;
    }
}

// ============================================================
//  Exp3: VRP 标准实例
// ============================================================

static const int AN32K5_N = 31;
static const int AN32K5_NODES = 32;
static const float an32k5_coords[AN32K5_NODES][2] = {
    {82,76},
    {96,44},{50,5},{49,8},{13,7},{29,89},{58,30},{84,39},{14,24},{2,39},
    {3,82},{5,10},{98,52},{84,25},{61,59},{1,65},{88,51},{91,2},{19,32},
    {93,3},{50,93},{98,14},{5,42},{42,9},{61,62},{9,97},{80,55},{57,69},
    {23,15},{20,70},{85,60},{98,5}
};
static const float an32k5_demands[AN32K5_N] = {
    19,21,6,19,7,12,16,6,16,8,14,21,16,3,22,18,19,1,24,8,12,4,8,24,24,2,20,15,2,14,9
};

static void exp3_vrp_quality() {
    fprintf(stderr, "[exp3] VRP standard instances\n");

    float dist[AN32K5_NODES * AN32K5_NODES];
    compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);

    float time_budgets[] = {1.0f, 5.0f, 10.0f, 30.0f};

    for (float t : time_budgets) {
        char cfg_name[64];
        snprintf(cfg_name, sizeof(cfg_name), "full_t%.0fs", t);

        SolverConfig c = make_timed_config(t);

        run_experiment_recreate("A-n32-k5", cfg_name,
            [&]() { return VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5); },
            c, 784.0f);
    }
}

// ============================================================
//  Exp4: 可扩展性（固定时间预算，TSP 规模递增）
// ============================================================
static void exp4_scalability() {
    fprintf(stderr, "[exp4] Scalability — fixed time budget, increasing N\n");

    TSPInstance instances[] = {
        {"eil51",   eil51_coords,   EIL51_N,   426.0f},
        {"kroA100", kroA100_coords, KROA100_N, 21282.0f},
        {"ch150",   CH150_coords,   CH150_N,   6528.0f},
        {"tsp225",  TSP225_coords,  TSP225_N,  3916.0f},
        {"lin318",  LIN318_coords,  LIN318_N,  42029.0f},
        {"pcb442",  PCB442_coords,  PCB442_N,  50778.0f},
    };

    float time_budgets[] = {5.0f, 10.0f, 30.0f};

    for (auto& inst : instances) {
        fprintf(stderr, "  [exp4] %s (n=%d)\n", inst.name, inst.n);

        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);

        for (float t : time_budgets) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "scale_t%.0fs", t);

            SolverConfig c = make_timed_config(t);

            if (inst.n <= 64) {
                run_experiment_recreate(inst.name, cfg_name,
                    [&]() { return TSPProblem::create(dist, inst.n); },
                    c, inst.optimal);
            } else if (inst.n <= 256) {
                run_experiment_recreate(inst.name, cfg_name,
                    [&]() { return TSPLargeProblem::create(dist, inst.n); },
                    c, inst.optimal);
            } else {
                run_experiment_recreate(inst.name, cfg_name,
                    [&]() { return TSPXLargeProblem::create(dist, inst.n); },
                    c, inst.optimal);
            }
        }

        delete[] dist;
    }
}

// ============================================================
//  Exp5: 消融实验
//  覆盖 3 种编码（Perm/Binary/Integer）× 有区分度的实例
//  同时做 additive（自底向上）和 leave-one-out（自顶向下）
//  所有配置使用自适应种群（pop=0）
// ============================================================

// 构建消融配置集（通用，不依赖具体问题）
struct AblationConfig {
    const char* name;
    SolverConfig cfg;
};

static constexpr int ABLATION_GEN = 10000;

static void build_ablation_configs(AblationConfig* out, int& count) {
    count = 0;

    // Full = default config
    SolverConfig full = make_default_config(ABLATION_GEN);

    // --- Additive ---
    SolverConfig hc = make_hc_config(ABLATION_GEN);

    SolverConfig sa = make_hc_config(ABLATION_GEN);
    sa.sa_temp_init = 50.0f;
    sa.sa_alpha = 0.999f;

    SolverConfig sa_isl = sa;
    sa_isl.num_islands = 4;
    sa_isl.migrate_interval = 50;
    sa_isl.migrate_strategy = MigrateStrategy::Hybrid;

    SolverConfig sa_isl_cx = sa_isl;
    sa_isl_cx.crossover_rate = 0.1f;

    // --- Leave-one-out ---
    SolverConfig no_sa = full;
    no_sa.sa_temp_init = 0.0f;

    SolverConfig no_isl = full;
    no_isl.num_islands = 1;

    SolverConfig no_cx = full;
    no_cx.crossover_rate = 0.0f;

    SolverConfig no_aos = full;
    no_aos.use_aos = false;

    out[count++] = {"HC",          hc};
    out[count++] = {"SA",          sa};
    out[count++] = {"SA_Isl4",     sa_isl};
    out[count++] = {"SA_Isl4_CX",  sa_isl_cx};
    out[count++] = {"Full",        full};
    out[count++] = {"Full_noSA",   no_sa};
    out[count++] = {"Full_noIsl",  no_isl};
    out[count++] = {"Full_noCX",   no_cx};
    out[count++] = {"Full_noAOS",  no_aos};
}

static void exp5_ablation() {
    fprintf(stderr, "[exp5] Ablation study — TSP + BinPacking + GraphColor\n");

    AblationConfig configs[16];
    int num_configs = 0;
    build_ablation_configs(configs, num_configs);

    // --- Part A: TSP (Permutation encoding) ---
    {
        struct TSPInst { const char* name; const float (*coords)[2]; int n; float optimal; };
        TSPInst tsp_instances[] = {
            {"kroA100", kroA100_coords, KROA100_N, 21282.0f},
            {"ch150",   CH150_coords,   CH150_N,   6528.0f},
        };

        for (auto& inst : tsp_instances) {
            fprintf(stderr, "  [exp5] TSP %s (n=%d)\n", inst.name, inst.n);
            float* dist = new float[inst.n * inst.n];
            compute_euc2d_dist(dist, inst.coords, inst.n);

            for (int i = 0; i < num_configs; i++) {
                run_experiment_recreate(inst.name, configs[i].name,
                    [&]() { return TSPLargeProblem::create(dist, inst.n); },
                    configs[i].cfg, inst.optimal);
            }
            delete[] dist;
        }
    }

    // --- Part B: BinPacking (Integer encoding) ---
    {
        fprintf(stderr, "  [exp5] BinPacking20\n");
        const int N = 20;
        float weights[N] = {7,5,3,4,6,2,8,1,9,3,5,7,4,6,2,8,3,5,7,4};
        auto prob_factory = [&]() {
            return BinPackingProblem::create(weights, N, 8, 15.0f);
        };
        for (int i = 0; i < num_configs; i++) {
            run_experiment_recreate("BinPack20", configs[i].name,
                prob_factory, configs[i].cfg, 0.0f);
        }
    }

    // --- Part C: GraphColor (Integer encoding) ---
    {
        fprintf(stderr, "  [exp5] GraphColor20\n");
        const int N = 20;
        int adj[N * N] = {};
        auto edge = [&](int a, int b) { adj[a*N+b] = 1; adj[b*N+a] = 1; };
        // 4-regular random-ish graph on 20 nodes
        edge(0,1); edge(0,5); edge(0,10); edge(0,15);
        edge(1,2); edge(1,6); edge(1,11);
        edge(2,3); edge(2,7); edge(2,12);
        edge(3,4); edge(3,8); edge(3,13);
        edge(4,5); edge(4,9); edge(4,14);
        edge(5,6); edge(5,16);
        edge(6,7); edge(6,17);
        edge(7,8); edge(7,18);
        edge(8,9); edge(8,19);
        edge(9,10); edge(9,15);
        edge(10,11); edge(10,16);
        edge(11,12); edge(11,17);
        edge(12,13); edge(12,18);
        edge(13,14); edge(13,19);
        edge(14,15); edge(14,16);
        edge(15,17);
        edge(16,18);
        edge(17,19);
        edge(18,0);
        edge(19,1);

        auto prob_factory = [&]() {
            return GraphColorProblem::create(adj, N, 4);
        };
        for (int i = 0; i < num_configs; i++) {
            run_experiment_recreate("GraphColor20", configs[i].name,
                prob_factory, configs[i].cfg, 0.0f);
        }
    }

    // --- Part D: Schedule (Binary encoding) ---
    {
        fprintf(stderr, "  [exp5] Schedule5x6\n");
        const int W = 5, S = 6, REQ = 3;
        float cost[W * S] = {
            5,3,8,4,6,2,  6,2,7,5,3,4,  4,6,3,7,5,8,
            7,4,5,3,6,2,  3,5,4,6,2,7
        };
        auto prob_factory = [&]() {
            return ScheduleProblem::create(cost, W, S, REQ);
        };
        for (int i = 0; i < num_configs; i++) {
            run_experiment_recreate("Schedule5x6", configs[i].name,
                prob_factory, configs[i].cfg, 0.0f);
        }
    }

    // --- Part E: JSP (Permutation multiset encoding) ---
    {
        fprintf(stderr, "  [exp5] JSP4x3\n");
        const int J = 4, M = 3;
        int machine[J * M] = {0,1,2, 1,2,0, 2,0,1, 0,2,1};
        float duration[J * M] = {3,2,4, 4,3,2, 2,4,3, 3,2,5};
        auto prob_factory = [&]() {
            return JSPPermProblem::create(machine, duration, J, M, M);
        };
        for (int i = 0; i < num_configs; i++) {
            run_experiment_recreate("JSP4x3_Perm", configs[i].name,
                prob_factory, configs[i].cfg, 0.0f);
        }
    }
}

// ============================================================
//  Exp6: 时间-质量曲线（细粒度时间预算）
//  与 exp2 类似但时间点更密，专注 2 个代表性实例
// ============================================================
static void exp6_time_quality() {
    fprintf(stderr, "[exp6] Time-quality curves — kroA100 + ch150\n");

    struct TQInstance {
        const char* name;
        const float (*coords)[2];
        int n;
        float optimal;
    };

    TQInstance instances[] = {
        {"kroA100", kroA100_coords, KROA100_N, 21282.0f},
        {"ch150",   CH150_coords,   CH150_N,   6528.0f},
    };

    float time_budgets[] = {0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 30.0f, 60.0f};

    for (auto& inst : instances) {
        fprintf(stderr, "  [exp6] %s (n=%d)\n", inst.name, inst.n);

        float* dist = new float[inst.n * inst.n];
        compute_euc2d_dist(dist, inst.coords, inst.n);

        for (float t : time_budgets) {
            char cfg_name[64];
            snprintf(cfg_name, sizeof(cfg_name), "tq_t%.1fs", t);

            SolverConfig c = make_timed_config(t);

            run_experiment_recreate(inst.name, cfg_name,
                [&]() { return TSPLargeProblem::create(dist, inst.n); },
                c, inst.optimal);
        }

        delete[] dist;
    }
}

// ============================================================
// main — 命令行选择实验组
// ============================================================
int main(int argc, char** argv) {
    const char* target = (argc > 1) ? argv[1] : "all";

    {
        int device;
        cudaDeviceProp prop;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        fprintf(stderr, "GPU: %s (SM=%d, Shared=%zuKB, Compute=%d.%d)\n",
                prop.name, prop.multiProcessorCount,
                prop.sharedMemPerBlock / 1024, prop.major, prop.minor);
    }
    fprintf(stderr, "Warming up GPU...\n");
    warmup();
    fprintf(stderr, "Warmup done.\n\n");

    print_csv_header();

    bool run_all = (strcmp(target, "all") == 0);

    if (run_all || strcmp(target, "exp1") == 0) exp1_generality();
    if (run_all || strcmp(target, "exp2") == 0) exp2_tsp_quality();
    if (run_all || strcmp(target, "exp3") == 0) exp3_vrp_quality();
    if (run_all || strcmp(target, "exp4") == 0) exp4_scalability();
    if (run_all || strcmp(target, "exp5") == 0) exp5_ablation();
    if (run_all || strcmp(target, "exp6") == 0) exp6_time_quality();
    fprintf(stderr, "\nAll requested experiments completed.\n");
    return 0;
}
