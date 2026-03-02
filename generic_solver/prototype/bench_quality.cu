/**
 * bench_quality.cu - 标准实例解质量验证
 * 
 * 使用 TSPLIB / CVRPLIB 标准实例，对比已知最优解
 * 
 * 实例列表：
 *   1. TSP eil51    (51城市,  最优=426)
 *   2. TSP eil76    (76城市,  最优=538)
 *   3. TSP kroA100  (100城市, 最优=21282)
 *   4. CVRP A-n32-k5 (31客户5车, 容量=100, 最优=784)
 *
 * 每个实例跑多个配置，5 seeds 取平均
 * 直接用 solve() 接口 + 捕获结果
 */

#include "solver.cuh"
#include "tsp.cuh"
#include "tsp_large.cuh"
#include "vrp.cuh"
#include <cmath>
#include <cstdio>

// ============================================================
// TSPLIB 距离计算：EUC_2D（四舍五入到整数）
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
// 1. TSP eil51 — 51 城市, 最优=426
// ============================================================
const int EIL51_N = 51;
const float eil51_coords[EIL51_N][2] = {
    {37,52},{49,49},{52,64},{20,26},{40,30},{21,47},{17,63},{31,62},{52,33},
    {51,21},{42,41},{31,32},{ 5,25},{12,42},{36,16},{52,41},{27,23},{17,33},
    {13,13},{57,58},{62,42},{42,57},{16,57},{ 8,52},{ 7,38},{27,68},{30,48},
    {43,67},{58,48},{58,27},{37,69},{38,46},{46,10},{61,33},{62,63},{63,69},
    {32,22},{45,35},{59,15},{ 5, 6},{10,17},{21,10},{ 5,64},{30,15},{39,10},
    {32,39},{25,32},{25,55},{48,28},{56,37},{30,40}
};

// ============================================================
// 2. TSP eil76 — 76 城市, 最优=538
// ============================================================
const int EIL76_N = 76;
const float eil76_coords[EIL76_N][2] = {
    {22,22},{36,26},{21,45},{45,35},{55,20},{33,34},{50,50},{55,45},{26,59},{40,66},
    {55,65},{35,51},{62,35},{62,57},{62,24},{21,36},{33,44},{ 9,56},{62,48},{66,14},
    {44,13},{26,13},{11,28},{ 7,43},{17,64},{41,46},{55,34},{35,16},{52,26},{43,26},
    {31,76},{22,53},{26,29},{50,40},{55,50},{54,10},{60,15},{47,66},{30,60},{30,50},
    {12,17},{15,14},{16,19},{21,48},{50,30},{51,42},{50,15},{48,21},{12,38},{15,56},
    {29,39},{54,38},{55,57},{67,41},{10,70},{ 6,25},{65,27},{40,60},{70,64},{64, 4},
    {36, 6},{30,20},{20,30},{15, 5},{50,70},{57,72},{45,42},{38,33},{50, 4},{66, 8},
    {59, 5},{35,60},{27,24},{40,20},{40,37},{40,40}
};

// ============================================================
// 3. TSP kroA100 — 100 城市, 最优=21282
// ============================================================
const int KROA100_N = 100;
const float kroA100_coords[KROA100_N][2] = {
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

// ============================================================
// 4. CVRP A-n32-k5 — 31客户, 5车, 容量=100, 最优=784
// ============================================================
const int AN32K5_N = 31;
const int AN32K5_NODES = 32;
const float an32k5_coords[AN32K5_NODES][2] = {
    {82,76},
    {96,44},{50,5},{49,8},{13,7},{29,89},{58,30},{84,39},{14,24},{2,39},
    {3,82},{5,10},{98,52},{84,25},{61,59},{1,65},{88,51},{91,2},{19,32},
    {93,3},{50,93},{98,14},{5,42},{42,9},{61,62},{9,97},{80,55},{57,69},
    {23,15},{20,70},{85,60},{98,5}
};
const float an32k5_demands[AN32K5_N] = {
    19,21,6,19,7,12,16,6,16,8,14,21,16,3,22,18,19,1,24,8,12,4,8,24,24,2,20,15,2,14,9
};

// ============================================================
// 通用求解 + 取结果 — 利用 solve() 但需要获取最终 obj
// 方法：用 solve() 求解，然后从 solve 内部 find_best 获取结果
// 由于 solve() 会自己打印 Result，我们解析它的输出不方便
// 更好的方式：直接复制 solve() 的核心逻辑来获取返回值
// 
// 简化方案：在 solve 前后标记，solve 已会打印 "Result: obj0=xxx"
// 我们直接用 solve() 输出，然后手动再 find_best 一次
// ============================================================

struct BenchResult {
    float best;
    float avg;
    float worst;
    float avg_ms;
    int num_runs;
};

// 单次求解并返回最优目标值和耗时
template<typename Problem>
float single_solve(Problem& prob, const SolverConfig& cfg, float& elapsed_ms) {
    using Sol = typename Problem::Sol;
    ProblemConfig pcfg = prob.config();
    ObjConfig oc = make_obj_config(pcfg);
    
    const int block_threads = BLOCK_LEVEL_THREADS;
    bool use_islands = cfg.num_islands > 1;
    bool use_sa = cfg.sa_temp_init > 0.0f;
    bool use_crossover = cfg.crossover_rate > 0.0f;
    int island_size = use_islands ? cfg.pop_size / cfg.num_islands : cfg.pop_size;
    
    // 交叉栈需求
    if (use_crossover) {
        size_t ox_arrays = Sol::DIM1 * Sol::DIM2 * sizeof(bool) + 512 * sizeof(bool) + 512 * sizeof(int);
        size_t need = sizeof(Sol) + ox_arrays + 512;
        if (need > 1024) cudaDeviceSetLimit(cudaLimitStackSize, need);
    }
    
    Population<Sol> pop;
    pop.allocate(cfg.pop_size, block_threads);
    pop.init_rng(cfg.seed, 256);
    pop.init_population(pcfg, 256);
    
    SeqRegistry seq_reg = build_seq_registry(pcfg.encoding, pcfg.dim1, pcfg.cross_row_prob);
    KStepConfig kstep = build_kstep_config();
    int* d_best_idx;
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));
    
    EvolveParams h_params;
    EvolveParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(EvolveParams)));
    
    Sol* d_global_best = nullptr;
    if (use_sa) {
        CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(Sol)));
    }
    
    // AOS
    bool use_aos = cfg.use_aos;
    AOSStats* d_aos_stats = nullptr;
    AOSStats* h_aos_stats = nullptr;
    if (use_aos) {
        CUDA_CHECK(cudaMalloc(&d_aos_stats, sizeof(AOSStats) * cfg.pop_size));
        h_aos_stats = new AOSStats[cfg.pop_size];
    }
    
    // Shared memory
    size_t prob_smem = prob.shared_mem_bytes();
    size_t total_smem = sizeof(Sol) + prob_smem + sizeof(MultiStepCandidate) * block_threads;
    if (use_aos) total_smem += sizeof(AOSStats);
    
    {
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (total_smem > (size_t)prop.sharedMemPerBlock) {
            prob_smem = 0;
            total_smem = sizeof(Sol) + sizeof(MultiStepCandidate) * block_threads;
            if (use_aos) total_smem += sizeof(AOSStats);
        }
        if (total_smem > 48 * 1024) {
            cudaFuncSetAttribute(
                evolve_block_kernel<Problem, Sol>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                total_smem);
            if (use_crossover) {
                cudaFuncSetAttribute(
                    crossover_block_kernel<Problem, Sol>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    total_smem);
            }
        }
    }
    
    // 初始评估
    {
        size_t eval_smem = prob.shared_mem_bytes();
        int eval_grid = calc_grid_size(cfg.pop_size, block_threads);
        evaluate_kernel<<<eval_grid, block_threads, eval_smem>>>(
            prob, pop.d_solutions, cfg.pop_size, eval_smem);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if (use_sa) {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_global_best, pop.d_solutions + idx, sizeof(Sol), cudaMemcpyDeviceToDevice));
    }
    
    // 主循环
    int batch = use_islands ? cfg.migrate_interval : cfg.max_gen;
    int gen_done = 0, migrate_round = 0;
    
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    
    while (gen_done < cfg.max_gen) {
        int gens = batch;
        if (gen_done + gens > cfg.max_gen) gens = cfg.max_gen - gen_done;
        float temp = use_sa ? cfg.sa_temp_init * powf(cfg.sa_alpha, (float)gen_done) : 0.0f;
        
        h_params.temp_start = temp;
        h_params.gens_per_batch = gens;
        h_params.seq_reg = seq_reg;
        h_params.kstep = kstep;
        h_params.migrate_round = migrate_round;
        CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice));
        
        evolve_block_kernel<<<cfg.pop_size, block_threads, total_smem>>>(
            prob, pop.d_solutions, cfg.pop_size,
            pcfg.encoding, pcfg.dim1,
            oc, pop.d_rng_states,
            cfg.sa_alpha, prob_smem,
            d_aos_stats, nullptr, nullptr, 0, 0, 0, d_params);
        gen_done += gens;
        
        // AOS 更新（与 solver.cuh 一致：按 seq_reg.count 聚合，更新 seq_reg.weights）
        if (use_aos) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_aos_stats, d_aos_stats,
                                  sizeof(AOSStats) * cfg.pop_size, cudaMemcpyDeviceToHost));
            int total_usage[MAX_SEQ] = {};
            int total_improve[MAX_SEQ] = {};
            int total_k_usage[MAX_K] = {};
            int total_k_improve[MAX_K] = {};
            for (int b = 0; b < cfg.pop_size; b++) {
                for (int i = 0; i < seq_reg.count; i++) {
                    total_usage[i] += h_aos_stats[b].usage[i];
                    total_improve[i] += h_aos_stats[b].improvement[i];
                }
                for (int i = 0; i < MAX_K; i++) {
                    total_k_usage[i] += h_aos_stats[b].k_usage[i];
                    total_k_improve[i] += h_aos_stats[b].k_improvement[i];
                }
            }
            constexpr float AOS_ALPHA = 0.8f;
            constexpr float AOS_EPSILON = 0.02f;
            float new_weights[MAX_SEQ];
            float sum = 0.0f;
            for (int i = 0; i < seq_reg.count; i++) {
                float rate = (total_usage[i] > 0)
                    ? (float)total_improve[i] / (float)total_usage[i]
                    : 0.0f;
                new_weights[i] = AOS_ALPHA * seq_reg.weights[i]
                               + (1.0f - AOS_ALPHA) * (rate + AOS_EPSILON);
                sum += new_weights[i];
            }
            if (sum > 0.0f) {
                for (int i = 0; i < seq_reg.count; i++) {
                    seq_reg.weights[i] = new_weights[i] / sum;
                }
            }
            // K-step 权重更新（与 solver.cuh 一致）
            {
                float new_w[MAX_K];
                float sum_k = 0.0f;
                for (int i = 0; i < MAX_K; i++) {
                    float rate = (total_k_usage[i] > 0)
                        ? (float)total_k_improve[i] / (float)total_k_usage[i]
                        : 0.0f;
                    new_w[i] = AOS_ALPHA * kstep.weights[i]
                             + (1.0f - AOS_ALPHA) * (rate + AOS_EPSILON);
                    sum_k += new_w[i];
                }
                if (sum_k > 0.0f) {
                    for (int i = 0; i < MAX_K; i++)
                        kstep.weights[i] = new_w[i] / sum_k;
                }
                // 停滞检测
                int total_improve_all = 0;
                for (int i = 0; i < seq_reg.count; i++) total_improve_all += total_improve[i];
                if (total_improve_all == 0) {
                    kstep.stagnation_count++;
                } else {
                    kstep.stagnation_count = 0;
                }
                if (kstep.stagnation_count >= kstep.stagnation_limit) {
                    kstep.weights[0] = 0.50f;
                    kstep.weights[1] = 0.30f;
                    kstep.weights[2] = 0.20f;
                    kstep.stagnation_count = 0;
                }
            }
        }
        
        if (use_crossover) {
            crossover_block_kernel<<<cfg.pop_size, block_threads, total_smem>>>(
                prob, pop.d_solutions, cfg.pop_size,
                pcfg.encoding, pcfg.dim1, oc, pop.d_rng_states,
                cfg.crossover_rate, prob_smem,
                pcfg.row_mode == RowMode::Partition ? pcfg.total_elements : pcfg.dim2_default);
        }
        if (use_islands) {
            migrate_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size,
                                      island_size, oc, cfg.migrate_strategy, d_params);
            migrate_round++;
        }
        if (use_sa) {
            elite_inject_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size,
                                           d_global_best, oc);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));
    
    // 获取最优解
    float best_val;
    if (use_sa) {
        Sol best;
        CUDA_CHECK(cudaMemcpy(&best, d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost));
        best_val = best.objectives[0];
        // SA 模式下也检查种群中是否有更优解
        find_best_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        Sol pop_best = pop.download_solution(idx);
        if (pop_best.objectives[0] < best_val && pop_best.penalty <= 0.0f)
            best_val = pop_best.objectives[0];
    } else {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        Sol best = pop.download_solution(idx);
        best_val = best.objectives[0];
        if (best.penalty > 0.0f) best_val = 1e30f;
    }
    
    // 清理
    CUDA_CHECK(cudaFree(d_best_idx));
    if (d_params) CUDA_CHECK(cudaFree(d_params));
    if (d_global_best) CUDA_CHECK(cudaFree(d_global_best));
    if (d_aos_stats) CUDA_CHECK(cudaFree(d_aos_stats));
    if (h_aos_stats) delete[] h_aos_stats;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    
    return best_val;
}

// 多次求解取统计
template<typename Problem>
BenchResult multi_solve(Problem& prob_template,
                         const float* h_data, int data_size,
                         const SolverConfig& cfg,
                         int num_seeds, const unsigned* seeds,
                         // VRP 专用参数
                         const float* h_demand = nullptr,
                         int n_customers = 0, float capacity = 0,
                         int num_vehicles = 0, int max_vehicles = 0) {
    // 此函数不使用 prob_template，仅用于类型推导
    (void)prob_template;
    (void)h_data; (void)data_size;
    (void)h_demand; (void)n_customers; (void)capacity;
    (void)num_vehicles; (void)max_vehicles;
    
    BenchResult r = {1e30f, 0, 0, 0, num_seeds};
    // 实际的创建在外面做
    return r;
}

void print_row(const char* name, float optimal, BenchResult r) {
    float gap_best = (r.best - optimal) / optimal * 100.0f;
    float gap_avg = (r.avg - optimal) / optimal * 100.0f;
    printf("  %-32s %7.0f %7.0f %7.0f  %5.1f%%  %5.1f%%  %6.0fms\n",
           name, r.best, r.avg, r.worst, gap_best, gap_avg, r.avg_ms);
    fflush(stdout);
}

// ============================================================
// 主程序
// ============================================================
int main() {
    print_device_info();
    fflush(stdout);
    
    const unsigned seeds[] = {42, 123, 456, 789, 2024};
    const int NS = 5;
    
    // 预热
    {
        float dd[25] = {};
        for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) dd[i*5+j] = (i==j)?0:10;
        auto p = TSPProblem::create(dd, 5);
        SolverConfig c; c.pop_size=64; c.max_gen=10; c.seed=1; c.verbose=false;
        solve(p, c);
        p.destroy();
        printf("  (warmup done)\n");
        fflush(stdout);
    }
    
    printf("\n");
    printf("================================================================\n");
    printf("  GenSolver v2.0 Block — 标准实例解质量验证\n");
    printf("  5 seeds, 128 threads/block\n");
    printf("================================================================\n");
    printf("  %-32s %7s %7s %7s  %5s  %5s  %6s\n",
           "Config", "Best", "Avg", "Worst", "GapB", "GapA", "Time");
    printf("----------------------------------------------------------------\n");
    fflush(stdout);
    
    // ============================================================
    // TSP eil51 (optimal=426)
    // ============================================================
    {
        float dist[EIL51_N * EIL51_N];
        compute_euc2d_dist(dist, eil51_coords, EIL51_N);
        const float OPT = 426.0f;
        printf("  --- TSP eil51 (n=%d, optimal=%.0f) ---\n", EIL51_N, OPT);
        fflush(stdout);
        
        // 配置 A: HC, pop=1024, gen=5000
        {
            SolverConfig c; c.pop_size=1024; c.max_gen=5000; c.verbose=false;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPProblem::create(dist, EIL51_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("eil51 HC p1024 g5k", OPT, r);
        }
        // 配置 B: SA T=20, pop=1024, gen=5000
        {
            SolverConfig c; c.pop_size=1024; c.max_gen=5000; c.verbose=false;
            c.sa_temp_init=20; c.sa_alpha=0.998f;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPProblem::create(dist, EIL51_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("eil51 SA T=20 p1024 g5k", OPT, r);
        }
        // 配置 C: SA + Islands, pop=4096, gen=5000
        {
            SolverConfig c; c.pop_size=4096; c.max_gen=5000; c.verbose=false;
            c.sa_temp_init=20; c.sa_alpha=0.998f;
            c.num_islands=16; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPProblem::create(dist, EIL51_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("eil51 SA+Isl16 p4096 g5k", OPT, r);
        }
        // 配置 D: SA + Islands, pop=4096, gen=10000
        {
            SolverConfig c; c.pop_size=4096; c.max_gen=10000; c.verbose=false;
            c.sa_temp_init=20; c.sa_alpha=0.999f;
            c.num_islands=16; c.migrate_interval=100; c.migrate_strategy=MigrateStrategy::Hybrid;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPProblem::create(dist, EIL51_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("eil51 SA+Isl16 p4096 g10k", OPT, r);
        }
    }
    
    printf("----------------------------------------------------------------\n");
    fflush(stdout);
    
    // ============================================================
    // TSP eil76 (optimal=538)
    // ============================================================
    {
        float dist[EIL76_N * EIL76_N];
        compute_euc2d_dist(dist, eil76_coords, EIL76_N);
        const float OPT = 538.0f;
        printf("  --- TSP eil76 (n=%d, optimal=%.0f) ---\n", EIL76_N, OPT);
        fflush(stdout);
        
        // 配置 A: SA + Islands, pop=2048, gen=5000
        {
            SolverConfig c; c.pop_size=2048; c.max_gen=5000; c.verbose=false;
            c.sa_temp_init=20; c.sa_alpha=0.999f;
            c.num_islands=8; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPLargeProblem::create(dist, EIL76_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("eil76 SA+Isl8 p2048 g5k", OPT, r);
        }
        // 配置 B: SA + Islands, pop=4096, gen=10000
        {
            SolverConfig c; c.pop_size=4096; c.max_gen=10000; c.verbose=false;
            c.sa_temp_init=20; c.sa_alpha=0.999f;
            c.num_islands=16; c.migrate_interval=100; c.migrate_strategy=MigrateStrategy::Hybrid;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPLargeProblem::create(dist, EIL76_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("eil76 SA+Isl16 p4096 g10k", OPT, r);
        }
    }
    
    printf("----------------------------------------------------------------\n");
    fflush(stdout);
    
    // ============================================================
    // TSP kroA100 (optimal=21282)
    // ============================================================
    {
        float dist[KROA100_N * KROA100_N];
        compute_euc2d_dist(dist, kroA100_coords, KROA100_N);
        const float OPT = 21282.0f;
        printf("  --- TSP kroA100 (n=%d, optimal=%.0f) ---\n", KROA100_N, OPT);
        fflush(stdout);
        
        // 配置 A: SA + Islands, pop=2048, gen=5000
        {
            SolverConfig c; c.pop_size=2048; c.max_gen=5000; c.verbose=false;
            c.sa_temp_init=50; c.sa_alpha=0.999f;
            c.num_islands=8; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPLargeProblem::create(dist, KROA100_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("kroA100 SA+Isl8 p2048 g5k", OPT, r);
        }
        // 配置 B: SA + Islands, pop=4096, gen=10000
        {
            SolverConfig c; c.pop_size=4096; c.max_gen=10000; c.verbose=false;
            c.sa_temp_init=100; c.sa_alpha=0.999f;
            c.num_islands=16; c.migrate_interval=100; c.migrate_strategy=MigrateStrategy::Hybrid;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = TSPLargeProblem::create(dist, KROA100_N);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("kroA100 SA+Isl16 p4096 g10k", OPT, r);
        }
    }
    
    printf("----------------------------------------------------------------\n");
    fflush(stdout);
    
    // ============================================================
    // CVRP A-n32-k5 (optimal=784)
    // ============================================================
    {
        float dist[AN32K5_NODES * AN32K5_NODES];
        compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);
        const float OPT = 784.0f;
        printf("  --- CVRP A-n32-k5 (n=%d, k=5, cap=100, optimal=%.0f) ---\n", AN32K5_N, OPT);
        fflush(stdout);
        
        // 配置 A: SA + Islands + AOS, pop=512, gen=5000
        {
            SolverConfig c; c.pop_size=512; c.max_gen=5000; c.verbose=false;
            c.sa_temp_init=20; c.sa_alpha=0.999f;
            c.num_islands=8; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
            c.use_aos=true;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("A-n32-k5 SA+AOS p512 g5k", OPT, r);
        }
        // 配置 B: SA + Islands + AOS, pop=1024, gen=10000
        {
            SolverConfig c; c.pop_size=1024; c.max_gen=10000; c.verbose=false;
            c.sa_temp_init=30; c.sa_alpha=0.999f;
            c.num_islands=16; c.migrate_interval=100; c.migrate_strategy=MigrateStrategy::Hybrid;
            c.use_aos=true;
            float sum=0, mn=1e30f, mx=0, tms=0;
            for (int s = 0; s < NS; s++) {
                c.seed = seeds[s];
                auto prob = VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5);
                float ms; float v = single_solve(prob, c, ms);
                sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
                prob.destroy();
            }
            BenchResult r = {mn, sum/NS, mx, tms/NS, NS};
            print_row("A-n32-k5 SA+AOS p1024 g10k", OPT, r);
        }
    }
    
    printf("================================================================\n");
    printf("\n  GapB = (Best - Optimal) / Optimal\n");
    printf("  GapA = (Avg  - Optimal) / Optimal\n");
    printf("  Seeds: {42, 123, 456, 789, 2024}\n\n");
    fflush(stdout);
    
    return 0;
}
