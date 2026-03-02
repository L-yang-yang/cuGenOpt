/**
 * bench_auto_pop.cu - 自动种群大小 vs 手动种群大小对比
 * 
 * 选择 ch150 (150 城市, 最优=6528) 作为对比实例：
 *   - 规模适中，之前最好成绩 GapB=0.5% GapA=1.0% (p64 g3000)
 *   - 对比 auto pop vs p64 vs p128，固定 g3000 + SA + Isl4
 *   - 5 seeds 取统计
 *
 * 同时测试 tsp225 (225 城市, 最优=3916) 验证更大规模的效果
 */

#include "solver.cuh"
#include "tsp_large.cuh"
#include "tsplib_data.h"
#include <cmath>
#include <cstdio>

// ============================================================
// 使用 solve() 的包装：捕获最优值和耗时
// ============================================================
// solve() 内部会打印结果，我们需要一个能返回数值的版本
// 最简单的方式：用 single_solve 但支持 auto pop

template<typename Problem>
float single_solve_v2(Problem& prob, SolverConfig cfg, float& elapsed_ms) {
    using Sol = typename Problem::Sol;
    ProblemConfig pcfg = prob.config();
    ObjConfig oc = make_obj_config(pcfg);
    const int block_threads = BLOCK_LEVEL_THREADS;
    bool use_islands = cfg.num_islands > 1;
    bool use_sa = cfg.sa_temp_init > 0.0f;

    // --- 与 solve() 相同的 auto pop 逻辑 ---
    size_t prob_smem = prob.shared_mem_bytes();
    size_t total_smem = sizeof(Sol) + prob_smem + sizeof(MultiStepCandidate) * block_threads;

    cudaDeviceProp prop; int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (total_smem > (size_t)prop.sharedMemPerBlock) {
        prob_smem = 0;
        total_smem = sizeof(Sol) + sizeof(MultiStepCandidate) * block_threads;
    }
    if (total_smem > 48*1024)
        cudaFuncSetAttribute(evolve_block_kernel<Problem, Sol>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, total_smem);

    int pop_size = cfg.pop_size;
    if (pop_size <= 0) {
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            evolve_block_kernel<Problem, Sol>,
            block_threads, total_smem);
        int full_capacity = max_blocks_per_sm * prop.multiProcessorCount;
        if (prob_smem > 0) {
            pop_size = full_capacity;
        } else {
            pop_size = full_capacity / 4;
            if (pop_size < prop.multiProcessorCount)
                pop_size = prop.multiProcessorCount;
        }
        if (use_islands && cfg.num_islands > 1)
            pop_size = (pop_size / cfg.num_islands) * cfg.num_islands;
        if (pop_size < 4) pop_size = 4;
    }

    int island_size = use_islands ? pop_size / cfg.num_islands : pop_size;

    Population<Sol> pop;
    pop.allocate(pop_size, block_threads);
    cudaGetLastError(); cudaDeviceSynchronize();
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
    if (use_sa) CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(Sol)));

    { size_t es = prob.shared_mem_bytes();
      int eg = calc_grid_size(pop_size, block_threads);
      evaluate_kernel<<<eg, block_threads, es>>>(prob, pop.d_solutions, pop_size, es); }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (use_sa) {
        find_best_kernel<<<1,1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_global_best, pop.d_solutions+idx, sizeof(Sol), cudaMemcpyDeviceToDevice));
    }

    int batch = use_islands ? cfg.migrate_interval : cfg.max_gen;
    int gen_done = 0, migrate_round = 0;
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
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
        
        evolve_block_kernel<<<pop_size, block_threads, total_smem>>>(
            prob, pop.d_solutions, pop_size,
            pcfg.encoding, pcfg.dim1, oc, pop.d_rng_states,
            cfg.sa_alpha, prob_smem,
            nullptr, nullptr, nullptr, 0, 0, 0, d_params);
        gen_done += gens;
        if (use_islands) {
            migrate_kernel<<<1,1>>>(pop.d_solutions, pop_size,
                island_size, oc, cfg.migrate_strategy, d_params);
            migrate_round++;
        }
        if (use_sa)
            elite_inject_kernel<<<1,1>>>(pop.d_solutions, pop_size, d_global_best, oc);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));

    float best_val;
    if (use_sa) {
        Sol best;
        CUDA_CHECK(cudaMemcpy(&best, d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost));
        best_val = best.objectives[0];
        find_best_kernel<<<1,1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        Sol pb = pop.download_solution(idx);
        if (pb.objectives[0] < best_val && pb.penalty <= 0.0f) best_val = pb.objectives[0];
    } else {
        find_best_kernel<<<1,1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        Sol best = pop.download_solution(idx);
        best_val = best.objectives[0];
    }

    CUDA_CHECK(cudaFree(d_best_idx));
    if (d_params) CUDA_CHECK(cudaFree(d_params));
    if (d_global_best) CUDA_CHECK(cudaFree(d_global_best));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(pop.d_solutions)); CUDA_CHECK(cudaFree(pop.d_rng_states));
    return best_val;
}

// ============================================================
// 辅助
// ============================================================
void compute_dist(const float coords[][2], int n, float* dist) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0]-coords[j][0], dy = coords[i][1]-coords[j][1];
            dist[i*n+j] = roundf(sqrtf(dx*dx+dy*dy));
        }
}

struct TestConfig {
    const char* name;
    int pop_size;       // 0 = auto
    int max_gen;
    float sa_temp;
    float sa_alpha;
    int num_islands;
    int migrate_interval;
};

template<typename Problem>
void run_comparison(const char* inst_name, float opt, const float coords[][2], int n,
                    TestConfig* configs, int nconfigs,
                    int ns, const unsigned* seeds) {
    float* dist = new float[n * n];
    compute_dist(coords, n, dist);
    
    printf("================================================================\n");
    printf("  %s (n=%d, optimal=%.0f)\n", inst_name, n, opt);
    printf("  %d seeds, auto pop vs manual pop\n", ns);
    printf("================================================================\n");
    printf("  %-44s %6s %8s %8s %8s  %5s  %5s  %7s\n",
           "Config", "Pop", "Best", "Avg", "Worst", "GapB", "GapA", "Time");
    printf("----------------------------------------------------------------\n");
    fflush(stdout);
    
    for (int ci = 0; ci < nconfigs; ci++) {
        auto& tc = configs[ci];
        float sum = 0, mn = 1e30f, mx = 0, tms = 0;
        int actual_pop = 0;
        
        for (int s = 0; s < ns; s++) {
            SolverConfig cfg;
            cfg.pop_size = tc.pop_size;  // 0 = auto
            cfg.max_gen = tc.max_gen;
            cfg.verbose = false;
            cfg.seed = seeds[s];
            if (tc.sa_temp > 0) {
                cfg.sa_temp_init = tc.sa_temp;
                cfg.sa_alpha = tc.sa_alpha;
            }
            if (tc.num_islands > 1) {
                cfg.num_islands = tc.num_islands;
                cfg.migrate_interval = tc.migrate_interval;
                cfg.migrate_strategy = MigrateStrategy::Hybrid;
            }
            
            // 第一次运行时记录 actual pop_size（auto 模式下需要知道实际值）
            if (s == 0 && tc.pop_size <= 0) {
                // 快速查询 auto pop_size（与 solver.cuh 逻辑一致）
                using Sol = typename Problem::Sol;
                auto tmp = Problem::create(dist, n);
                size_t ps = tmp.shared_mem_bytes();
                size_t ts = sizeof(Sol) + ps + sizeof(MultiStepCandidate) * BLOCK_LEVEL_THREADS;
                cudaDeviceProp prop; int dev;
                cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
                if (ts > (size_t)prop.sharedMemPerBlock) {
                    ps = 0;
                    ts = sizeof(Sol) + sizeof(MultiStepCandidate) * BLOCK_LEVEL_THREADS;
                }
                int mbps = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &mbps, evolve_block_kernel<Problem, Sol>,
                    BLOCK_LEVEL_THREADS, ts);
                int full_cap = mbps * prop.multiProcessorCount;
                if (ps > 0) {
                    actual_pop = full_cap;
                } else {
                    actual_pop = full_cap / 4;
                    if (actual_pop < prop.multiProcessorCount)
                        actual_pop = prop.multiProcessorCount;
                }
                if (tc.num_islands > 1)
                    actual_pop = (actual_pop / tc.num_islands) * tc.num_islands;
                tmp.destroy();
            }
            
            auto prob = Problem::create(dist, n);
            float ms;
            float v = single_solve_v2(prob, cfg, ms);
            sum += v; if (v < mn) mn = v; if (v > mx) mx = v; tms += ms;
            prob.destroy();
        }
        
        int show_pop = (tc.pop_size > 0) ? tc.pop_size : actual_pop;
        long long total_evals = (long long)show_pop * tc.max_gen * 128;
        float gb = (mn - opt) / opt * 100;
        float ga = (sum / ns - opt) / opt * 100;
        char label[128];
        snprintf(label, sizeof(label), "%s (~%.1fM evals)", tc.name, total_evals / 1e6);
        printf("  %-44s %6d %8.0f %8.0f %8.0f  %4.1f%%  %4.1f%%  %6.0fms\n",
               label, show_pop, mn, sum/ns, mx, gb, ga, tms/ns);
        fflush(stdout);
    }
    
    printf("================================================================\n\n");
    fflush(stdout);
    delete[] dist;
}

// ============================================================
// main
// ============================================================
int main() {
    print_device_info(); fflush(stdout);
    
    const unsigned seeds[] = {42, 123, 456, 789, 2024};
    const int NS = 5;
    
    printf("\n");
    printf("########################################################\n");
    printf("#  Auto Pop Size v2 对比实验                            #\n");
    printf("#  v1: 打满 SM (100%%)  vs  v2: 1/4 容量 (gmem 场景)   #\n");
    printf("#  固定策略: SA T=50 + Isl4 + g3000                    #\n");
    printf("########################################################\n\n");
    fflush(stdout);
    
    { void* tmp; cudaMalloc(&tmp, 1024); cudaFree(tmp); }
    printf("  (warmup done)\n\n"); fflush(stdout);
    
    // ---- ch150 ----
    {
        // auto v2 = 320/4 = 80 → 对齐 Isl4 → 80
        // 等量预算: 80*3000*128=30.7M, p64 等量 → g3750
        TestConfig configs[] = {
            {"p64 SA Isl4 g3000",           64, 3000, 50, 0.998f, 4, 200},
            {"AUTO-v2 SA Isl4 g3000",        0, 3000, 50, 0.998f, 4, 200},
            {"p128 SA Isl4 g3000",         128, 3000, 50, 0.998f, 4, 200},
            // 等量预算对比: auto-v2(80) g3000 ≈ 30.7M vs p64 g3750 ≈ 30.7M
            {"p64 SA Isl4 g3750 (=budget)",  64, 3750, 50, 0.998f, 4, 200},
        };
        run_comparison<TSPLargeProblem>("ch150", 6528.0f, CH150_coords, CH150_N,
                                        configs, sizeof(configs)/sizeof(configs[0]), NS, seeds);
    }
    
    // ---- tsp225 ----
    {
        TestConfig configs[] = {
            {"p64 SA Isl4 g3000",           64, 3000, 50, 0.998f, 4, 200},
            {"AUTO-v2 SA Isl4 g3000",        0, 3000, 50, 0.998f, 4, 200},
            {"p128 SA Isl4 g3000",         128, 3000, 50, 0.998f, 4, 200},
            {"p64 SA Isl4 g3750 (=budget)",  64, 3750, 50, 0.998f, 4, 200},
        };
        run_comparison<TSPLargeProblem>("tsp225", 3916.0f, TSP225_coords, TSP225_N,
                                        configs, sizeof(configs)/sizeof(configs[0]), NS, seeds);
    }
    
    printf("########################################################\n");
    printf("#  完成！                                               #\n");
    printf("########################################################\n");
    fflush(stdout);
    return 0;
}
