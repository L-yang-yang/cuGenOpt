/**
 * bench_large_scale.cu - 大规模 TSP 实例验证
 * 
 * 实例：ch150 (n=150, opt=6528), tsp225 (n=225, opt=3916),
 *       lin318 (n=318, opt=42029), pcb442 (n=442, opt=50778)
 *
 * 设计原则（基于此前 kroA100 的经验）：
 *   - "少解多代" 远优于 "多解少代"
 *   - 大规模问题搜索空间指数增长，需要更多代数
 *   - 不给太多预算，让差距暴露
 *   - 距离矩阵 > 48KB 时走 global memory（ch150: 90KB, tsp225: 202KB, lin318: 404KB, pcb442: 781KB）
 *
 * 每代搜索量 = pop_size × 128 (block threads)
 */

#include "solver.cuh"
#include "tsp_large.cuh"
#include "tsp_xlarge.cuh"
#include "tsplib_data.h"
#include <cmath>
#include <cstdio>

// ============================================================
// single_solve: 单次求解，返回最优值和耗时
// ============================================================
template<typename Problem>
float single_solve(Problem& prob, const SolverConfig& cfg, float& elapsed_ms) {
    using Sol = typename Problem::Sol;
    ProblemConfig pcfg = prob.config();
    ObjConfig oc = make_obj_config(pcfg);
    const int block_threads = BLOCK_LEVEL_THREADS;
    bool use_islands = cfg.num_islands > 1;
    bool use_sa = cfg.sa_temp_init > 0.0f;
    int island_size = use_islands ? cfg.pop_size / cfg.num_islands : cfg.pop_size;

    Population<Sol> pop;
    pop.allocate(cfg.pop_size, block_threads);
    // 清除可能的残留 CUDA 错误（evolve_block_kernel 在距离矩阵走 global memory 时可能产生）
    cudaGetLastError();
    cudaDeviceSynchronize();
    pop.init_rng(cfg.seed, 256);
    pop.init_population(pcfg, 256);

    // v3.0: 使用 SeqRegistry 替代 d_op_weights
    SeqRegistry seq_reg = build_seq_registry(pcfg.encoding, pcfg.dim1, pcfg.cross_row_prob);
    KStepConfig kstep = build_kstep_config();
    int* d_best_idx;
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));
    
    EvolveParams h_params;
    EvolveParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(EvolveParams)));
    
    Sol* d_global_best = nullptr;
    if (use_sa) CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(Sol)));

    size_t prob_smem = prob.shared_mem_bytes();
    size_t total_smem = sizeof(Sol) + prob_smem + sizeof(MultiStepCandidate) * block_threads;
    {
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
    }

    { size_t es = prob.shared_mem_bytes();
      int eg = calc_grid_size(cfg.pop_size, block_threads);
      evaluate_kernel<<<eg, block_threads, es>>>(prob, pop.d_solutions, cfg.pop_size, es); }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (use_sa) {
        find_best_kernel<<<1,1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
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
        
        evolve_block_kernel<<<cfg.pop_size, block_threads, total_smem>>>(
            prob, pop.d_solutions, cfg.pop_size,
            pcfg.encoding, pcfg.dim1, oc, pop.d_rng_states,
            cfg.sa_alpha, prob_smem,
            nullptr, nullptr, nullptr, 0, 0, 0, d_params);
        gen_done += gens;
        if (use_islands) {
            migrate_kernel<<<1,1>>>(pop.d_solutions, cfg.pop_size,
                island_size, oc, cfg.migrate_strategy, d_params);
            migrate_round++;
        }
        if (use_sa)
            elite_inject_kernel<<<1,1>>>(pop.d_solutions, cfg.pop_size, d_global_best, oc);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));

    float best_val;
    if (use_sa) {
        Sol best;
        CUDA_CHECK(cudaMemcpy(&best, d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost));
        best_val = best.objectives[0];
        find_best_kernel<<<1,1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        Sol pb = pop.download_solution(idx);
        if (pb.objectives[0] < best_val && pb.penalty <= 0.0f) best_val = pb.objectives[0];
    } else {
        find_best_kernel<<<1,1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
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
// 辅助：计算距离矩阵
// ============================================================
void compute_dist(const float coords[][2], int n, float* dist) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0]-coords[j][0], dy = coords[i][1]-coords[j][1];
            dist[i*n+j] = roundf(sqrtf(dx*dx+dy*dy));
        }
}

// ============================================================
// 运行一个实例的多种配置
// ============================================================
template<typename Problem>
void run_instance(const char* inst_name, float opt, const float coords[][2], int n,
                  int ns, const unsigned* seeds) {
    // 计算距离矩阵（堆上分配，n 可能很大）
    float* dist = new float[n * n];
    compute_dist(coords, n, dist);
    
    printf("================================================================\n");
    printf("  %s (n=%d, optimal=%.0f)\n", inst_name, n, opt);
    printf("  %d seeds, 128 threads/block\n", ns);
    printf("================================================================\n");
    printf("  %-42s %8s %8s %8s  %6s  %6s  %7s\n",
           "Config", "Best", "Avg", "Worst", "GapB", "GapA", "Time");
    printf("----------------------------------------------------------------\n");
    fflush(stdout);
    
    // 根据问题规模设计预算梯度
    // 基准：n=100 时 p64 g200 = 1.6M evals 是"极低"
    // 大规模需要更多代数，但保持"少解多代"原则
    
    // 配置结构体数组
    struct TestConfig {
        const char* name;
        int pop_size;
        int max_gen;
        float sa_temp;
        float sa_alpha;
        int num_islands;
        int migrate_interval;
    };
    
    // 预算设计：
    // ch150 (n=150): 搜索空间约 100 的 ~1.5 倍 → 需要 2-3x 预算
    // tsp225 (n=225): 约 2-3 倍 → 需要 4-6x
    // lin318 (n=318): 约 3-4 倍 → 需要 8-12x
    // pcb442 (n=442): 约 4-5 倍 → 需要 16-25x
    //
    // 但我们要控制预算让差距暴露，所以给的比"足够"少
    
    TestConfig configs[] = {
        // 低预算 HC（基线）
        {"HC p64 g500",                64, 500,   0,    0,     1, 0},
        // 低预算 SA
        {"SA T=50 p64 g500",           64, 500,  50, 0.99f,   1, 0},
        // SA + Islands（少解多代，推荐配置）
        {"SA T=50 Isl4 p64 g1000",     64, 1000, 50, 0.995f,  4, 100},
        // 中等预算 SA + Islands
        {"SA T=50 Isl4 p128 g1000",   128, 1000, 50, 0.995f,  4, 100},
        // 较高预算
        {"SA T=50 Isl4 p64 g3000",     64, 3000, 50, 0.998f,  4, 200},
        // 高预算（看收敛极限）
        {"SA T=50 Isl4 p128 g5000",   128, 5000, 50, 0.999f,  4, 300},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);
    
    for (int ci = 0; ci < nconfigs; ci++) {
        auto& tc = configs[ci];
        float sum = 0, mn = 1e30f, mx = 0, tms = 0;
        long long total_evals = (long long)tc.pop_size * tc.max_gen * 128;
        
        for (int s = 0; s < ns; s++) {
            SolverConfig cfg;
            cfg.pop_size = tc.pop_size;
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
            
            auto prob = Problem::create(dist, n);
            float ms;
            float v = single_solve(prob, cfg, ms);
            sum += v; if (v < mn) mn = v; if (v > mx) mx = v; tms += ms;
            prob.destroy();
        }
        
        float gb = (mn - opt) / opt * 100;
        float ga = (sum / ns - opt) / opt * 100;
        char label[128];
        snprintf(label, sizeof(label), "%s (~%.1fM evals)", tc.name, total_evals / 1e6);
        printf("  %-42s %8.0f %8.0f %8.0f  %5.1f%%  %5.1f%%  %6.0fms\n",
               label, mn, sum/ns, mx, gb, ga, tms/ns);
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
    printf("#  大规模 TSP 实例验证 — GenSolver Block 级架构          #\n");
    printf("#  策略：少解多代 + SA + Islands                        #\n");
    printf("#  每代搜索量 = pop_size × 128                          #\n");
    printf("########################################################\n\n");
    fflush(stdout);
    
    // 预热（简单 cudaMalloc/cudaFree 触发 context 初始化）
    {
        void* tmp; cudaMalloc(&tmp, 1024); cudaFree(tmp);
    }
    printf("  (warmup done)\n\n"); fflush(stdout);
    
    // ---- ch150: n=150, optimal=6528 ----
    // D2=256 足够
    run_instance<TSPLargeProblem>("ch150", 6528.0f, CH150_coords, CH150_N, NS, seeds);
    
    // ---- tsp225: n=225, optimal=3916 ----
    // D2=256 足够
    run_instance<TSPLargeProblem>("tsp225", 3916.0f, TSP225_coords, TSP225_N, NS, seeds);
    
    // ---- lin318: n=318, optimal=42029 ----
    // D2=512 (需要 TSPXLargeProblem)
    run_instance<TSPXLargeProblem>("lin318", 42029.0f, LIN318_coords, LIN318_N, NS, seeds);
    
    // ---- pcb442: n=442, optimal=50778 ----
    // D2=512 (需要 TSPXLargeProblem)
    run_instance<TSPXLargeProblem>("pcb442", 50778.0f, PCB442_coords, PCB442_N, NS, seeds);
    
    printf("########################################################\n");
    printf("#  完成！                                               #\n");
    printf("########################################################\n");
    fflush(stdout);
    return 0;
}
