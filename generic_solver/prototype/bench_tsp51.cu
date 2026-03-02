/**
 * bench_tsp51.cu - eil51 性能基准测试
 * 
 * v2.0: Block 级架构
 *   Test 1: GPU vs CPU 规模扩展
 *   Test 2: 算法对比（纯爬山 vs SA，有/无岛屿）
 */

#include "solver.cuh"
#include "tsp.cuh"
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================
// eil51 坐标数据
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

void compute_dist_matrix(float* dist, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = eil51_coords[i][0] - eil51_coords[j][0];
            float dy = eil51_coords[i][1] - eil51_coords[j][1];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
}

// ============================================================
// CPU 单线程基线 — 纯爬山
// ============================================================

struct CPUSolution { int route[EIL51_N]; float cost; };

float cpu_eval(const int* route, const float* dist, int n) {
    float t = 0;
    for (int i = 0; i < n; i++) t += dist[route[i] * n + route[(i+1)%n]];
    return t;
}

void cpu_shuffle(int* a, int n, unsigned& s) {
    for (int i = n-1; i > 0; i--) { s=s*1103515245+12345; int j=(s>>16)%(i+1); std::swap(a[i],a[j]); }
}

void cpu_mutate(int* r, int n, unsigned& s) {
    s=s*1103515245+12345; int op=(s>>16)%3;
    s=s*1103515245+12345; int i=(s>>16)%n;
    s=s*1103515245+12345; int j=(s>>16)%(n-1); if(j>=i) j++;
    if (op==0) { std::swap(r[i],r[j]); }
    else if (op==1) { int lo=std::min(i,j),hi=std::max(i,j); while(lo<hi){std::swap(r[lo],r[hi]);lo++;hi--;} }
    else { int v=r[i]; if(i<j){for(int k=i;k<j;k++)r[k]=r[k+1];}else{for(int k=i;k>j;k--)r[k]=r[k-1];}r[j]=v; }
}

float cpu_solve(const float* dist, int n, int pop_size, int max_gen,
                unsigned seed, float& best_out) {
    auto t0 = std::chrono::high_resolution_clock::now();
    CPUSolution* pop = new CPUSolution[pop_size];
    int* tmp = new int[n];
    for (int p = 0; p < pop_size; p++) {
        for (int i = 0; i < n; i++) pop[p].route[i] = i;
        unsigned s = seed + p; cpu_shuffle(pop[p].route, n, s);
        pop[p].cost = cpu_eval(pop[p].route, dist, n);
    }
    for (int g = 0; g < max_gen; g++) {
        for (int p = 0; p < pop_size; p++) {
            memcpy(tmp, pop[p].route, sizeof(int)*n);
            float old = pop[p].cost;
            unsigned s = seed + g*pop_size + p;
            cpu_mutate(pop[p].route, n, s);
            pop[p].cost = cpu_eval(pop[p].route, dist, n);
            if (pop[p].cost >= old) { memcpy(pop[p].route, tmp, sizeof(int)*n); pop[p].cost = old; }
        }
    }
    int best = 0;
    for (int i = 1; i < pop_size; i++) if (pop[i].cost < pop[best].cost) best = i;
    best_out = pop[best].cost;
    delete[] pop; delete[] tmp;
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================
// GPU 求解（使用 solve() 框架 + 计时）
// ============================================================

float gpu_solve(const float* h_dist, int n, const SolverConfig& cfg, float& best_out) {
    using Sol = TSPProblem::Sol;
    auto prob = TSPProblem::create(h_dist, n);
    ProblemConfig pcfg = prob.config();
    
    bool use_sa = cfg.sa_temp_init > 0.0f;
    bool use_islands = cfg.num_islands > 1;
    int island_size = use_islands ? cfg.pop_size / cfg.num_islands : cfg.pop_size;
    
    const int block_threads = BLOCK_LEVEL_THREADS;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    Population<Sol> pop;
    pop.allocate(cfg.pop_size, block_threads);
    pop.init_rng(cfg.seed, 256);
    pop.init_population(pcfg, 256);
    
    ObjConfig oc = make_obj_config(pcfg);
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
    
    size_t prob_smem = prob.shared_mem_bytes();
    size_t total_smem = sizeof(Sol) + prob_smem + sizeof(MultiStepCandidate) * block_threads;
    
    int grid = cfg.pop_size;
    
    // 初始评估
    {
        size_t eval_smem = prob.shared_mem_bytes();
        int eval_grid = calc_grid_size(cfg.pop_size, block_threads);
        evaluate_kernel<<<eval_grid, block_threads, eval_smem>>>(
            prob, pop.d_solutions, cfg.pop_size, eval_smem);
    }
    
    if (use_sa) {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_global_best, pop.d_solutions + idx, sizeof(Sol), cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int batch;
    if (use_islands)
        batch = cfg.migrate_interval;
    else if (use_sa)
        batch = 50;
    else
        batch = cfg.max_gen;
    
    CUDA_CHECK(cudaEventRecord(start));
    
    int gen_done = 0;
    int migrate_round = 0;
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
        
        evolve_block_kernel<<<grid, block_threads, total_smem>>>(
            prob, pop.d_solutions, cfg.pop_size,
            pcfg.encoding, pcfg.dim1,
            oc, pop.d_rng_states,
            cfg.sa_alpha, prob_smem,
            nullptr, nullptr, nullptr, 0, 0, 0, d_params);
        gen_done += gens;
        
        if (use_islands) {
            migrate_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size,
                                      island_size, oc,
                                      cfg.migrate_strategy, d_params);
            migrate_round++;
        }
        
        if (use_sa) {
            elite_inject_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size,
                                           d_global_best, oc);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    
    if (use_sa) {
        Sol best;
        CUDA_CHECK(cudaMemcpy(&best, d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost));
        best_out = best.objectives[0];
    } else {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int h_best_idx;
        CUDA_CHECK(cudaMemcpy(&h_best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        Sol best = pop.download_solution(h_best_idx);
        best_out = best.objectives[0];
    }
    
    CUDA_CHECK(cudaFree(d_best_idx));
    if (d_params) CUDA_CHECK(cudaFree(d_params));
    if (d_global_best) CUDA_CHECK(cudaFree(d_global_best));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    prob.destroy();
    return gpu_ms;
}

// ============================================================
// 主程序
// ============================================================

int main() {
    print_device_info();
    printf("  Sol size:       %zu bytes\n", sizeof(TSPProblem::Sol));
    
    float h_dist[EIL51_N * EIL51_N];
    compute_dist_matrix(h_dist, EIL51_N);
    
    const int GEN = 2000;
    const unsigned SEED = 42;
    
    // 预热
    SolverConfig warmup; warmup.pop_size=256; warmup.max_gen=10;
    warmup.seed=SEED; warmup.verbose=false;
    float dummy; gpu_solve(h_dist, EIL51_N, warmup, dummy);
    
    // ============================================================
    // 测试 1: GPU vs CPU 规模扩展（纯爬山）
    // ============================================================
    printf("\n╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Test 1: 规模扩展 (纯爬山, Gen=%d, Block级)         ║\n", GEN);
    printf("║  Pop      GPU(ms)   CPU(ms)   Speedup  GPU_best    ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    
    int test_pops[] = {256, 1024, 4096, 8192};
    for (int tp : test_pops) {
        SolverConfig c; c.pop_size=tp; c.max_gen=GEN; c.seed=SEED; c.verbose=false;
        float gb, cb;
        float tg = gpu_solve(h_dist, EIL51_N, c, gb);
        float tc = cpu_solve(h_dist, EIL51_N, tp, GEN, SEED, cb);
        float sp = tc / tg;
        printf("║  %5d  %8.1f  %8.1f  %6.1fx    %.0f (%.1f%%)    ║\n",
               tp, tg, tc, sp, gb, (gb-426)/426*100);
    }
    printf("╚═══════════════════════════════════════════════════════╝\n");
    
    // ============================================================
    // 测试 2: 算法对比（Pop=4096, Gen=2000, 5 seeds 平均）
    // ============================================================
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 2: 算法对比 (Pop=4096, Gen=%d, optimal=426)        ║\n", GEN);
    printf("║  算法                       Best   Gap     ms            ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    
    const int QPOP = 4096;
    
    struct TestCase {
        const char* name;
        int num_islands;
        int migrate_interval;
        float sa_temp;
        float sa_alpha;
    };
    TestCase cases[] = {
        {"HC, no islands",            1,  0,  0.0f,  0.0f},
        {"HC, Hybrid 16x256",         16, 50, 0.0f,  0.0f},
        {"SA T=5, no islands",        1,  0,  5.0f,  0.998f},
        {"SA T=10, no islands",       1,  0,  10.0f, 0.998f},
        {"SA T=20, no islands",       1,  0,  20.0f, 0.998f},
        {"SA T=10, Hybrid 16x256",    16, 50, 10.0f, 0.998f},
        {"SA T=20, Hybrid 16x256",    16, 50, 20.0f, 0.998f},
    };
    
    unsigned seeds[] = {42, 123, 456, 789, 2024};
    int num_seeds = 5;
    
    for (auto& tc : cases) {
        float sum_best = 0, sum_ms = 0;
        for (int s = 0; s < num_seeds; s++) {
            SolverConfig c;
            c.pop_size = QPOP; c.max_gen = GEN; c.seed = seeds[s]; c.verbose = false;
            c.num_islands = tc.num_islands;
            c.migrate_interval = tc.migrate_interval;
            c.sa_temp_init = tc.sa_temp;
            c.sa_alpha = tc.sa_alpha;
            float best, ms;
            ms = gpu_solve(h_dist, EIL51_N, c, best);
            sum_best += best;
            sum_ms += ms;
        }
        float avg_best = sum_best / num_seeds;
        float avg_ms = sum_ms / num_seeds;
        printf("║  %-27s  %.0f  %4.1f%%  %6.1f            ║\n",
               tc.name, avg_best, (avg_best-426)/426*100, avg_ms);
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  (5 seeds 平均, HC=Hill Climbing, SA=Simulated Annealing)\n");
    
    return 0;
}
