/**
 * bench_tsp_scale.cu - TSP 规模扩展基准测试
 * 
 * v2.0: Block 级架构，使用 solve() 框架
 * 
 * 测试规模: n = 51, 128, 256
 * 策略: HC + Hybrid Islands (Pop=4096, 16×256)
 */

#include "solver.cuh"
#include "tsp.cuh"
#include "tsp_large.cuh"
#include <chrono>
#include <cmath>
#include <cstdlib>

// ============================================================
// eil51 坐标
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

void compute_dist_from_coords(float* dist, const float coords[][2], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
}

void generate_random_dist(float* dist, int n, unsigned seed) {
    float* coords = new float[n * 2];
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245 + 12345;
        coords[i * 2] = (float)((s >> 16) % 1000);
        s = s * 1103515245 + 12345;
        coords[i * 2 + 1] = (float)((s >> 16) % 1000);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i*2] - coords[j*2];
            float dy = coords[i*2+1] - coords[j*2+1];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
    delete[] coords;
}

// ============================================================
// 主程序 — 使用 solve() 框架
// ============================================================

int main() {
    print_device_info();
    
    SolverConfig cfg;
    cfg.pop_size = 4096;
    cfg.max_gen = 2000;
    cfg.num_islands = 16;
    cfg.migrate_interval = 50;
    cfg.migrate_strategy = MigrateStrategy::Hybrid;
    cfg.verbose = false;
    
    unsigned seeds[] = {42, 123, 456};
    int num_seeds = 3;
    
    float* dist51 = new float[51 * 51];
    compute_dist_from_coords(dist51, eil51_coords, 51);
    
    float* dist128 = new float[128 * 128];
    generate_random_dist(dist128, 128, 7777);
    
    float* dist256 = new float[256 * 256];
    generate_random_dist(dist256, 256, 8888);
    
    // 预热
    {
        auto prob = TSPProblem::create(dist51, 51);
        SolverConfig w; w.pop_size = 256; w.max_gen = 10; w.seed = 42; w.verbose = false;
        solve(prob, w);
        prob.destroy();
    }
    
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  TSP Scale Test (Block 级架构)                              ║\n");
    printf("║  Pop=4096, Gen=2000, HC+Hybrid 16x256, %d seeds avg        ║\n", num_seeds);
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    
    // n=51 (TSPProblem, D2=64)
    {
        float sum_ms = 0, sum_best = 0;
        for (int s = 0; s < num_seeds; s++) {
            auto prob = TSPProblem::create(dist51, 51);
            SolverConfig c = cfg; c.seed = seeds[s];
            
            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0));
            CUDA_CHECK(cudaEventCreate(&t1));
            CUDA_CHECK(cudaEventRecord(t0));
            solve(prob, c);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            sum_ms += ms;
            CUDA_CHECK(cudaEventDestroy(t0));
            CUDA_CHECK(cudaEventDestroy(t1));
            prob.destroy();
        }
        printf("║  n=51   Sol=%zuB   avg %.1f ms                              ║\n",
               sizeof(TSPProblem::Sol), sum_ms / num_seeds);
    }
    
    // n=128, n=256 (TSPLargeProblem, D2=256)
    struct ScaleTest { const char* label; float* dist; int n; };
    ScaleTest tests[] = {
        {"128", dist128, 128},
        {"256", dist256, 256},
    };
    
    for (auto& t : tests) {
        float sum_ms = 0;
        for (int s = 0; s < num_seeds; s++) {
            auto prob = TSPLargeProblem::create(t.dist, t.n);
            SolverConfig c = cfg; c.seed = seeds[s];
            
            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0));
            CUDA_CHECK(cudaEventCreate(&t1));
            CUDA_CHECK(cudaEventRecord(t0));
            solve(prob, c);
            CUDA_CHECK(cudaEventRecord(t1));
            CUDA_CHECK(cudaEventSynchronize(t1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            sum_ms += ms;
            CUDA_CHECK(cudaEventDestroy(t0));
            CUDA_CHECK(cudaEventDestroy(t1));
            prob.destroy();
        }
        printf("║  n=%-4s Sol=%zuB   avg %.1f ms                              ║\n",
               t.label, sizeof(TSPLargeProblem::Sol), sum_ms / num_seeds);
    }
    
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    delete[] dist51;
    delete[] dist128;
    delete[] dist256;
    
    return 0;
}
