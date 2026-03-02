/**
 * bench_crossover.cu - OX 交叉效果对比测试
 * 
 * v2.0: Block 级架构，使用 solve() 框架
 */

#include "solver.cuh"
#include "tsp.cuh"
#include "tsp_large.cuh"
#include <cmath>
#include <cstdlib>

void generate_random_tsp(float* dist, int n, unsigned seed) {
    float* x = new float[n];
    float* y = new float[n];
    srand(seed);
    for (int i = 0; i < n; i++) {
        x[i] = (float)(rand() % 1000);
        y[i] = (float)(rand() % 1000);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = x[i] - x[j], dy = y[i] - y[j];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
    delete[] x;
    delete[] y;
}

const int N51 = 51;
const float eil51_coords[N51][2] = {
    {37,52},{49,49},{52,64},{20,26},{40,30},{21,47},{17,63},{31,62},{52,33},
    {51,21},{42,41},{31,32},{ 5,25},{12,42},{36,16},{52,41},{27,23},{17,33},
    {13,13},{57,58},{62,42},{42,57},{16,57},{ 8,52},{ 7,38},{27,68},{30,48},
    {43,67},{58,48},{58,27},{37,69},{38,46},{46,10},{61,33},{62,63},{63,69},
    {32,22},{45,35},{59,15},{ 5, 6},{10,17},{21,10},{ 5,64},{30,15},{39,10},
    {32,39},{25,32},{25,55},{48,28},{56,37},{30,40}
};

void compute_eil51_dist(float* dist) {
    for (int i = 0; i < N51; i++)
        for (int j = 0; j < N51; j++) {
            float dx = eil51_coords[i][0] - eil51_coords[j][0];
            float dy = eil51_coords[i][1] - eil51_coords[j][1];
            dist[i * N51 + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
}

int main() {
    print_device_info();
    
    // Test 1: TSP200 随机实例
    const int N200 = 200;
    float* h_dist200 = new float[N200 * N200];
    generate_random_tsp(h_dist200, N200, 12345);
    
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  Test 1: TSP200 随机实例 — 交叉效果对比 (Block 级)\n");
    printf("  Pop=4096, Gen=5000, SA T=15, Hybrid 16x256\n");
    printf("  Sol size = %zu bytes\n", sizeof(TSPLargeProblem::Sol));
    printf("══════════════════════════════════════════════════════════\n");
    
    struct TestCase {
        const char* label;
        float cx_rate;
    };
    TestCase cases[] = {
        {"[A] No crossover",       0.0f},
        {"[B] OX rate=10%",        0.1f},
        {"[C] OX rate=30%",        0.3f},
    };
    
    unsigned seeds[] = {42, 123, 456};
    
    for (auto& tc : cases) {
        printf("\n──── %s ────\n", tc.label);
        for (unsigned seed : seeds) {
            printf("  seed=%u:\n", seed);
            auto prob = TSPLargeProblem::create(h_dist200, N200);
            SolverConfig cfg;
            cfg.pop_size = 4096;
            cfg.max_gen = 5000;
            cfg.seed = seed;
            cfg.verbose = true;
            cfg.print_every = 500;
            cfg.num_islands = 16;
            cfg.migrate_interval = 50;
            cfg.migrate_strategy = MigrateStrategy::Hybrid;
            cfg.sa_temp_init = 15.0f;
            cfg.sa_alpha = 0.9995f;
            cfg.crossover_rate = tc.cx_rate;
            
            solve(prob, cfg);
            prob.destroy();
        }
    }
    delete[] h_dist200;
    
    // Test 2: TSP51 eil51
    float h_dist51[N51 * N51];
    compute_eil51_dist(h_dist51);
    
    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  Test 2: TSP51 eil51 (optimal=426) — 小规模验证 (Block 级)\n");
    printf("  Pop=4096, Gen=3000, SA T=10, Hybrid 16x256\n");
    printf("══════════════════════════════════════════════════════════\n");
    
    for (auto& tc : cases) {
        printf("\n──── %s ────\n", tc.label);
        for (unsigned seed : seeds) {
            printf("  seed=%u:\n", seed);
            auto prob = TSPProblem::create(h_dist51, N51);
            SolverConfig cfg;
            cfg.pop_size = 4096;
            cfg.max_gen = 3000;
            cfg.seed = seed;
            cfg.verbose = true;
            cfg.print_every = 500;
            cfg.num_islands = 16;
            cfg.migrate_interval = 50;
            cfg.migrate_strategy = MigrateStrategy::Hybrid;
            cfg.sa_temp_init = 10.0f;
            cfg.sa_alpha = 0.998f;
            cfg.crossover_rate = tc.cx_rate;
            
            solve(prob, cfg);
            prob.destroy();
        }
    }
    
    return 0;
}
