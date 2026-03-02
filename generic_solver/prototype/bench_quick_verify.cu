/**
 * bench_quick_verify.cu - 快速回归验证
 * 
 * 低资源配置（小 pop、少 gen），3 seeds
 * 目的：用有限资源暴露重构引入的 bug
 *   - 资源充足时全部收敛到最优，看不出差异
 *   - 资源紧张时，如果搜索逻辑有问题会表现为 gap 变大
 * 
 * 预期基线（重构前）：
 *   eil51  HC p64  g500:  ~430-440 (gap ~1-3%)
 *   eil51  SA p128 g1k:   ~426-428 (gap ~0-0.5%)
 *   kroA100 SA p128 g1k:  ~22000-23000 (gap ~3-8%)
 *   CVRP   SA p64  g1k:   feasible, ~800-900
 */

#include "solver.cuh"
#include "tsp.cuh"
#include "tsp_large.cuh"
#include "knapsack.cuh"
#include "vrp.cuh"
#include <cmath>
#include <cstdio>

static void compute_euc2d_dist(float* dist, const float coords[][2], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
}

// TSP eil51 坐标
const int EIL51_N = 51;
const float eil51_coords[EIL51_N][2] = {
    {37,52},{49,49},{52,64},{20,26},{40,30},{21,47},{17,63},{31,62},{52,33},
    {51,21},{42,41},{31,32},{ 5,25},{12,42},{36,16},{52,41},{27,23},{17,33},
    {13,13},{57,58},{62,42},{42,57},{16,57},{ 8,52},{ 7,38},{27,68},{30,48},
    {43,67},{58,48},{58,27},{37,69},{38,46},{46,10},{61,33},{62,63},{63,69},
    {32,22},{45,35},{59,15},{ 5, 6},{10,17},{21,10},{ 5,64},{30,15},{39,10},
    {32,39},{25,32},{25,55},{48,28},{56,37},{30,40}
};

// TSP kroA100 坐标
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

// CVRP A-n32-k5
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

// 单次求解
template<typename Problem>
float single_solve(Problem& prob, const SolverConfig& cfg, float& elapsed_ms) {
    using Sol = typename Problem::Sol;
    ProblemConfig pcfg = prob.config();
    ObjConfig oc = make_obj_config(pcfg);
    const int block_threads = BLOCK_LEVEL_THREADS;
    bool use_sa = cfg.sa_temp_init > 0.0f;
    bool use_islands = cfg.num_islands > 1;
    int island_size = use_islands ? cfg.pop_size / cfg.num_islands : cfg.pop_size;
    bool use_aos = cfg.use_aos;

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
    if (use_sa) CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(Sol)));

    AOSStats* d_aos_stats = nullptr;
    AOSStats* h_aos_stats = nullptr;
    if (use_aos) {
        CUDA_CHECK(cudaMalloc(&d_aos_stats, sizeof(AOSStats) * cfg.pop_size));
        h_aos_stats = new AOSStats[cfg.pop_size];
    }

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
            cudaFuncSetAttribute(evolve_block_kernel<Problem, Sol>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, total_smem);
        }
    }

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

        if (use_aos) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_aos_stats, d_aos_stats,
                sizeof(AOSStats) * cfg.pop_size, cudaMemcpyDeviceToHost));
            int total_usage[MAX_SEQ] = {}, total_improve[MAX_SEQ] = {};
            for (int b = 0; b < cfg.pop_size; b++)
                for (int i = 0; i < seq_reg.count; i++) {
                    total_usage[i] += h_aos_stats[b].usage[i];
                    total_improve[i] += h_aos_stats[b].improvement[i];
                }
            constexpr float A = 0.8f;
            float new_w[MAX_SEQ]; float sum = 0;
            for (int i = 0; i < seq_reg.count; i++) {
                float rate = total_usage[i] > 0 ? (float)total_improve[i]/total_usage[i] : 0;
                new_w[i] = A * seq_reg.weights[i] + (1-A) * (rate + 0.02f);
                sum += new_w[i];
            }
            if (sum > 0) for (int i = 0; i < seq_reg.count; i++) seq_reg.weights[i] = new_w[i]/sum;
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

    float best_val;
    if (use_sa) {
        Sol best;
        CUDA_CHECK(cudaMemcpy(&best, d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost));
        best_val = best.objectives[0];
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

    CUDA_CHECK(cudaFree(d_best_idx));
    if (d_params) CUDA_CHECK(cudaFree(d_params));
    if (d_global_best) CUDA_CHECK(cudaFree(d_global_best));
    if (d_aos_stats) CUDA_CHECK(cudaFree(d_aos_stats));
    if (h_aos_stats) delete[] h_aos_stats;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return best_val;
}

struct BenchResult { float best, avg, worst, avg_ms; int n; };

void print_row(const char* name, float opt, BenchResult r) {
    float gb = (r.best - opt) / opt * 100;
    float ga = (r.avg  - opt) / opt * 100;
    printf("  %-36s %7.0f %7.0f %7.0f  %5.1f%%  %5.1f%%  %5.0fms\n",
           name, r.best, r.avg, r.worst, gb, ga, r.avg_ms);
    fflush(stdout);
}

int main() {
    print_device_info();

    const unsigned seeds[] = {42, 123, 456};
    const int NS = 3;

    // 预热
    {
        float dd[25] = {};
        for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) dd[i*5+j] = (i==j)?0:10;
        auto p = TSPProblem::create(dd, 5);
        SolverConfig c; c.pop_size=32; c.max_gen=10; c.seed=1; c.verbose=false;
        solve(p, c);
        p.destroy();
    }

    printf("\n");
    printf("================================================================\n");
    printf("  快速回归验证 — 低资源配置, 3 seeds\n");
    printf("================================================================\n");
    printf("  %-36s %7s %7s %7s  %5s  %5s  %5s\n",
           "Config", "Best", "Avg", "Worst", "GapB", "GapA", "Time");
    printf("----------------------------------------------------------------\n");
    fflush(stdout);

    // --- eil51 ---
    {
        float dist[EIL51_N * EIL51_N];
        compute_euc2d_dist(dist, eil51_coords, EIL51_N);
        const float OPT = 426.0f;
        printf("  --- TSP eil51 (optimal=426) ---\n"); fflush(stdout);

        // HC p64 g500 — 很紧张，应该看到 gap
        {
            float sum=0,mn=1e30f,mx=0,tms=0;
            for (int s = 0; s < NS; s++) {
                SolverConfig c; c.pop_size=64; c.max_gen=500; c.verbose=false; c.seed=seeds[s];
                auto prob = TSPProblem::create(dist, EIL51_N);
                float ms; float v = single_solve(prob, c, ms);
                sum+=v; if(v<mn)mn=v; if(v>mx)mx=v; tms+=ms;
                prob.destroy();
            }
            print_row("eil51 HC p64 g500", OPT, {mn,sum/NS,mx,tms/NS,NS});
        }
        // SA p128 g1k
        {
            float sum=0,mn=1e30f,mx=0,tms=0;
            for (int s = 0; s < NS; s++) {
                SolverConfig c; c.pop_size=128; c.max_gen=1000; c.verbose=false; c.seed=seeds[s];
                c.sa_temp_init=10; c.sa_alpha=0.998f;
                auto prob = TSPProblem::create(dist, EIL51_N);
                float ms; float v = single_solve(prob, c, ms);
                sum+=v; if(v<mn)mn=v; if(v>mx)mx=v; tms+=ms;
                prob.destroy();
            }
            print_row("eil51 SA p128 g1k", OPT, {mn,sum/NS,mx,tms/NS,NS});
        }
        // SA p256 g2k — 中等
        {
            float sum=0,mn=1e30f,mx=0,tms=0;
            for (int s = 0; s < NS; s++) {
                SolverConfig c; c.pop_size=256; c.max_gen=2000; c.verbose=false; c.seed=seeds[s];
                c.sa_temp_init=15; c.sa_alpha=0.999f;
                auto prob = TSPProblem::create(dist, EIL51_N);
                float ms; float v = single_solve(prob, c, ms);
                sum+=v; if(v<mn)mn=v; if(v>mx)mx=v; tms+=ms;
                prob.destroy();
            }
            print_row("eil51 SA p256 g2k", OPT, {mn,sum/NS,mx,tms/NS,NS});
        }
    }

    printf("----------------------------------------------------------------\n"); fflush(stdout);

    // --- kroA100 ---
    {
        float dist[KROA100_N * KROA100_N];
        compute_euc2d_dist(dist, kroA100_coords, KROA100_N);
        const float OPT = 21282.0f;
        printf("  --- TSP kroA100 (optimal=21282) ---\n"); fflush(stdout);

        // SA p64 g500 — 很紧张
        {
            float sum=0,mn=1e30f,mx=0,tms=0;
            for (int s = 0; s < NS; s++) {
                SolverConfig c; c.pop_size=64; c.max_gen=500; c.verbose=false; c.seed=seeds[s];
                c.sa_temp_init=50; c.sa_alpha=0.998f;
                auto prob = TSPLargeProblem::create(dist, KROA100_N);
                float ms; float v = single_solve(prob, c, ms);
                sum+=v; if(v<mn)mn=v; if(v>mx)mx=v; tms+=ms;
                prob.destroy();
            }
            print_row("kroA100 SA p64 g500", OPT, {mn,sum/NS,mx,tms/NS,NS});
        }
        // SA p128 g1k
        {
            float sum=0,mn=1e30f,mx=0,tms=0;
            for (int s = 0; s < NS; s++) {
                SolverConfig c; c.pop_size=128; c.max_gen=1000; c.verbose=false; c.seed=seeds[s];
                c.sa_temp_init=50; c.sa_alpha=0.999f;
                auto prob = TSPLargeProblem::create(dist, KROA100_N);
                float ms; float v = single_solve(prob, c, ms);
                sum+=v; if(v<mn)mn=v; if(v>mx)mx=v; tms+=ms;
                prob.destroy();
            }
            print_row("kroA100 SA p128 g1k", OPT, {mn,sum/NS,mx,tms/NS,NS});
        }
    }

    printf("----------------------------------------------------------------\n"); fflush(stdout);

    // --- CVRP A-n32-k5 ---
    {
        float dist[AN32K5_NODES * AN32K5_NODES];
        compute_euc2d_dist(dist, an32k5_coords, AN32K5_NODES);
        const float OPT = 784.0f;
        printf("  --- CVRP A-n32-k5 (optimal=784) ---\n"); fflush(stdout);

        // SA+AOS p64 g1k
        {
            float sum=0,mn=1e30f,mx=0,tms=0;
            for (int s = 0; s < NS; s++) {
                SolverConfig c; c.pop_size=64; c.max_gen=1000; c.verbose=false; c.seed=seeds[s];
                c.sa_temp_init=20; c.sa_alpha=0.999f;
                c.num_islands=4; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
                c.use_aos=true;
                auto prob = VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5);
                float ms; float v = single_solve(prob, c, ms);
                sum+=v; if(v<mn)mn=v; if(v>mx)mx=v; tms+=ms;
                prob.destroy();
            }
            print_row("A-n32-k5 SA+AOS p64 g1k", OPT, {mn,sum/NS,mx,tms/NS,NS});
        }
        // SA+AOS p128 g2k
        {
            float sum=0,mn=1e30f,mx=0,tms=0;
            for (int s = 0; s < NS; s++) {
                SolverConfig c; c.pop_size=128; c.max_gen=2000; c.verbose=false; c.seed=seeds[s];
                c.sa_temp_init=20; c.sa_alpha=0.999f;
                c.num_islands=8; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
                c.use_aos=true;
                auto prob = VRPProblem::create(dist, an32k5_demands, AN32K5_N, 100.0f, 5, 5);
                float ms; float v = single_solve(prob, c, ms);
                sum+=v; if(v<mn)mn=v; if(v>mx)mx=v; tms+=ms;
                prob.destroy();
            }
            print_row("A-n32-k5 SA+AOS p128 g2k", OPT, {mn,sum/NS,mx,tms/NS,NS});
        }
    }

    printf("================================================================\n");
    printf("  GapB = (Best-Opt)/Opt, GapA = (Avg-Opt)/Opt\n");
    printf("  Seeds: {42, 123, 456}\n");
    printf("================================================================\n");
    fflush(stdout);
    return 0;
}
