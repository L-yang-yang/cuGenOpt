/**
 * bench_config_compare.cu - 配置策略对比（压力测试版）
 * 
 * 用 kroA100 (100城市, 最优=21282) + 极低预算
 * 让不同策略的差距充分暴露
 *
 * 每代实际搜索量 = pop_size × 128 (block threads)
 * p64 g200 = 64×200×128 = 164万次 move 评估（很紧张）
 */

#include "solver.cuh"
#include "tsp_large.cuh"
#include <cmath>
#include <cstdio>

const int N = 100;
const float coords[N][2] = {
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
const float OPT = 21282.0f;

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
    return best_val;
}

void run(float* dist, const char* name, SolverConfig cfg, int ns, const unsigned* seeds) {
    float sum=0, mn=1e30f, mx=0, tms=0;
    for (int s = 0; s < ns; s++) {
        cfg.seed = seeds[s];
        auto prob = TSPLargeProblem::create(dist, N);
        float ms; float v = single_solve(prob, cfg, ms);
        sum += v; if(v<mn)mn=v; if(v>mx)mx=v; tms += ms;
        prob.destroy();
    }
    float gb = (mn-OPT)/OPT*100, ga = (sum/ns-OPT)/OPT*100;
    printf("  %-38s %6.0f %6.0f %6.0f  %5.1f%%  %5.1f%%  %5.0fms\n",
           name, mn, sum/ns, mx, gb, ga, tms/ns);
    fflush(stdout);
}

int main() {
    print_device_info(); fflush(stdout);

    float dist[N*N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dx = coords[i][0]-coords[j][0], dy = coords[i][1]-coords[j][1];
            dist[i*N+j] = roundf(sqrtf(dx*dx+dy*dy));
        }

    // 预热
    { auto p=TSPLargeProblem::create(dist,N); SolverConfig c; c.pop_size=64; c.max_gen=10; c.seed=1; c.verbose=false; solve(p,c); p.destroy(); }
    printf("  (warmup done)\n\n"); fflush(stdout);

    const unsigned seeds[] = {42, 123, 456, 789, 2024, 7, 314, 999, 1111, 8888};
    const int NS = 10;

    printf("================================================================\n");
    printf("  kroA100 (n=100, optimal=21282) — 低预算策略对比\n");
    printf("  10 seeds, 128 threads/block, 实际搜索=pop×gen×128\n");
    printf("================================================================\n");
    printf("  %-38s %6s %6s %6s  %5s  %5s  %5s\n",
           "Config", "Best", "Avg", "Worst", "GapB", "GapA", "Time");
    printf("----------------------------------------------------------------\n"); fflush(stdout);

    // ============================================================
    // 实验 1: HC vs SA（极低预算 p64 g200）
    // 搜索量 = 64 × 200 × 128 = 164 万次
    // ============================================================
    printf("  >>> HC vs SA (p64 g200, ~1.6M evals)\n"); fflush(stdout);
    {
        SolverConfig c; c.pop_size=64; c.max_gen=200; c.verbose=false;
        run(dist, "HC p64 g200", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=64; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=10; c.sa_alpha=0.98f;
        run(dist, "SA T=10 p64 g200", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=64; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.98f;
        run(dist, "SA T=50 p64 g200", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=64; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=200; c.sa_alpha=0.98f;
        run(dist, "SA T=200 p64 g200", c, NS, seeds);
    }
    printf("----------------------------------------------------------------\n"); fflush(stdout);

    // ============================================================
    // 实验 2: 有/无岛屿（p256 g200）
    // 搜索量 = 256 × 200 × 128 = 655 万次
    // ============================================================
    printf("  >>> Islands (SA T=50, p256 g200, ~6.5M evals)\n"); fflush(stdout);
    {
        SolverConfig c; c.pop_size=256; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.98f;
        run(dist, "SA no-isl p256 g200", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.98f;
        c.num_islands=4; c.migrate_interval=25; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "SA Isl4 p256 g200", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.98f;
        c.num_islands=8; c.migrate_interval=25; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "SA Isl8 p256 g200", c, NS, seeds);
    }
    printf("----------------------------------------------------------------\n"); fflush(stdout);

    // ============================================================
    // 实验 3: 种群大小（固定总搜索量 ~655万次）
    // p64×g800  p128×g400  p256×g200  p512×g100
    // ============================================================
    printf("  >>> Pop vs Gen tradeoff (~6.5M evals total)\n"); fflush(stdout);
    {
        SolverConfig c; c.pop_size=64; c.max_gen=800; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.99f;
        c.num_islands=4; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "SA Isl4 p64 g800 (少解多代)", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=128; c.max_gen=400; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.99f;
        c.num_islands=4; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "SA Isl4 p128 g400", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.98f;
        c.num_islands=4; c.migrate_interval=25; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "SA Isl4 p256 g200", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=512; c.max_gen=100; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.97f;
        c.num_islands=4; c.migrate_interval=25; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "SA Isl4 p512 g100 (多解少代)", c, NS, seeds);
    }
    printf("----------------------------------------------------------------\n"); fflush(stdout);

    // ============================================================
    // 实验 4: 逐步加预算看收敛速度
    // ============================================================
    printf("  >>> Budget scaling (SA T=50, Isl4, p256)\n"); fflush(stdout);
    {
        SolverConfig c; c.pop_size=256; c.max_gen=50; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.96f;
        c.num_islands=4; c.migrate_interval=25; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "p256 g50   (~1.6M evals)", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=100; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.97f;
        c.num_islands=4; c.migrate_interval=25; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "p256 g100  (~3.3M evals)", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=200; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.98f;
        c.num_islands=4; c.migrate_interval=25; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "p256 g200  (~6.5M evals)", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=500; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.99f;
        c.num_islands=4; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "p256 g500  (~16M evals)", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=1000; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.995f;
        c.num_islands=4; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "p256 g1000 (~33M evals)", c, NS, seeds);
    }
    {
        SolverConfig c; c.pop_size=256; c.max_gen=2000; c.verbose=false;
        c.sa_temp_init=50; c.sa_alpha=0.998f;
        c.num_islands=4; c.migrate_interval=50; c.migrate_strategy=MigrateStrategy::Hybrid;
        run(dist, "p256 g2000 (~65M evals)", c, NS, seeds);
    }

    printf("================================================================\n");
    printf("\n  optimal=21282\n");
    printf("  实际搜索量 = pop_size × max_gen × 128 (block threads)\n\n");
    fflush(stdout);
    return 0;
}
