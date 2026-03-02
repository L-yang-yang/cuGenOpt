/**
 * bench_ablation_lns.cu - LNS 算子消融实验（P1: 含 GUIDED_REBUILD）
 * 
 * 对比两组：
 *   A) 完整序列集（含 LNS_SEGMENT_SHUFFLE + LNS_SCATTER_SHUFFLE + LNS_GUIDED_REBUILD）
 *   B) 去掉 LNS 的序列集（仅原子算子）
 * 
 * 测试实例（按规模递增）：
 *   1. eil51     (n=51,  optimal=426)    — 基线，原子算子即可收敛
 *   2. kroA100   (n=100, optimal=21282)  — 中等规模
 *   3. ch150     (n=150, optimal=6528)   — 大规模，原子算子难以收敛
 *   4. tsp225    (n=225, optimal=3916)   — 超大规模，LNS 应有明显优势
 * 
 * 配置策略：小种群 + 长搜索，让 G/O 矩阵有足够时间积累信息
 */

#include "solver.cuh"
#include "tsp.cuh"
#include "tsp_large.cuh"
#include "tsp_xlarge.cuh"
#include "tsplib_data.h"
#include "vrp.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>

static void compute_euc2d_dist(float* dist, const float coords[][2], int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0] - coords[j][0];
            float dy = coords[i][1] - coords[j][1];
            dist[i * n + j] = roundf(sqrtf(dx * dx + dy * dy));
        }
}

// eil51
const int EIL51_N = 51;
const float eil51_coords[EIL51_N][2] = {
    {37,52},{49,49},{52,64},{20,26},{40,30},{21,47},{17,63},{31,62},{52,33},
    {51,21},{42,41},{31,32},{ 5,25},{12,42},{36,16},{52,41},{27,23},{17,33},
    {13,13},{57,58},{62,42},{42,57},{16,57},{ 8,52},{ 7,38},{27,68},{30,48},
    {43,67},{58,48},{58,27},{37,69},{38,46},{46,10},{61,33},{62,63},{63,69},
    {32,22},{45,35},{59,15},{ 5, 6},{10,17},{21,10},{ 5,64},{30,15},{39,10},
    {32,39},{25,32},{25,55},{48,28},{56,37},{30,40}
};

// kroA100
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

// SeqID 名称映射
const char* seq_name(int seq_id) {
    switch (seq_id) {
        case 0: return "SWAP";
        case 1: return "REVERSE";
        case 2: return "INSERT";
        case 3: return "3OPT";
        case 4: return "OR_OPT";
        case 5: return "X_RELOC";
        case 6: return "X_SWAP";
        case 7: return "SEG_RELOC";
        case 8: return "SEG_SWAP";
        case 9: return "X_EXCH";
        case 10: return "ROW_SWAP";
        case 11: return "ROW_REV";
        case 12: return "ROW_SPL";
        case 13: return "ROW_MRG";
        case 14: return "PERTURB";
        case 20: return "LNS_SEG";
        case 21: return "LNS_SCT";
        case 22: return "LNS_GUIDE";
        default: return "???";
    }
}

// 累积 AOS 统计
struct AccumAOS {
    long long usage[MAX_SEQ];
    long long improvement[MAX_SEQ];
    int count;
    int ids[MAX_SEQ];
};

// 求解函数：返回 best obj，输出 AOS 累积统计
template<typename Problem>
float solve_with_stats(Problem& prob, const SolverConfig& cfg,
                       SeqRegistry seq_reg,  // 按值传入，可修改
                       float& elapsed_ms, AccumAOS& accum) {
    using Sol = typename Problem::Sol;
    ProblemConfig pcfg = prob.config();
    ObjConfig oc = make_obj_config(pcfg);
    const int block_threads = BLOCK_LEVEL_THREADS;
    bool use_sa = cfg.sa_temp_init > 0.0f;
    bool use_islands = cfg.num_islands > 1;
    int island_size = use_islands ? cfg.pop_size / cfg.num_islands : cfg.pop_size;

    // 记录 registry 信息到 accum
    accum.count = seq_reg.count;
    for (int i = 0; i < seq_reg.count; i++) {
        accum.ids[i] = seq_reg.ids[i];
        accum.usage[i] = 0;
        accum.improvement[i] = 0;
    }

    Population<Sol> pop;
    pop.allocate(cfg.pop_size, block_threads);
    pop.init_rng(cfg.seed, 256);
    pop.init_population(pcfg, 256);

    KStepConfig kstep = build_kstep_config();
    int* d_best_idx;
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));
    
    EvolveParams h_params;
    EvolveParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(EvolveParams)));

    Sol* d_global_best = nullptr;
    if (use_sa) CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(Sol)));

    // AOS 始终开启（消融实验需要统计）
    AOSStats* d_aos_stats = nullptr;
    AOSStats* h_aos_stats = nullptr;
    CUDA_CHECK(cudaMalloc(&d_aos_stats, sizeof(AOSStats) * cfg.pop_size));
    h_aos_stats = new AOSStats[cfg.pop_size];

    // 关系矩阵（GUIDED_REBUILD 使用）
    bool use_rel = false;
    RelationMatrix rel_mat = {};
    int rel_N = 0;
    if (pcfg.encoding == EncodingType::Permutation) {
        for (int i = 0; i < seq_reg.count; i++) {
            if (seq_reg.ids[i] == seq::SEQ_LNS_GUIDED_REBUILD) {
                use_rel = true; break;
            }
        }
    }
    if (use_rel && pcfg.dim2_default > 0) {
        rel_N = pcfg.dim2_default;
        rel_mat = relation_matrix_create(rel_N, 0.95f);
        prob.init_relation_matrix(rel_mat.h_G, rel_mat.h_O, rel_N);
        relation_matrix_upload(rel_mat);
    } else {
        use_rel = false;
    }

    size_t prob_smem = prob.shared_mem_bytes();
    size_t total_smem = sizeof(Sol) + prob_smem
                      + sizeof(MultiStepCandidate) * block_threads
                      + sizeof(AOSStats);
    {
        cudaDeviceProp prop; int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (total_smem > (size_t)prop.sharedMemPerBlock) {
            prob_smem = 0;
            total_smem = sizeof(Sol) + sizeof(MultiStepCandidate) * block_threads + sizeof(AOSStats);
        }
        if (total_smem > 48 * 1024)
            cudaFuncSetAttribute(evolve_block_kernel<Problem, Sol>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, total_smem);
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

    // 关系矩阵需要定期更新，即使非 island 模式也要分 batch
    int batch;
    if (use_islands) batch = cfg.migrate_interval;
    else if (use_rel) batch = 200;  // 每 200 代更新一次 G/O 矩阵
    else batch = cfg.max_gen;
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
            d_aos_stats,
            use_rel ? rel_mat.d_G : nullptr,
            use_rel ? rel_mat.d_O : nullptr,
            rel_N, 0, 0, d_params);
        gen_done += gens;

        // AOS 权重更新 + 累积统计
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_aos_stats, d_aos_stats,
                sizeof(AOSStats) * cfg.pop_size, cudaMemcpyDeviceToHost));
            int tu[MAX_SEQ] = {}, ti[MAX_SEQ] = {};
            for (int b = 0; b < cfg.pop_size; b++)
                for (int i = 0; i < seq_reg.count; i++) {
                    tu[i] += h_aos_stats[b].usage[i];
                    ti[i] += h_aos_stats[b].improvement[i];
                }
            // 累积到 accum
            for (int i = 0; i < seq_reg.count; i++) {
                accum.usage[i] += tu[i];
                accum.improvement[i] += ti[i];
            }
            // 权重更新
            constexpr float A = 0.8f;
            float new_w[MAX_SEQ]; float sum = 0;
            for (int i = 0; i < seq_reg.count; i++) {
                float rate = tu[i] > 0 ? (float)ti[i]/tu[i] : 0;
                new_w[i] = A * seq_reg.weights[i] + (1-A) * (rate + 0.02f);
                sum += new_w[i];
            }
            if (sum > 0) for (int i = 0; i < seq_reg.count; i++) seq_reg.weights[i] = new_w[i]/sum;
        }

        // 关系矩阵更新（从 top-K 解统计）
        if (use_rel) {
            constexpr int REL_TOP_K = 4;
            Sol* h_pop_ptr = new Sol[cfg.pop_size];
            CUDA_CHECK(cudaMemcpy(h_pop_ptr, pop.d_solutions,
                                  sizeof(Sol) * cfg.pop_size, cudaMemcpyDeviceToHost));
            float* h_scores = new float[cfg.pop_size];
            for (int b = 0; b < cfg.pop_size; b++) {
                h_scores[b] = scalar_objective(h_pop_ptr[b], oc);
                if (h_pop_ptr[b].penalty > 0.0f) h_scores[b] = 1e30f;
            }
            int actual_k = (cfg.pop_size < REL_TOP_K) ? cfg.pop_size : REL_TOP_K;
            for (int k = 0; k < actual_k; k++) {
                int mi = 0;
                for (int b = 1; b < cfg.pop_size; b++)
                    if (h_scores[b] < h_scores[mi]) mi = b;
                relation_matrix_update(rel_mat, h_pop_ptr[mi], pcfg.dim1);
                h_scores[mi] = 1e30f;
            }
            delete[] h_scores;
            delete[] h_pop_ptr;
            relation_matrix_upload(rel_mat);
        }

        if (use_islands) {
            migrate_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size,
                island_size, oc, cfg.migrate_strategy, d_params);
            migrate_round++;
        }
        if (use_sa)
            elite_inject_kernel<<<1, 1>>>(pop.d_solutions, cfg.pop_size, d_global_best, oc);
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
        Sol pb = pop.download_solution(idx);
        if (pb.objectives[0] < best_val && pb.penalty <= 0.0f) best_val = pb.objectives[0];
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
    CUDA_CHECK(cudaFree(d_aos_stats));
    delete[] h_aos_stats;
    if (use_rel) relation_matrix_destroy(rel_mat);
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return best_val;
}

// 从 SeqRegistry 中移除 LNS 序列并重新归一化
SeqRegistry remove_lns(const SeqRegistry& src) {
    SeqRegistry dst;
    dst.count = 0;
    for (int i = 0; i < MAX_SEQ; i++) { dst.ids[i] = -1; dst.weights[i] = 0; }
    for (int i = 0; i < src.count; i++) {
        if (src.ids[i] >= 20) continue;  // 跳过 LNS 序列（id >= 20）
        dst.ids[dst.count] = src.ids[i];
        dst.weights[dst.count] = src.weights[i];
        dst.count++;
    }
    float sum = 0;
    for (int i = 0; i < dst.count; i++) sum += dst.weights[i];
    if (sum > 0) for (int i = 0; i < dst.count; i++) dst.weights[i] /= sum;
    return dst;
}

void print_aos_stats(const AccumAOS& a) {
    printf("    %-10s %10s %10s %8s %8s\n", "Operator", "Usage", "Improve", "Rate%", "Weight%");
    long long total_u = 0, total_i = 0;
    for (int i = 0; i < a.count; i++) { total_u += a.usage[i]; total_i += a.improvement[i]; }
    for (int i = 0; i < a.count; i++) {
        float rate = a.usage[i] > 0 ? 100.0f * a.improvement[i] / a.usage[i] : 0;
        float pct = total_u > 0 ? 100.0f * a.usage[i] / total_u : 0;
        printf("    %-10s %10lld %10lld %7.2f%% %7.1f%%\n",
               seq_name(a.ids[i]), a.usage[i], a.improvement[i], rate, pct);
    }
    printf("    %-10s %10lld %10lld %7.2f%%\n", "TOTAL", total_u, total_i,
           total_u > 0 ? 100.0f * total_i / total_u : 0);
}

struct RunResult { float best, avg, worst, avg_ms; };

template<typename Problem>
RunResult run_group(const char* label, Problem& prob_template,
                    const float* dist, int dist_n,
                    const SolverConfig& base_cfg,
                    SeqRegistry reg,
                    const unsigned* seeds, int ns,
                    // VRP 专用
                    const float* demands = nullptr,
                    int n_cust = 0, float cap = 0, int nv = 0, int mv = 0) {
    (void)prob_template; (void)dist; (void)dist_n;
    
    float sum = 0, mn = 1e30f, mx = 0, tms = 0;
    AccumAOS total_accum;
    total_accum.count = reg.count;
    for (int i = 0; i < reg.count; i++) {
        total_accum.ids[i] = reg.ids[i];
        total_accum.usage[i] = 0;
        total_accum.improvement[i] = 0;
    }

    for (int s = 0; s < ns; s++) {
        SolverConfig c = base_cfg;
        c.seed = seeds[s];
        
        Problem prob;
        if constexpr (std::is_same_v<Problem, VRPProblem>) {
            prob = VRPProblem::create(dist, demands, n_cust, cap, nv, mv);
        } else {
            prob = Problem::create(dist, dist_n);
        }
        
        float ms;
        AccumAOS run_accum;
        float v = solve_with_stats(prob, c, reg, ms, run_accum);
        
        sum += v; if (v < mn) mn = v; if (v > mx) mx = v; tms += ms;
        for (int i = 0; i < reg.count; i++) {
            total_accum.usage[i] += run_accum.usage[i];
            total_accum.improvement[i] += run_accum.improvement[i];
        }
        prob.destroy();
    }

    printf("  [%s] Best=%.0f Avg=%.0f Worst=%.0f  Time=%.0fms\n",
           label, mn, sum/ns, mx, tms/ns);
    print_aos_stats(total_accum);
    printf("\n");
    fflush(stdout);
    
    return {mn, sum/ns, mx, tms/ns};
}

int main() {
    print_device_info();

    const unsigned seeds[] = {42, 123, 456, 789, 2024};
    const int NS = 5;

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
    printf("  LNS 消融实验（P1: GUIDED_REBUILD 关系矩阵引导）\n");
    printf("  A = 完整序列集（含 LNS）  B = 去掉 LNS（仅原子算子）\n");
    printf("  5 seeds, AOS 开启, 小种群+长搜索\n");
    printf("================================================================\n\n");
    fflush(stdout);

    // ============================================================
    // 1. TSP eil51 (n=51) — 基线，原子算子即可收敛
    //    SA p64 g3k（资源受限，看 LNS 是否加速收敛）
    // ============================================================
    {
        float dist[EIL51_N * EIL51_N];
        compute_euc2d_dist(dist, eil51_coords, EIL51_N);
        const float OPT = 426.0f;
        printf("--- 1. TSP eil51 (n=%d, optimal=%.0f) | SA p64 g3k ---\n", EIL51_N, OPT);
        fflush(stdout);

        SolverConfig c;
        c.pop_size = 64; c.max_gen = 3000; c.verbose = false;
        c.sa_temp_init = 15; c.sa_alpha = 0.999f;
        c.use_aos = true;

        ProblemConfig pcfg;
        pcfg.encoding = EncodingType::Permutation;
        pcfg.dim1 = 1;
        SeqRegistry full_reg = build_seq_registry(pcfg.encoding, pcfg.dim1, 0.0f);
        SeqRegistry no_lns_reg = remove_lns(full_reg);

        TSPProblem dummy = TSPProblem::create(dist, EIL51_N);
        auto rA = run_group<TSPProblem>("A:+LNS", dummy, dist, EIL51_N, c, full_reg, seeds, NS);
        auto rB = run_group<TSPProblem>("B:-LNS", dummy, dist, EIL51_N, c, no_lns_reg, seeds, NS);
        dummy.destroy();

        printf("  Delta(A-B): Avg=%.1f (%.2f%%)  Best: A=%.0f B=%.0f\n",
               rA.avg - rB.avg, (rA.avg - rB.avg)/OPT*100, rA.best, rB.best);
        printf("  GapA=%.2f%%  GapB=%.2f%%\n\n",
               (rA.avg - OPT)/OPT*100, (rB.avg - OPT)/OPT*100);
        fflush(stdout);
    }

    // ============================================================
    // 2. TSP kroA100 (n=100) — 中等规模
    //    SA p64 g5k（资源受限，G/O 有时间积累）
    // ============================================================
    {
        float dist[KROA100_N * KROA100_N];
        compute_euc2d_dist(dist, kroA100_coords, KROA100_N);
        const float OPT = 21282.0f;
        printf("--- 2. TSP kroA100 (n=%d, optimal=%.0f) | SA p64 g5k ---\n", KROA100_N, OPT);
        fflush(stdout);

        SolverConfig c;
        c.pop_size = 64; c.max_gen = 5000; c.verbose = false;
        c.sa_temp_init = 50; c.sa_alpha = 0.9995f;
        c.use_aos = true;

        ProblemConfig pcfg;
        pcfg.encoding = EncodingType::Permutation;
        pcfg.dim1 = 1;
        SeqRegistry full_reg = build_seq_registry(pcfg.encoding, pcfg.dim1, 0.0f);
        SeqRegistry no_lns_reg = remove_lns(full_reg);

        TSPLargeProblem dummy = TSPLargeProblem::create(dist, KROA100_N);
        auto rA = run_group<TSPLargeProblem>("A:+LNS", dummy, dist, KROA100_N, c, full_reg, seeds, NS);
        auto rB = run_group<TSPLargeProblem>("B:-LNS", dummy, dist, KROA100_N, c, no_lns_reg, seeds, NS);
        dummy.destroy();

        printf("  Delta(A-B): Avg=%.1f (%.2f%%)  Best: A=%.0f B=%.0f\n",
               rA.avg - rB.avg, (rA.avg - rB.avg)/OPT*100, rA.best, rB.best);
        printf("  GapA=%.2f%%  GapB=%.2f%%\n\n",
               (rA.avg - OPT)/OPT*100, (rB.avg - OPT)/OPT*100);
        fflush(stdout);
    }

    // ============================================================
    // 3. TSP ch150 (n=150) — 大规模，原子算子难以收敛
    //    SA p64 g8k（长搜索，G/O 矩阵应积累足够信息）
    // ============================================================
    {
        float* dist = new float[CH150_N * CH150_N];
        compute_euc2d_dist(dist, CH150_coords, CH150_N);
        const float OPT = 6528.0f;
        printf("--- 3. TSP ch150 (n=%d, optimal=%.0f) | SA p64 g8k ---\n", CH150_N, OPT);
        fflush(stdout);

        SolverConfig c;
        c.pop_size = 64; c.max_gen = 8000; c.verbose = false;
        c.sa_temp_init = 100; c.sa_alpha = 0.9995f;
        c.use_aos = true;

        ProblemConfig pcfg;
        pcfg.encoding = EncodingType::Permutation;
        pcfg.dim1 = 1;
        SeqRegistry full_reg = build_seq_registry(pcfg.encoding, pcfg.dim1, 0.0f);
        SeqRegistry no_lns_reg = remove_lns(full_reg);

        TSPXLargeProblem dummy = TSPXLargeProblem::create(dist, CH150_N);
        auto rA = run_group<TSPXLargeProblem>("A:+LNS", dummy, dist, CH150_N, c, full_reg, seeds, NS);
        auto rB = run_group<TSPXLargeProblem>("B:-LNS", dummy, dist, CH150_N, c, no_lns_reg, seeds, NS);
        dummy.destroy();

        printf("  Delta(A-B): Avg=%.1f (%.2f%%)  Best: A=%.0f B=%.0f\n",
               rA.avg - rB.avg, (rA.avg - rB.avg)/OPT*100, rA.best, rB.best);
        printf("  GapA=%.2f%%  GapB=%.2f%%\n\n",
               (rA.avg - OPT)/OPT*100, (rB.avg - OPT)/OPT*100);
        fflush(stdout);
        delete[] dist;
    }

    // ============================================================
    // 4. TSP tsp225 (n=225) — 超大规模，LNS 应有明显优势
    //    SA p64 g10k（最长搜索，最大规模）
    // ============================================================
    {
        float* dist = new float[TSP225_N * TSP225_N];
        compute_euc2d_dist(dist, TSP225_coords, TSP225_N);
        const float OPT = 3916.0f;
        printf("--- 4. TSP tsp225 (n=%d, optimal=%.0f) | SA p64 g10k ---\n", TSP225_N, OPT);
        fflush(stdout);

        SolverConfig c;
        c.pop_size = 64; c.max_gen = 10000; c.verbose = false;
        c.sa_temp_init = 80; c.sa_alpha = 0.9995f;
        c.use_aos = true;

        ProblemConfig pcfg;
        pcfg.encoding = EncodingType::Permutation;
        pcfg.dim1 = 1;
        SeqRegistry full_reg = build_seq_registry(pcfg.encoding, pcfg.dim1, 0.0f);
        SeqRegistry no_lns_reg = remove_lns(full_reg);

        TSPXLargeProblem dummy = TSPXLargeProblem::create(dist, TSP225_N);
        auto rA = run_group<TSPXLargeProblem>("A:+LNS", dummy, dist, TSP225_N, c, full_reg, seeds, NS);
        auto rB = run_group<TSPXLargeProblem>("B:-LNS", dummy, dist, TSP225_N, c, no_lns_reg, seeds, NS);
        dummy.destroy();

        printf("  Delta(A-B): Avg=%.1f (%.2f%%)  Best: A=%.0f B=%.0f\n",
               rA.avg - rB.avg, (rA.avg - rB.avg)/OPT*100, rA.best, rB.best);
        printf("  GapA=%.2f%%  GapB=%.2f%%\n\n",
               (rA.avg - OPT)/OPT*100, (rB.avg - OPT)/OPT*100);
        fflush(stdout);
        delete[] dist;
    }

    // ============================================================
    // 汇总
    // ============================================================
    printf("================================================================\n");
    printf("  汇总：Delta < 0 表示 +LNS 更好，> 0 表示 -LNS 更好\n");
    printf("  预期：规模越大，LNS（尤其 GUIDED_REBUILD）优势越明显\n");
    printf("================================================================\n");
    fflush(stdout);

    return 0;
}
