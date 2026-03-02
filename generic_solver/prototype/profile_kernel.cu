/**
 * profile_kernel.cu - 用于 ncu profiling 的精简测试
 * 
 * 只跑 ch150 的 evolve_block_kernel，分别用 p64 和 p320
 * 每次只跑少量代数（100 代），让 ncu 能快速采集
 *
 * 用法：
 *   ncu --set full -o profile_p64  ./profile_kernel 64
 *   ncu --set full -o profile_p320 ./profile_kernel 320
 * 或快速模式：
 *   ncu --metrics ... ./profile_kernel 64
 */

#include "solver.cuh"
#include "tsp_large.cuh"
#include "tsplib_data.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void compute_dist(const float coords[][2], int n, float* dist) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float dx = coords[i][0]-coords[j][0], dy = coords[i][1]-coords[j][1];
            dist[i*n+j] = roundf(sqrtf(dx*dx+dy*dy));
        }
}

int main(int argc, char** argv) {
    int pop_size = (argc > 1) ? atoi(argv[1]) : 64;
    int num_gens = (argc > 2) ? atoi(argv[2]) : 100;
    
    printf("Profiling: ch150, pop=%d, gens=%d\n", pop_size, num_gens);
    
    using Problem = TSPLargeProblem;
    using Sol = Problem::Sol;
    
    const int N = CH150_N;
    float* dist = new float[N * N];
    compute_dist(CH150_coords, N, dist);
    
    auto prob = Problem::create(dist, N);
    ProblemConfig pcfg = prob.config();
    ObjConfig oc = make_obj_config(pcfg);
    const int block_threads = BLOCK_LEVEL_THREADS;
    
    // shared memory
    size_t prob_smem = prob.shared_mem_bytes();
    size_t total_smem = sizeof(Sol) + prob_smem + sizeof(MultiStepCandidate) * block_threads;
    {
        cudaDeviceProp prop; int dev;
        cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
        printf("GPU: %s, SM=%d\n", prop.name, prop.multiProcessorCount);
        printf("Sol size: %zu bytes\n", sizeof(Sol));
        printf("Prob smem: %zu bytes\n", prob_smem);
        printf("Total smem/block: %zu bytes\n", total_smem);
        
        if (total_smem > (size_t)prop.sharedMemPerBlock) {
            prob_smem = 0;
            total_smem = sizeof(Sol) + sizeof(MultiStepCandidate) * block_threads;
            printf("  -> prob data in global memory, smem=%zu\n", total_smem);
        }
        
        // occupancy 查询
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            evolve_block_kernel<Problem, Sol>,
            block_threads, total_smem);
        printf("Max active blocks/SM: %d (total capacity: %d)\n",
               max_blocks_per_sm, max_blocks_per_sm * prop.multiProcessorCount);
        printf("Actual blocks: %d → occupancy: %.0f%%\n",
               pop_size, (float)pop_size / (max_blocks_per_sm * prop.multiProcessorCount) * 100);
    }
    
    // 分配
    Population<Sol> pop;
    pop.allocate(pop_size, block_threads);
    cudaGetLastError(); cudaDeviceSynchronize();
    pop.init_rng(42, 256);
    pop.init_population(pcfg, 256);
    
    SeqRegistry seq_reg = build_seq_registry(pcfg.encoding, pcfg.dim1, pcfg.cross_row_prob);
    KStepConfig kstep = build_kstep_config();
    
    EvolveParams h_params;
    h_params.temp_start = 50.0f;
    h_params.gens_per_batch = 10;
    h_params.seq_reg = seq_reg;
    h_params.kstep = kstep;
    h_params.migrate_round = 0;
    
    EvolveParams* d_params = nullptr;
    cudaMalloc(&d_params, sizeof(EvolveParams));
    cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice);
    
    // 初始评估
    {
        size_t es = prob.shared_mem_bytes();
        int eg = calc_grid_size(pop_size, block_threads);
        evaluate_kernel<<<eg, block_threads, es>>>(prob, pop.d_solutions, pop_size, es);
    }
    cudaDeviceSynchronize();
    
    // 预热一次 evolve kernel（让 ncu 跳过）
    h_params.gens_per_batch = 10;
    cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice);
    evolve_block_kernel<<<pop_size, block_threads, total_smem>>>(
        prob, pop.d_solutions, pop_size,
        pcfg.encoding, pcfg.dim1, oc, pop.d_rng_states,
        0.998f, prob_smem,
        nullptr, nullptr, nullptr, 0, 0, 0, d_params);
    cudaDeviceSynchronize();
    
    // === 目标 kernel：这是 ncu 要采集的 ===
    printf("\n>>> Launching evolve_block_kernel: grid=%d, block=%d, smem=%zu, gens=%d\n",
           pop_size, block_threads, total_smem, num_gens);
    
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    
    h_params.gens_per_batch = num_gens;
    cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice);
    evolve_block_kernel<<<pop_size, block_threads, total_smem>>>(
        prob, pop.d_solutions, pop_size,
        pcfg.encoding, pcfg.dim1, oc, pop.d_rng_states,
        0.998f, prob_smem,
        nullptr, nullptr, nullptr, 0, 0, 0, d_params);
    
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    
    printf("Kernel time: %.2f ms\n", ms);
    printf("Per-gen: %.4f ms\n", ms / num_gens);
    printf("Per-gen-per-block: %.6f ms\n", ms / num_gens / pop_size);
    long long evals = (long long)pop_size * num_gens * block_threads;
    printf("Total evals: %.1fM, ns/eval: %.1f\n", evals/1e6, ms*1e6/evals);
    
    // 清理
    if (d_params) cudaFree(d_params);
    cudaFree(pop.d_solutions); cudaFree(pop.d_rng_states);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    prob.destroy();
    delete[] dist;
    
    return 0;
}
