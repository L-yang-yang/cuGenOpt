/**
 * solver.cuh - Main solve loop
 * 
 * v2.0: Block-level architecture refactor
 *   - 1 block = 1 solution (neighborhood parallelism)
 *   - Solution lives in shared memory
 *   - Each generation: K threads each propose a candidate move + evaluate delta -> reduce to best -> thread 0 applies
 *   - Crossover uses a simplified path for now (thread 0 runs crossover, others wait)
 *   - Migration / elite injection remain single-thread kernels (global memory)
 *
 * Required Problem interface:
 *   size_t shared_mem_bytes() const;
 *   __device__ void load_shared(char* smem, int tid, int bsz);
 *   __device__ void evaluate(Sol& sol) const;
 */

#pragma once
#include "types.cuh"
#include "population.cuh"
#include "operators.cuh"
#include "relation_matrix.cuh"
#include "cuda_utils.cuh"
#include "init_selection.cuh"
#include "init_heuristic.cuh"
#include <cmath>

// ============================================================
// Compile-time constants
// ============================================================
constexpr int BLOCK_LEVEL_THREADS = 128;  // Default threads per block for block-level architecture

// ============================================================
// EvolveParams — CUDA Graph mutable parameters (device memory)
// ============================================================
// Per-batch parameters are packed into one struct;
// evolve_block_kernel reads via pointer; CUDA Graph capture binds the pointer.
// Before each replay, only cudaMemcpy this device memory block.

struct EvolveParams {
    float       temp_start;
    int         gens_per_batch;
    SeqRegistry seq_reg;
    KStepConfig kstep;
    int         migrate_round;
    ObjConfig   oc;
};

// ============================================================
// Helpers: cooperative load/store Solution (shared memory ↔ global memory)
// ============================================================

template<typename Sol>
__device__ inline void cooperative_load_sol(Sol& dst, const Sol& src,
                                             int tid, int num_threads) {
    // Cooperative copy of entire Solution struct in int-sized chunks
    const int* src_ptr = reinterpret_cast<const int*>(&src);
    int* dst_ptr = reinterpret_cast<int*>(&dst);
    constexpr int n_ints = (sizeof(Sol) + sizeof(int) - 1) / sizeof(int);
    for (int i = tid; i < n_ints; i += num_threads)
        dst_ptr[i] = src_ptr[i];
}

template<typename Sol>
__device__ inline void cooperative_store_sol(Sol& dst, const Sol& src,
                                              int tid, int num_threads) {
    cooperative_load_sol(dst, src, tid, num_threads);  // Same copy logic
}

// ============================================================
// Kernel 1: Initial evaluation (once; 1 block = 1 solution)
// ============================================================

template<typename Problem, typename Sol>
__global__ void evaluate_kernel(Problem prob, Sol* pop, int pop_size,
                                 size_t smem_size) {
    extern __shared__ char smem[];
    Problem lp = prob;
    if (smem_size > 0) { lp.load_shared(smem, threadIdx.x, blockDim.x); __syncthreads(); }
    
    // One-thread-per-solution initial evaluation (simple; called once)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pop_size) lp.evaluate(pop[tid]);
}

// ============================================================
// Kernel 2: Block-level batched evolution (neighborhood parallelism)
// ============================================================
//
// Per-generation flow:
//   1. Each of K threads generates one candidate move
//   2. Each thread evaluates delta for its move (does not modify sol in shared memory)
//   3. Block reduction: pick move with smallest delta
//   4. Thread 0 accepts or rejects (SA / HC)
//   5. Thread 0 applies best move and updates sol
//   6. __syncthreads() so all threads see updated sol
//
// Solution and Problem data live in shared memory

// ============================================================
// MultiStepCandidate — multi-step result (for reduction)
// ============================================================
struct MultiStepCandidate {
    float delta;
    float new_penalty;
    int   seq_indices[MAX_K];
    int   k_steps;
    int   winner_tid;
};

template<typename Problem, typename Sol>
__global__ void evolve_block_kernel(Problem prob, Sol* pop, int pop_size,
                                     EncodingType encoding, int dim1,
                                     ObjConfig oc_legacy,
                                     curandState* rng_states,
                                     float alpha,
                                     size_t prob_smem_size,
                                     AOSStats* d_aos_stats,
                                     const float* d_G,
                                     const float* d_O,
                                     int rel_N,
                                     int val_lb,
                                     int val_ub,
                                     const EvolveParams* d_params) {
    extern __shared__ char smem[];
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    if (bid >= pop_size) return;
    
    const int gens_per_batch = d_params->gens_per_batch;
    const SeqRegistry seq_reg = d_params->seq_reg;
    const KStepConfig kstep = d_params->kstep;
    const float temp_start = d_params->temp_start;
    const ObjConfig oc = d_params->oc;
    
    // --- shared memory layout ---
    // [0 .. sizeof(Sol)-1]                              : Solution
    // [sizeof(Sol) .. sizeof(Sol)+prob_smem-1]          : Problem data
    // [after .. ]                                       : MultiStepCandidate[num_threads] reduction workspace
    // [after .. ]                                       : AOSStats (if enabled)
    
    Sol* s_sol = reinterpret_cast<Sol*>(smem);
    char* prob_smem_ptr = smem + sizeof(Sol);
    MultiStepCandidate* s_cands = reinterpret_cast<MultiStepCandidate*>(
        smem + sizeof(Sol) + prob_smem_size);
    
    // AOS stats (after MultiStepCandidate array)
    AOSStats* s_aos = nullptr;
    if (d_aos_stats) {
        s_aos = reinterpret_cast<AOSStats*>(
            smem + sizeof(Sol) + prob_smem_size + sizeof(MultiStepCandidate) * num_threads);
        // Thread 0 initializes AOS counters
        if (tid == 0) {
            for (int i = 0; i < MAX_SEQ; i++) {
                s_aos->usage[i] = 0;
                s_aos->improvement[i] = 0;
            }
            for (int i = 0; i < MAX_K; i++) {
                s_aos->k_usage[i] = 0;
                s_aos->k_improvement[i] = 0;
            }
        }
    }
    
    // Load Problem data into shared memory
    Problem lp = prob;
    if (prob_smem_size > 0) {
        lp.load_shared(prob_smem_ptr, tid, num_threads);
    }
    
    // Cooperatively load Solution into shared memory
    cooperative_load_sol(*s_sol, pop[bid], tid, num_threads);
    __syncthreads();
    
    int rng_idx = bid * num_threads + tid;
    curandState rng = rng_states[rng_idx];
    
    float temp = temp_start;
    
    for (int g = 0; g < gens_per_batch; g++) {
        // ============================================================
        // Step 1: Each thread independently samples K steps + K sequences on local copy
        // ============================================================
        
        // Sample K (step count): weighted by kstep.weights
        float kr = curand_uniform(&rng);
        int my_k = 1;  // default K=1
        {
            float cum = 0.0f;
            for (int i = 0; i < MAX_K; i++) {
                cum += kstep.weights[i];
                if (kr < cum) { my_k = i + 1; break; }
            }
        }
        
        // Copy sol in local memory, apply K moves
        Sol local_sol = *s_sol;
        MultiStepCandidate my_cand;
        my_cand.k_steps = my_k;
        my_cand.winner_tid = tid;
        for (int i = 0; i < MAX_K; i++) {
            my_cand.seq_indices[i] = -1;
        }
        
        bool all_noop = true;
        for (int step = 0; step < my_k; step++) {
            int seq_idx = -1;
            bool changed = ops::sample_and_execute(
                seq_reg, local_sol, dim1, encoding, &rng, seq_idx,
                d_G, d_O, rel_N, val_lb, val_ub,
                static_cast<const void*>(&lp));
            my_cand.seq_indices[step] = seq_idx;
            if (changed) all_noop = false;
        }
        
        // Step 2: Evaluate final delta (after K steps vs original sol)
        if (all_noop) {
            my_cand.delta = 1e30f;
            my_cand.new_penalty = s_sol->penalty;
        } else {
            lp.evaluate(local_sol);
            float old_scalar = obj_scalar(s_sol->objectives, oc);
            float new_scalar = obj_scalar(local_sol.objectives, oc);
            
            bool old_feasible = (s_sol->penalty <= 0.0f);
            bool new_feasible = (local_sol.penalty <= 0.0f);
            
            if (new_feasible && !old_feasible) {
                my_cand.delta = -1e20f;
            } else if (!new_feasible && old_feasible) {
                my_cand.delta = 1e20f;
            } else if (!new_feasible && !old_feasible) {
                my_cand.delta = local_sol.penalty - s_sol->penalty;
            } else {
                my_cand.delta = new_scalar - old_scalar;
            }
            my_cand.new_penalty = local_sol.penalty;
        }
        
        s_cands[tid] = my_cand;
        __syncthreads();
        
        // Step 3: Parallel reduction in block to find candidate with smallest delta
        for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_cands[tid + stride].delta < s_cands[tid].delta)
                    s_cands[tid] = s_cands[tid + stride];
            }
            __syncthreads();
        }
        
        // Step 4: Thread 0 decides accept/reject
        if (tid == 0) {
            MultiStepCandidate& best = s_cands[0];
            bool has_valid = (best.delta < 1e29f);
            
            if (has_valid) {
                bool improved = (best.delta < 0.0f);
                
                bool accept;
                if (improved) {
                    accept = true;
                } else if (temp > 0.0f && s_sol->penalty <= 0.0f && best.new_penalty <= 0.0f) {
                    accept = curand_uniform(&rng) < expf(-best.delta / temp);
                } else {
                    accept = false;
                }
                
                if (accept) {
                    // AOS stats: K layer + operator layer
                    if (s_aos) {
                        int ki = best.k_steps - 1;
                        if (ki >= 0 && ki < MAX_K) {
                            s_aos->k_usage[ki]++;
                            if (improved) s_aos->k_improvement[ki]++;
                        }
                        for (int step = 0; step < best.k_steps; step++) {
                            int si = best.seq_indices[step];
                            if (si >= 0 && si < seq_reg.count) {
                                s_aos->usage[si]++;
                                if (improved) s_aos->improvement[si]++;
                            }
                        }
                    }
                    // Signal: keep winner_tid as-is (accept)
                } else {
                    s_cands[0].winner_tid = -1;  // Signal: reject
                }
            } else {
                s_cands[0].winner_tid = -1;  // Signal: no valid candidate
            }
            
            temp *= alpha;
        }
        __syncthreads();
        
        // Step 5: Winner thread writes local_sol to s_sol
        int winner = s_cands[0].winner_tid;
        if (winner >= 0 && tid == winner) {
            *s_sol = local_sol;
        }
        __syncthreads();
    }
    
    // Write Solution back to global memory
    cooperative_store_sol(pop[bid], *s_sol, tid, num_threads);
    
    // Write AOS stats back to global memory
    if (d_aos_stats && tid == 0) {
        d_aos_stats[bid] = *s_aos;
    }
    
    // Save RNG state
    rng_states[rng_idx] = rng;
}

// ============================================================
// Kernel 2b: Block-level crossover
// ============================================================
// Simplified: thread 0 runs crossover; others cooperative load/store
// Phase 3 may add multi-thread cooperative crossover

template<typename Problem, typename Sol>
__global__ void crossover_block_kernel(Problem prob, Sol* pop, int pop_size,
                                        EncodingType encoding, int dim1,
                                        ObjConfig oc,
                                        curandState* rng_states,
                                        float crossover_rate,
                                        size_t prob_smem_size,
                                        int total_elements = 0) {
    extern __shared__ char smem[];
    
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int K = blockDim.x;
    
    if (bid >= pop_size) return;
    
    // Shared memory layout: Sol + Problem data
    Sol* s_sol = reinterpret_cast<Sol*>(smem);
    char* prob_smem_ptr = smem + sizeof(Sol);
    
    Problem lp = prob;
    if (prob_smem_size > 0) {
        lp.load_shared(prob_smem_ptr, tid, K);
    }
    
    cooperative_load_sol(*s_sol, pop[bid], tid, K);
    __syncthreads();
    
    // Thread 0 runs crossover
    if (tid == 0) {
        int rng_idx = bid * K;
        curandState rng = rng_states[rng_idx];
        
        if (curand_uniform(&rng) < crossover_rate) {
            int c1 = rand_int(&rng, pop_size);
            int c2 = rand_int(&rng, pop_size - 1);
            if (c2 >= c1) c2++;
            int mate_idx = is_better(pop[c1], pop[c2], oc) ? c1 : c2;
            
            if (mate_idx != bid) {
                const Sol& mate = pop[mate_idx];
                Sol child;
                bool did_crossover = false;
                
                if (encoding == EncodingType::Permutation) {
                    int te = total_elements;
                    if (te <= 0) te = s_sol->dim2_sizes[0];
                    ops::perm_ox_crossover(child, *s_sol, mate, dim1, te, &rng);
                    did_crossover = true;
                } else if (encoding == EncodingType::Binary) {
                    ops::uniform_crossover(child, *s_sol, mate, dim1, &rng);
                    did_crossover = true;
                }
                
                if (did_crossover) {
                    lp.evaluate(child);
                    if (is_better(child, *s_sol, oc)) {
                        *s_sol = child;
                    }
                }
            }
        }
        
        rng_states[rng_idx] = rng;
    }
    __syncthreads();
    
    // Write back (possibly updated by crossover)
    cooperative_store_sol(pop[bid], *s_sol, tid, K);
}

// ============================================================
// Kernel 3: Inter-island migration (unchanged; single-thread kernel)
// ============================================================

template<typename Sol>
__device__ inline int find_worst_in_island(const Sol* pop, int base, int island_size,
                                            const ObjConfig& oc) {
    int worst = base;
    for (int i = base + 1; i < base + island_size; i++)
        if (is_better(pop[worst], pop[i], oc)) worst = i;
    return worst;
}

constexpr int MAX_ISLANDS = 64;

template<typename Sol>
__global__ void migrate_kernel(Sol* pop, int pop_size, int island_size,
                                ObjConfig oc,
                                MigrateStrategy strategy,
                                const EvolveParams* d_params) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int round = d_params->migrate_round;
    int num_islands = pop_size / island_size;
    if (num_islands > MAX_ISLANDS) num_islands = MAX_ISLANDS;
    if (num_islands <= 1) return;
    
    int candidates[MAX_ISLANDS];
    for (int isle = 0; isle < num_islands; isle++) {
        int base = isle * island_size;
        int best = base;
        for (int i = base + 1; i < base + island_size; i++)
            if (is_better(pop[i], pop[best], oc)) best = i;
        candidates[isle] = best;
    }
    
    int topn[MAX_ISLANDS];
    if (strategy == MigrateStrategy::TopN || strategy == MigrateStrategy::Hybrid) {
        bool selected[MAX_ISLANDS] = {};
        for (int t = 0; t < num_islands; t++) {
            int best_c = -1;
            for (int c = 0; c < num_islands; c++) {
                if (selected[c]) continue;
                if (best_c < 0 || is_better(pop[candidates[c]], pop[candidates[best_c]], oc))
                    best_c = c;
            }
            topn[t] = candidates[best_c];
            selected[best_c] = true;
        }
        for (int i = 0; i < num_islands; i++) {
            int dst_isle = (i + round) % num_islands;
            int dst_base = dst_isle * island_size;
            int worst = find_worst_in_island(pop, dst_base, island_size, oc);
            if (is_better(pop[topn[i]], pop[worst], oc))
                pop[worst] = pop[topn[i]];
        }
    }
    
    if (strategy == MigrateStrategy::Ring || strategy == MigrateStrategy::Hybrid) {
        for (int isle = 0; isle < num_islands; isle++) {
            int dst_isle = (isle + 1) % num_islands;
            int dst_base = dst_isle * island_size;
            int worst = find_worst_in_island(pop, dst_base, island_size, oc);
            int src = candidates[isle];
            if (is_better(pop[src], pop[worst], oc))
                pop[worst] = pop[src];
        }
    }
}

// ============================================================
// Kernel 4: Elite injection (unchanged)
// ============================================================

template<typename Sol>
__global__ void elite_inject_kernel(Sol* pop, int pop_size,
                                     Sol* global_best, ObjConfig oc) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++)
        if (is_better(pop[i], pop[best_idx], oc)) best_idx = i;
    
    if (is_better(pop[best_idx], *global_best, oc))
        *global_best = pop[best_idx];
    
    int worst_idx = 0;
    for (int i = 1; i < pop_size; i++)
        if (is_better(pop[worst_idx], pop[i], oc)) worst_idx = i;
    
    if (is_better(*global_best, pop[worst_idx], oc))
        pop[worst_idx] = *global_best;
}

// ============================================================
// v5.0: Multi-GPU coordination — inject external solutions into islands
// ============================================================

template<typename Sol>
__global__ void inject_to_islands_kernel(Sol* pop, int pop_size, int island_size,
                                          const Sol* inject_solutions, int num_inject,
                                          MultiGpuInjectMode mode, ObjConfig oc) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (num_inject == 0) return;
    
    int num_islands = pop_size / island_size;
    if (num_islands == 0) return;
    
    // Number of islands to inject into depends on mode
    int islands_to_inject = 0;
    if (mode == MultiGpuInjectMode::OneIsland) {
        islands_to_inject = 1;
    } else if (mode == MultiGpuInjectMode::HalfIslands) {
        islands_to_inject = (num_islands + 1) / 2;
    } else {  // AllIslands
        islands_to_inject = num_islands;
    }
    
    // Place each injected solution at worst slot of an island
    for (int i = 0; i < islands_to_inject && i < num_inject; i++) {
        int target_isle = i % num_islands;
        int base = target_isle * island_size;
        
        // Find worst solution on this island
        int worst = find_worst_in_island(pop, base, island_size, oc);
        
        // Replace if injection is better
        if (is_better(inject_solutions[i], pop[worst], oc)) {
            pop[worst] = inject_solutions[i];
        }
    }
}

// ============================================================
// v5.0 plan B3: inject_check_kernel — passive injection check
// ============================================================
// During migrate, GPU checks InjectBuffer; if new solution exists, inject into
// target islands based on MultiGpuInjectMode.
//
// Design notes:
// 1. Single thread (thread 0 of block 0) — serial over target islands (count is small)
// 2. atomicExch reads flag and clears it so each solution is handled once
// 3. Inject mode: OneIsland (island 0), HalfIslands (random half), AllIslands (all)
// 4. Optional: if inject_buf is nullptr, skip (single-GPU unaffected)

template<typename Sol>
__global__ void inject_check_kernel(Sol* pop, int pop_size, int island_size,
                                     InjectBuffer<Sol>* inject_buf, ObjConfig oc,
                                     MultiGpuInjectMode mode) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (inject_buf == nullptr) return;

    int flag = atomicExch(inject_buf->d_flag, 0);
    if (flag != 1) return;

    Sol inject_sol = *(inject_buf->d_solution);

    int num_islands = pop_size / island_size;
    if (num_islands == 0) return;

    if (mode == MultiGpuInjectMode::OneIsland) {
        int worst = find_worst_in_island(pop, 0, island_size, oc);
        if (is_better(inject_sol, pop[worst], oc))
            pop[worst] = inject_sol;

    } else if (mode == MultiGpuInjectMode::AllIslands) {
        for (int i = 0; i < num_islands; i++) {
            int worst = find_worst_in_island(pop, i * island_size, island_size, oc);
            if (is_better(inject_sol, pop[worst], oc))
                pop[worst] = inject_sol;
        }

    } else {  // HalfIslands — randomly select num_islands/2 islands
        int half = (num_islands + 1) / 2;
        unsigned seed = (unsigned)clock();
        for (int count = 0; count < half; count++) {
            seed = seed * 1664525u + 1013904223u;  // LCG
            int isle = (int)(seed % (unsigned)num_islands);
            int worst = find_worst_in_island(pop, isle * island_size, island_size, oc);
            if (is_better(inject_sol, pop[worst], oc))
                pop[worst] = inject_sol;
        }
    }
}

// ============================================================
// solve<Problem>: main loop (block-level architecture)
// ============================================================

using RegistryCallback = void(*)(SeqRegistry&);

template<typename Problem>
SolveResult<typename Problem::Sol> solve(Problem& prob, const SolverConfig& cfg,
                                          const typename Problem::Sol* init_solutions = nullptr,
                                          int num_init_solutions = 0,
                                          RegistryCallback custom_registry_fn = nullptr,
                                          InjectBuffer<typename Problem::Sol>* inject_buf = nullptr,
                                          typename Problem::Sol** d_global_best_out = nullptr) {
    using Sol = typename Problem::Sol;
    ProblemConfig pcfg = prob.config();
    SolveResult<Sol> result;
    
    bool use_sa = cfg.sa_temp_init > 0.0f;
    bool use_crossover = cfg.crossover_rate > 0.0f;
    bool use_aos = cfg.use_aos;
    bool use_time_limit = cfg.time_limit_sec > 0.0f;
    bool use_stagnation = cfg.stagnation_limit > 0;
    
    // Block-level parameters
    const int block_threads = BLOCK_LEVEL_THREADS;  // 128 threads/block
    
    // --- 0. Shared memory sizing (before pop_size; used for occupancy query) ---
    size_t prob_smem = prob.shared_mem_bytes();
    // v3.1: reduction workspace is MultiStepCandidate (K-step moves + seq_indices)
    size_t total_smem = sizeof(Sol) + prob_smem + sizeof(MultiStepCandidate) * block_threads;
    if (use_aos) total_smem += sizeof(AOSStats);
    
    // Query GPU device properties
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Try to raise shared memory cap (V100: 96KB, A100: 164KB, etc.)
    size_t max_smem = (size_t)prop.sharedMemPerBlock;
    if (total_smem > 48 * 1024) {
        cudaError_t err1 = cudaFuncSetAttribute(
            evolve_block_kernel<Problem, Sol>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)total_smem);
        cudaError_t err2 = cudaFuncSetAttribute(
            crossover_block_kernel<Problem, Sol>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)total_smem);
        if (err1 == cudaSuccess && err2 == cudaSuccess) {
            max_smem = total_smem;
        }
    }
    
    // Check shared memory limit
    bool smem_overflow = false;
    if (total_smem > max_smem) {
        smem_overflow = (prob_smem > 0);
        prob_smem = 0;
        total_smem = sizeof(Sol) + sizeof(MultiStepCandidate) * block_threads;
        if (use_aos) total_smem += sizeof(AOSStats);
    }
    
    // --- 0b. Determine pop_size (auto or user) ---
    int pop_size = cfg.pop_size;
    bool auto_pop = (pop_size <= 0);
    
    if (auto_pop) {
        // Query occupancy: how many blocks per SM
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            evolve_block_kernel<Problem, Sol>,
            block_threads,
            total_smem);
        
        int full_capacity = max_blocks_per_sm * prop.multiProcessorCount;
        
        if (prob_smem > 0) {
            // Problem data in shared memory → no L2 pressure; fill SMs
            pop_size = full_capacity;
        } else {
            // Problem data in global memory → estimate concurrency from L2 size
            //
            // Model: pop = L2_size / working_set_bytes
            //   All blocks read same read-only data; L2/ws approximates cache-supported concurrency
            //
            // SM floor policy: if L2/ws >= sm_min/2, raise to sm_min (trade some cache pressure for diversity)
            //   ch150: L2/ws=70, sm_min=128 -> 70 >= 64 -> raise to 128 (diversity first)
            //   pcb442: L2/ws=8, sm_min=128 -> 8 < 64 -> do not raise (avoid thrashing)
            
            size_t ws = prob.working_set_bytes();
            if (ws > 0) {
                int l2_pop = (int)((size_t)prop.l2CacheSize / ws);
                pop_size = (l2_pop < full_capacity) ? l2_pop : full_capacity;
            } else {
                pop_size = full_capacity / 4;
            }
            
            int sm_min = 1;
            while (sm_min < prop.multiProcessorCount) sm_min *= 2;
            if (pop_size < sm_min) {
                bool l2_can_afford = (ws == 0) ||
                    ((size_t)prop.l2CacheSize / ws >= (size_t)sm_min / 2);
                if (l2_can_afford) pop_size = sm_min;
            }
        }
        
        // Round down to power of 2 (warp alignment, reduction-friendly, island divisibility)
        {
            int p = 1;
            while (p * 2 <= pop_size) p *= 2;
            pop_size = p;
        }
        
        // Absolute floor: 32 (at least 1 island x 32 individuals)
        if (pop_size < 32) pop_size = 32;
    }
    
    // Adaptive island count (when num_islands=0)
    int num_islands = cfg.num_islands;
    if (num_islands == 0) {
        // Policy: at least 32 individuals per island, at most 8 islands
        // pop < 64   -> 1 island (pure HC)
        // 64-127     -> 2 islands
        // 128-255    -> 4 islands
        // 256-511    -> 8 islands
        // >= 512     -> 8 islands
        if (pop_size < 64) {
            num_islands = 1;
        } else if (pop_size < 128) {
            num_islands = 2;
        } else if (pop_size < 256) {
            num_islands = 4;
        } else {
            num_islands = 8;
        }
    }
    
    bool use_islands = num_islands > 1;
    int island_size = use_islands ? pop_size / num_islands : pop_size;
    
    if (cfg.verbose) {
        const char* enc_name = pcfg.encoding == EncodingType::Permutation ? "Perm" 
                             : pcfg.encoding == EncodingType::Binary ? "Bin" : "Int";
        const char* strat_name = 
            cfg.migrate_strategy == MigrateStrategy::Ring ? "Ring" :
            cfg.migrate_strategy == MigrateStrategy::TopN ? "TopN" : "Hybrid";
        printf("\n[GenSolver v2.0 Block] %s%s [%d][%d] pop=%d%s gen=%d blk=%d",
               enc_name, pcfg.row_mode == RowMode::Partition ? "/Part" : "",
               pcfg.dim1, pcfg.row_mode == RowMode::Partition ? pcfg.total_elements : pcfg.dim2_default,
               pop_size, auto_pop ? "(auto)" : "",
               cfg.max_gen, block_threads);
        if (auto_pop) {
            size_t ws = prob.working_set_bytes();
            if (prob_smem > 0) {
                printf("\n  [AUTO] GPU=%s SM=%d strategy=full(smem) → pop=%d",
                       prop.name, prop.multiProcessorCount, pop_size);
            } else {
                printf("\n  [AUTO] GPU=%s SM=%d L2=%dKB ws=%zuKB → pop=%d",
                       prop.name, prop.multiProcessorCount,
                       prop.l2CacheSize / 1024, ws / 1024, pop_size);
            }
        }
        if (smem_overflow) {
            printf("\n  [WARN] Shared memory overflow, problem data stays in global memory");
        }
        if (use_islands) {
            if (cfg.num_islands == 0) {
                printf(" isl=%dx%d/%s(auto)", num_islands, island_size, strat_name);
            } else {
                printf(" isl=%dx%d/%s", num_islands, island_size, strat_name);
            }
        }
        if (use_sa)      printf(" SA=%.0f/%.4f", cfg.sa_temp_init, cfg.sa_alpha);
        if (use_crossover) printf(" CX=%.0f%%", cfg.crossover_rate * 100.0f);
        if (use_aos) printf(" AOS");
        if (use_time_limit) printf(" T=%.1fs", cfg.time_limit_sec);
        if (use_stagnation) printf(" stag=%d", cfg.stagnation_limit);
        if (num_init_solutions > 0) printf(" init=%d", num_init_solutions);
        if (cfg.use_cuda_graph) printf(" GRAPH");
        printf(" seed=%u\n", cfg.seed);
    }
    
    // --- 1. Allocation ---
    // Crossover stack needs (thread 0 builds child in local memory)
    if (use_crossover) {
        size_t ox_arrays = Sol::DIM1 * Sol::DIM2 * sizeof(bool)
                         + 512 * sizeof(bool)
                         + 512 * sizeof(int);
        size_t need = sizeof(Sol) + ox_arrays + 512;
        if (need > 1024) cudaDeviceSetLimit(cudaLimitStackSize, need);
    }
    
    ObjConfig oc = make_obj_config(pcfg);
    
    // --- 1b. Sample-and-select initialization ---
    int oversample = cfg.init_oversample;
    if (oversample < 1) oversample = 1;
    int candidate_size = pop_size * oversample;
    bool do_oversample = (oversample > 1);
    
    Population<Sol> pop;
    
    if (do_oversample) {
        // Generate K x pop_size candidate solutions
        Population<Sol> candidates;
        candidates.allocate(candidate_size, block_threads);
        candidates.init_rng(cfg.seed, 256);
        candidates.init_population(pcfg, 256);
        
        // Inject heuristic initial solutions (replace tail of candidate pool)
        if (pcfg.encoding == EncodingType::Permutation) {
            HeuristicMatrix heur_mats[8];
            int num_mats = prob.heuristic_matrices(heur_mats, 8);
            if (num_mats > 0) {
                bool is_partition = (pcfg.row_mode == RowMode::Partition);
                auto heur_sols = heuristic_init::build_from_matrices<Sol>(
                    heur_mats, num_mats, pcfg.dim1, pcfg.dim2_default, pcfg.encoding,
                    is_partition, pcfg.total_elements);
                int inject = (int)heur_sols.size();
                if (inject > candidate_size / 8) inject = candidate_size / 8;
                if (inject > 0) {
                    CUDA_CHECK(cudaMemcpy(
                        candidates.d_solutions + candidate_size - inject,
                        heur_sols.data(), sizeof(Sol) * inject,
                        cudaMemcpyHostToDevice));
                    if (cfg.verbose) {
                        printf("  [INIT] injected %d heuristic solutions into candidate pool\n", inject);
                    }
                }
            }
        }
        
        // Evaluate all candidates on GPU
        {
            size_t eval_smem = prob.shared_mem_bytes();
            if (eval_smem > 48 * 1024) {
                cudaFuncSetAttribute(evaluate_kernel<Problem, Sol>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)eval_smem);
            }
            int eval_grid = calc_grid_size(candidate_size, block_threads);
            evaluate_kernel<<<eval_grid, block_threads, eval_smem>>>(
                prob, candidates.d_solutions, candidate_size, eval_smem);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Download all candidates to host
        Sol* h_candidates = new Sol[candidate_size];
        CUDA_CHECK(cudaMemcpy(h_candidates, candidates.d_solutions,
                              sizeof(Sol) * candidate_size, cudaMemcpyDeviceToHost));
        
        // Build candidate metadata
        std::vector<init_sel::CandidateInfo> cand_info(candidate_size);
        for (int i = 0; i < candidate_size; i++) {
            cand_info[i].idx = i;
            cand_info[i].penalty = h_candidates[i].penalty;
            cand_info[i].rank = 0;
            cand_info[i].crowding = 0.0f;
            cand_info[i].selected = false;
            for (int m = 0; m < oc.num_obj; m++) {
                cand_info[i].objs[m] = normalize_obj(
                    h_candidates[i].objectives[m], oc.dirs[m]);
            }
        }
        
        // Compute objective importance
        float importance[MAX_OBJ];
        compute_importance(oc, importance);
        
        // Pure-random quota (floor)
        int num_random = (int)(pop_size * cfg.init_random_ratio);
        if (num_random < 1) num_random = 1;
        if (num_random > pop_size / 2) num_random = pop_size / 2;
        
        // Selection
        std::vector<int> selected;
        if (oc.num_obj == 1) {
            selected = init_sel::top_n_select(cand_info, pop_size, num_random);
        } else {
            selected = init_sel::nsga2_select(cand_info, oc.num_obj, importance,
                                               pop_size, num_random);
        }
        
        // Allocate final population
        pop.allocate(pop_size, block_threads);
        // Could reuse candidate RNG state (first pop_size entries)
        // Re-init RNG is safer (candidate RNGs were already used)
        pop.init_rng(cfg.seed + 1, 256);
        
        // Upload selected solutions to front of population
        int num_selected = (int)selected.size();
        for (int i = 0; i < num_selected; i++) {
            CUDA_CHECK(cudaMemcpy(pop.d_solutions + i,
                                  candidates.d_solutions + selected[i],
                                  sizeof(Sol), cudaMemcpyDeviceToDevice));
        }
        
        // Remaining slots (pure-random floor): fill from unselected candidates
        // Simple approach: use later candidates that were not selected
        if (num_selected < pop_size) {
            int fill_idx = num_selected;
            for (int i = 0; i < candidate_size && fill_idx < pop_size; i++) {
                if (!cand_info[i].selected) {
                    CUDA_CHECK(cudaMemcpy(pop.d_solutions + fill_idx,
                                          candidates.d_solutions + i,
                                          sizeof(Sol), cudaMemcpyDeviceToDevice));
                    fill_idx++;
                }
            }
        }
        
        if (cfg.verbose) {
            // Compare mean quality of selected vs all candidates
            float sel_avg = 0.0f, all_avg = 0.0f;
            for (int i = 0; i < candidate_size; i++) all_avg += cand_info[i].objs[0];
            all_avg /= candidate_size;
            for (int i = 0; i < num_selected; i++) sel_avg += cand_info[selected[i]].objs[0];
            if (num_selected > 0) sel_avg /= num_selected;
            
            const char* method = (oc.num_obj > 1) ? "NSGA-II" : "top-N";
            printf("  [INIT] oversample=%dx → %d candidates, %s select %d + %d random",
                   oversample, candidate_size, method, num_selected,
                   pop_size - num_selected);
            printf(" (obj0 avg: %.1f → %.1f, %.1f%% better)\n",
                   all_avg, sel_avg,
                   all_avg != 0.0f ? (1.0f - sel_avg / all_avg) * 100.0f : 0.0f);
        }
        
        delete[] h_candidates;
        // candidates dtor frees GPU memory
    } else {
        // oversample=1: pure random, same as before
        pop.allocate(pop_size, block_threads);
        pop.init_rng(cfg.seed, 256);
        pop.init_population(pcfg, 256);
    }
    
    // --- 1c. Inject user-provided initial solutions ---
    // Policy: validate -> valid solutions replace population tail (keep oversample winners at front)
    if (init_solutions && num_init_solutions > 0) {
        int max_inject = pop_size / 16;  // at most ~6% of population (diversity)
        if (max_inject < 1) max_inject = 1;
        if (max_inject > 16) max_inject = 16;  // hard cap
        int want = num_init_solutions;
        if (want > max_inject) want = max_inject;
        
        int injected = 0;
        for (int i = 0; i < want; i++) {
            const Sol& s = init_solutions[i];
            bool valid = true;
            
            // Basic dimension checks
            for (int r = 0; r < pcfg.dim1 && valid; r++) {
                if (s.dim2_sizes[r] < 0 || s.dim2_sizes[r] > Sol::DIM2) {
                    valid = false; break;
                }
            }
            
            // Encoding-specific checks
            if (valid && pcfg.encoding == EncodingType::Permutation) {
                if (pcfg.row_mode == RowMode::Partition) {
                    // Partition mode: no duplicate elements across rows; total = total_elements
                    bool seen[512] = {};
                    int total = 0;
                    for (int r = 0; r < pcfg.dim1 && valid; r++) {
                        for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                            int v = s.data[r][c];
                            if (v < 0 || v >= pcfg.total_elements) { valid = false; break; }
                            if (v < 512 && seen[v]) { valid = false; break; }
                            if (v < 512) seen[v] = true;
                            total++;
                        }
                    }
                    if (valid && total != pcfg.total_elements) valid = false;
                } else if (pcfg.perm_repeat_count > 1) {
                    // Multiset permutation: each value in [0, N) appears repeat_count times per row
                    int R = pcfg.perm_repeat_count;
                    int N = pcfg.dim2_default / R;
                    for (int r = 0; r < pcfg.dim1 && valid; r++) {
                        if (s.dim2_sizes[r] != pcfg.dim2_default) { valid = false; break; }
                        int cnt[512] = {};
                        for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                            int v = s.data[r][c];
                            if (v < 0 || v >= N) { valid = false; break; }
                            if (v < 512) cnt[v]++;
                        }
                        if (valid) {
                            for (int v = 0; v < N && v < 512 && valid; v++)
                                if (cnt[v] != R) valid = false;
                        }
                    }
                } else {
                    // Standard permutation: each row is a permutation of [0, dim2_default)
                    for (int r = 0; r < pcfg.dim1 && valid; r++) {
                        if (s.dim2_sizes[r] != pcfg.dim2_default) { valid = false; break; }
                        bool seen[512] = {};
                        for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                            int v = s.data[r][c];
                            if (v < 0 || v >= pcfg.dim2_default) { valid = false; break; }
                            if (v < 512 && seen[v]) { valid = false; break; }
                            if (v < 512) seen[v] = true;
                        }
                    }
                }
            } else if (valid && pcfg.encoding == EncodingType::Binary) {
                for (int r = 0; r < pcfg.dim1 && valid; r++) {
                    for (int c = 0; c < s.dim2_sizes[r] && valid; c++) {
                        if (s.data[r][c] != 0 && s.data[r][c] != 1) { valid = false; break; }
                    }
                }
            }
            
            if (valid) {
                // Inject at population tail (fill from end; keep oversample winners at front)
                int target_idx = pop_size - 1 - injected;
                CUDA_CHECK(cudaMemcpy(pop.d_solutions + target_idx, &s,
                                      sizeof(Sol), cudaMemcpyHostToDevice));
                injected++;
            } else if (cfg.verbose) {
                printf("  [INIT] user solution #%d invalid, skipped\n", i);
            }
        }
        if (cfg.verbose && injected > 0) {
            printf("  [INIT] injected %d/%d user solutions (tail of population)\n",
                   injected, num_init_solutions);
        }
    }
    
    // v3.0: Build sequence registry (replaces old d_op_weights)
    ProblemProfile profile = classify_problem(pcfg);
    SeqRegistry seq_reg = build_seq_registry(profile);

    if (custom_registry_fn) {
        custom_registry_fn(seq_reg);
    }
    
    // v3.1: K-step config (multi-step execution)
    KStepConfig kstep = build_kstep_config();
    
    if (cfg.verbose) {
        const char* scale_names[] = {"Small", "Medium", "Large"};
        const char* struct_names[] = {"SingleSeq", "MultiFixed", "MultiPartition"};
        printf("  [PROFILE] scale=%s structure=%s\n",
               scale_names[(int)profile.scale], struct_names[(int)profile.structure]);
        printf("  [SEQ] %d sequences registered:", seq_reg.count);
        for (int i = 0; i < seq_reg.count; i++)
            printf(" %d(%.2f)", seq_reg.ids[i], seq_reg.weights[i]);
        printf("\n");
        printf("  [K-STEP] K weights: K1=%.2f K2=%.2f K3=%.2f\n",
               kstep.weights[0], kstep.weights[1], kstep.weights[2]);
    }
    
    int* d_best_idx;
    CUDA_CHECK(cudaMalloc(&d_best_idx, sizeof(int)));
    
    Sol* d_global_best = nullptr;
    if (use_sa) {
        CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(Sol)));
        // v5.0 plan B3: expose d_global_best pointer for external read (optional)
        if (d_global_best_out != nullptr) {
            *d_global_best_out = d_global_best;
        }
    }
    
    // AOS: allocate global stats buffer (per-sequence granularity)
    AOSStats* d_aos_stats = nullptr;
    AOSStats* h_aos_stats = nullptr;
    
    if (use_aos) {
        CUDA_CHECK(cudaMalloc(&d_aos_stats, sizeof(AOSStats) * pop_size));
        h_aos_stats = new AOSStats[pop_size];
    }
    
    // --- Relation matrices (G/O) for SEQ_LNS_GUIDED_REBUILD ---
    // Enabled only for Permutation encoding when GUIDED_REBUILD is in registry
    bool use_relation_matrix = false;
    RelationMatrix rel_mat = {};
    int rel_N = 0;
    if (pcfg.encoding == EncodingType::Permutation) {
        for (int i = 0; i < seq_reg.count; i++) {
            if (seq_reg.ids[i] == seq::SEQ_LNS_GUIDED_REBUILD) {
                use_relation_matrix = true;
                break;
            }
        }
    }
    if (use_relation_matrix) {
        // N = dim2_default (number of elements in permutation)
        rel_N = pcfg.dim2_default;
        if (rel_N > 0) {
            rel_mat = relation_matrix_create(rel_N, 0.95f);
            // Optional prior init of G/O via user hook (default: no-op)
            prob.init_relation_matrix(rel_mat.h_G, rel_mat.h_O, rel_N);
            relation_matrix_upload(rel_mat);
        } else {
            use_relation_matrix = false;
        }
    }
    
    // grid = pop_size (one block per solution)
    int grid = pop_size;
    
    // --- 2. Initial evaluation ---
    // Sample-select path already evaluated candidates; final pop may still have randoms — re-evaluate
    {
        size_t eval_smem = prob.shared_mem_bytes();
        if (eval_smem > 48 * 1024) {
            cudaFuncSetAttribute(evaluate_kernel<Problem, Sol>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)eval_smem);
        }
        int eval_grid = calc_grid_size(pop_size, block_threads);
        evaluate_kernel<<<eval_grid, block_threads, eval_smem>>>(
            prob, pop.d_solutions, pop_size, eval_smem);
    }
    
    if (use_sa) {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_global_best, pop.d_solutions + idx, sizeof(Sol), cudaMemcpyDeviceToDevice));
    }
    
    // --- 3. Main loop ---
    // Batch size sets update cadence for AOS / relation matrix / convergence checks
    // Balance: too small -> sync overhead; too slow to react if too large
    int batch;
    if (use_islands)
        batch = cfg.migrate_interval;
    else if (cfg.verbose)
        batch = cfg.print_every;
    else
        batch = cfg.max_gen;
    
    // Features needing periodic updates: force batch <= 200
    if (use_relation_matrix || use_aos || use_time_limit || use_stagnation) {
        if (batch > 200) batch = 200;
    }
    
    int gen_done = 0;
    int migrate_round = 0;
    StopReason stop_reason = StopReason::MaxGen;
    
    // Convergence-check state
    float prev_best_scalar = 1e30f;
    int stagnation_count = 0;
    
    // --- EvolveParams: mutable fields (device memory) ---
    EvolveParams h_params;
    h_params.temp_start = 0.0f;
    h_params.gens_per_batch = batch;
    h_params.seq_reg = seq_reg;
    h_params.kstep = kstep;
    h_params.migrate_round = 0;
    h_params.oc = oc;
    
    EvolveParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(EvolveParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice));
    
    // --- CUDA Graph ---
    const bool use_graph = cfg.use_cuda_graph;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t stream = nullptr;
    
    if (use_graph) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    // Lambda: launch one batch of GPU kernels on stream
    auto launch_batch_kernels = [&](cudaStream_t s) {
        evolve_block_kernel<<<grid, block_threads, total_smem, s>>>(
            prob, pop.d_solutions, pop_size,
            pcfg.encoding, pcfg.dim1,
            oc, pop.d_rng_states,
            cfg.sa_alpha, prob_smem,
            d_aos_stats,
            use_relation_matrix ? rel_mat.d_G : nullptr,
            use_relation_matrix ? rel_mat.d_O : nullptr,
            rel_N,
            pcfg.value_lower_bound, pcfg.value_upper_bound,
            d_params);
        
        if (use_crossover) {
            crossover_block_kernel<<<grid, block_threads, total_smem, s>>>(
                prob, pop.d_solutions, pop_size,
                pcfg.encoding, pcfg.dim1,
                oc, pop.d_rng_states,
                cfg.crossover_rate, prob_smem,
                pcfg.row_mode == RowMode::Partition ? pcfg.total_elements : pcfg.dim2_default);
        }
        
        if (use_islands) {
            migrate_kernel<<<1, 1, 0, s>>>(pop.d_solutions, pop_size,
                                            island_size, oc,
                                            cfg.migrate_strategy, d_params);
        }
        
        if (use_sa) {
            elite_inject_kernel<<<1, 1, 0, s>>>(pop.d_solutions, pop_size,
                                                  d_global_best, oc);
        }
    };
    
    // Capture CUDA Graph (first time)
    if (use_graph) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch_batch_kernels(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        // 5-arg form: compatible with CUDA 10+; 3-arg form requires CUDA 12+
#if CUDART_VERSION >= 12000
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
#else
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
#endif
        if (cfg.verbose) printf("  [CUDA Graph] captured and instantiated\n");
    }
    
    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));
    CUDA_CHECK(cudaEventRecord(t_start));
    
    // Time-aware AOS: window accumulators
    int win_seq_usage[MAX_SEQ] = {};
    int win_seq_improve[MAX_SEQ] = {};
    int win_k_usage[MAX_K] = {};
    int win_k_improve[MAX_K] = {};
    int batch_count = 0;
    const int aos_interval = (cfg.aos_update_interval > 0) ? cfg.aos_update_interval : 1;
    
    // v4.0: constraint-directed + phased search (require AOS enabled)
    const bool use_constraint_directed = cfg.use_constraint_directed && use_aos;
    const bool use_phased_search = cfg.use_phased_search && use_aos;
    if (cfg.verbose) {
        if (cfg.use_constraint_directed && !use_aos)
            printf("  [WARN] constraint_directed requires AOS, disabled\n");
        if (cfg.use_phased_search && !use_aos)
            printf("  [WARN] phased_search requires AOS, disabled\n");
    }
    float base_max_w[MAX_SEQ];
    for (int i = 0; i < seq_reg.count; i++) base_max_w[i] = seq_reg.max_w[i];
    
    if (cfg.verbose && (use_constraint_directed || use_phased_search)) {
        printf("  [P2] constraint_directed=%s phased_search=%s\n",
               use_constraint_directed ? "ON" : "OFF",
               use_phased_search ? "ON" : "OFF");
        if (use_phased_search)
            printf("  [P2] phases: explore=[0,%.0f%%) transition=[%.0f%%,%.0f%%) refine=[%.0f%%,100%%]\n",
                   cfg.phase_explore_end * 100, cfg.phase_explore_end * 100,
                   cfg.phase_refine_start * 100, cfg.phase_refine_start * 100);
    }
    
    while (gen_done < cfg.max_gen) {
        int gens = batch;
        if (gen_done + gens > cfg.max_gen) gens = cfg.max_gen - gen_done;
        
        float temp = use_sa ? cfg.sa_temp_init * powf(cfg.sa_alpha, (float)gen_done) : 0.0f;
        
        // Update mutable device parameters
        h_params.temp_start = temp;
        h_params.gens_per_batch = gens;
        h_params.seq_reg = seq_reg;
        h_params.kstep = kstep;
        h_params.migrate_round = migrate_round;
        CUDA_CHECK(cudaMemcpy(d_params, &h_params, sizeof(EvolveParams), cudaMemcpyHostToDevice));
        
        // Launch GPU kernel sequence
        if (use_graph) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            launch_batch_kernels(nullptr);
        }
        
        // v5.0 plan B3: passive injection check (outside Graph)
        // Must be outside Graph: inject_buf content changes dynamically
        if (inject_buf != nullptr && use_islands) {
            inject_check_kernel<<<1, 1>>>(pop.d_solutions, pop_size,
                                           island_size, inject_buf, oc,
                                           cfg.multi_gpu_inject_mode);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        gen_done += gens;
        if (use_islands) migrate_round++;
        batch_count++;
        
        // AOS: two-level weight update (EMA) + stagnation detection
        if (use_aos && (batch_count % aos_interval == 0)) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_aos_stats, d_aos_stats,
                                  sizeof(AOSStats) * pop_size,
                                  cudaMemcpyDeviceToHost));
            
            // --- Fold current batch stats into window accumulators ---
            for (int b = 0; b < pop_size; b++) {
                for (int i = 0; i < seq_reg.count; i++) {
                    win_seq_usage[i] += h_aos_stats[b].usage[i];
                    win_seq_improve[i] += h_aos_stats[b].improvement[i];
                }
                for (int i = 0; i < MAX_K; i++) {
                    win_k_usage[i] += h_aos_stats[b].k_usage[i];
                    win_k_improve[i] += h_aos_stats[b].k_improvement[i];
                }
            }
            
            constexpr float AOS_ALPHA = 0.6f;
            
            // --- v4.0: constraint-directed — population infeasibility ratio ---
            float penalty_ratio = 0.0f;
            if (use_constraint_directed) {
                Sol* h_pop_snap = new Sol[pop_size];
                CUDA_CHECK(cudaMemcpy(h_pop_snap, pop.d_solutions,
                                      sizeof(Sol) * pop_size, cudaMemcpyDeviceToHost));
                int infeasible = 0;
                for (int b = 0; b < pop_size; b++) {
                    if (h_pop_snap[b].penalty > 0.0f) infeasible++;
                }
                penalty_ratio = (float)infeasible / (float)pop_size;
                delete[] h_pop_snap;
            }
            
            // --- v4.0: phased search — phase floor/cap multipliers ---
            float phase_floor_mult = 1.0f;
            float phase_cap_mult   = 1.0f;
            if (use_phased_search) {
                float progress;
                if (use_time_limit && cfg.time_limit_sec > 0.0f) {
                    float elapsed_ms = 0.0f;
                    CUDA_CHECK(cudaEventRecord(t_stop));
                    CUDA_CHECK(cudaEventSynchronize(t_stop));
                    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));
                    progress = elapsed_ms / (cfg.time_limit_sec * 1000.0f);
                    if (progress > 1.0f) progress = 1.0f;
                } else {
                    progress = (float)gen_done / (float)cfg.max_gen;
                }
                if (progress < cfg.phase_explore_end) {
                    phase_floor_mult = 1.5f;   // explore: raise floor -> more uniform
                    phase_cap_mult   = 0.7f;   // explore: lower cap -> avoid early concentration
                } else if (progress >= cfg.phase_refine_start) {
                    phase_floor_mult = 0.5f;   // refine: lower floor -> weak ops can fade
                    phase_cap_mult   = 1.5f;   // refine: raise cap -> exploit strong ops
                }
            }
            
            // --- Layer 2: operator weights (EMA) ---
            {
                float new_w[MAX_SEQ];
                // Deferred normalization: EMA + bounds (no renormalize to sum 1)
                for (int i = 0; i < seq_reg.count; i++) {
                    float signal = (win_seq_usage[i] > 0)
                        ? (float)win_seq_improve[i] / (float)win_seq_usage[i]
                        : 0.0f;
                    new_w[i] = AOS_ALPHA * seq_reg.weights[i]
                             + (1.0f - AOS_ALPHA) * (signal + AOS_WEIGHT_FLOOR);
                }
                
                float uniform = 1.0f / seq_reg.count;
                float base_floor = cfg.aos_weight_floor / seq_reg.count;
                if (base_floor < uniform * 0.5f) base_floor = uniform * 0.5f;
                float floor_val = base_floor * phase_floor_mult;
                float global_cap = cfg.aos_weight_cap * phase_cap_mult;
                
                // --- v4.0: constraint-directed — boost cross-row/row-level weights + relax cap ---
                if (use_constraint_directed && penalty_ratio > 0.1f) {
                    float boost = 1.0f + (penalty_ratio - 0.1f) / 0.9f
                                  * (cfg.constraint_boost_max - 1.0f);
                    for (int i = 0; i < seq_reg.count; i++) {
                        if (seq_reg.categories[i] == SeqCategory::CrossRow ||
                            seq_reg.categories[i] == SeqCategory::RowLevel) {
                            new_w[i] *= boost;
                            float orig = (base_max_w[i] > 0.0f) ? base_max_w[i] : AOS_WEIGHT_CAP;
                            seq_reg.max_w[i] = orig * boost;
                        }
                    }
                } else if (use_constraint_directed) {
                    for (int i = 0; i < seq_reg.count; i++)
                        seq_reg.max_w[i] = base_max_w[i];
                }
                
                // Apply bounds (no renormalize to sum 1)
                float sum = 0.0f;
                for (int i = 0; i < seq_reg.count; i++) {
                    float cap_val = (seq_reg.max_w[i] > 0.0f) ? seq_reg.max_w[i] : global_cap;
                    seq_reg.weights[i] = fmaxf(floor_val, fminf(cap_val, new_w[i]));
                    sum += seq_reg.weights[i];
                }
                
                // Update cached weight sum
                seq_reg.weights_sum = sum;
            }
            
            // --- Layer 1: K-step weights (EMA + deferred normalize) ---
            {
                float new_w[MAX_K];
                for (int i = 0; i < MAX_K; i++) {
                    float rate = (win_k_usage[i] > 0)
                        ? (float)win_k_improve[i] / (float)win_k_usage[i]
                        : 0.0f;
                    new_w[i] = AOS_ALPHA * kstep.weights[i]
                             + (1.0f - AOS_ALPHA) * (rate + AOS_WEIGHT_FLOOR);
                }
                
                // Apply bounds (no renormalize to sum 1)
                float floor_val = cfg.aos_weight_floor;
                float cap_val = 0.95f;
                for (int i = 0; i < MAX_K; i++) {
                    kstep.weights[i] = fmaxf(floor_val, fminf(cap_val, new_w[i]));
                }
                
                // Renormalize K-step weights (legacy behavior; K choice is not roulette)
                float sum = 0.0f;
                for (int i = 0; i < MAX_K; i++) sum += kstep.weights[i];
                if (sum > 0.0f) {
                    for (int i = 0; i < MAX_K; i++)
                        kstep.weights[i] /= sum;
                }
            }
            
            // --- Debug: print stats for first 5 batches ---
            if (cfg.verbose && gen_done <= batch * 5) {
                fprintf(stderr, "  [AOS batch g=%d] usage:", gen_done);
                for (int i = 0; i < seq_reg.count; i++) fprintf(stderr, " %d", win_seq_usage[i]);
                fprintf(stderr, " | improve:");
                for (int i = 0; i < seq_reg.count; i++) fprintf(stderr, " %d", win_seq_improve[i]);
                fprintf(stderr, " | w:");
                for (int i = 0; i < seq_reg.count; i++) fprintf(stderr, " %.3f", seq_reg.weights[i]);
                fprintf(stderr, " | sum=%.3f", seq_reg.weights_sum);
                fprintf(stderr, " | K: %.2f/%.2f/%.2f stag=%d",
                        kstep.weights[0], kstep.weights[1], kstep.weights[2], kstep.stagnation_count);
                if (use_constraint_directed)
                    fprintf(stderr, " | pen=%.1f%%", penalty_ratio * 100.0f);
                if (use_phased_search)
                    fprintf(stderr, " | phase_f=%.2f phase_c=%.2f", phase_floor_mult, phase_cap_mult);
                fprintf(stderr, "\n");
            }
            
            
            // --- Stagnation detection ---
            {
                int total_improve_all = 0;
                for (int i = 0; i < seq_reg.count; i++)
                    total_improve_all += win_seq_improve[i];
                
                if (total_improve_all == 0) {
                    kstep.stagnation_count++;
                } else {
                    kstep.stagnation_count = 0;
                }
                
                if (kstep.stagnation_count >= kstep.stagnation_limit) {
                    kstep.weights[0] = 0.80f;
                    kstep.weights[1] = 0.15f;
                    kstep.weights[2] = 0.05f;
                    kstep.stagnation_count = 0;
                }
            }
            
            // --- Clear window accumulators ---
            memset(win_seq_usage, 0, sizeof(win_seq_usage));
            memset(win_seq_improve, 0, sizeof(win_seq_improve));
            memset(win_k_usage, 0, sizeof(win_k_usage));
            memset(win_k_improve, 0, sizeof(win_k_improve));
        }
        
        // --- Relation matrix update (between batches, from population top-K) ---
        // Several good solutions contribute G/O signal to build the matrix faster
        if (use_relation_matrix) {
            if (!use_aos) {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            
            // Download population objectives and find top-K
            constexpr int REL_TOP_K = 4;
            int top_indices[REL_TOP_K];
            {
                // Simple approach: scalar objectives on host, pick top-K minima
                float* h_scores = new float[pop_size];
                Sol* h_pop_ptr = new Sol[pop_size];
                CUDA_CHECK(cudaMemcpy(h_pop_ptr, pop.d_solutions,
                                      sizeof(Sol) * pop_size, cudaMemcpyDeviceToHost));
                for (int b = 0; b < pop_size; b++) {
                    h_scores[b] = scalar_objective(h_pop_ptr[b], oc);
                    if (h_pop_ptr[b].penalty > 0.0f) h_scores[b] = 1e30f;
                }
                // Find top-K smallest scores
                for (int k = 0; k < REL_TOP_K && k < pop_size; k++) {
                    int mi = 0;
                    for (int b = 1; b < pop_size; b++) {
                        if (h_scores[b] < h_scores[mi]) mi = b;
                    }
                    top_indices[k] = mi;
                    h_scores[mi] = 1e30f;  // mark as taken
                }
                // Update G/O from top-K solutions
                int actual_k = (pop_size < REL_TOP_K) ? pop_size : REL_TOP_K;
                for (int k = 0; k < actual_k; k++) {
                    relation_matrix_update(rel_mat, h_pop_ptr[top_indices[k]], pcfg.dim1);
                }
                delete[] h_scores;
                delete[] h_pop_ptr;
            }
            
            relation_matrix_upload(rel_mat);
        }
        
        // Crossover / migrate / elite inject already launched in launch_batch_kernels
        
        // --- Time limit check ---
        if (use_time_limit) {
            CUDA_CHECK(cudaEventRecord(t_stop));
            CUDA_CHECK(cudaEventSynchronize(t_stop));
            float ms_so_far = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms_so_far, t_start, t_stop));
            if (ms_so_far >= cfg.time_limit_sec * 1000.0f) {
                stop_reason = StopReason::TimeLimit;
                if (cfg.verbose) printf("  [STOP] time limit %.1fs reached at gen %d\n",
                                         cfg.time_limit_sec, gen_done);
                break;
            }
        }
        
        // --- Convergence check + reheat ---
        if (use_stagnation) {
            find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
            CUDA_CHECK(cudaDeviceSynchronize());
            int bi; CUDA_CHECK(cudaMemcpy(&bi, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
            Sol cur_best = pop.download_solution(bi);
            float cur_scalar = scalar_objective(cur_best, oc);
            if (cur_best.penalty > 0.0f) cur_scalar = 1e30f;
            
            constexpr float IMPROVE_EPS = 1e-6f;
            if (prev_best_scalar - cur_scalar > IMPROVE_EPS) {
                prev_best_scalar = cur_scalar;
                stagnation_count = 0;
            } else {
                stagnation_count++;
            }
            
            if (stagnation_count >= cfg.stagnation_limit) {
                if (use_sa && cfg.reheat_ratio > 0.0f) {
                    // Reheat: restore temperature to reheat_ratio * initial
                    // Implemented by rolling back gen_done (temp = init * alpha^gen_done)
                    float target_temp = cfg.sa_temp_init * cfg.reheat_ratio;
                    int reheat_gen = (int)(logf(target_temp / cfg.sa_temp_init) / logf(cfg.sa_alpha));
                    if (reheat_gen < 0) reheat_gen = 0;
                    // Not a true gen_done rollback for termination; conceptually temp_offset
                    // Simplified: next batch temp follows from adjusted gen_done
                    if (cfg.verbose) {
                        float cur_temp = cfg.sa_temp_init * powf(cfg.sa_alpha, (float)gen_done);
                        printf("  [REHEAT] stagnation=%d at gen %d, temp %.4f → %.4f\n",
                               cfg.stagnation_limit, gen_done, cur_temp, target_temp);
                    }
                    // Roll gen_done back to match target_temp (but not below half of completed gens)
                    int min_gen = gen_done / 2;
                    if (reheat_gen < min_gen) reheat_gen = min_gen;
                    gen_done = reheat_gen;
                    stagnation_count = 0;
                } else {
                    // No SA: stagnation triggers early stop
                    stop_reason = StopReason::Stagnation;
                    if (cfg.verbose) printf("  [STOP] stagnation=%d at gen %d, no SA to reheat\n",
                                             cfg.stagnation_limit, gen_done);
                    break;
                }
            }
        }
        
        // Progress printout
        if (cfg.verbose && gen_done % cfg.print_every == 0) {
            if (!use_stagnation) {
                find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            int idx; CUDA_CHECK(cudaMemcpy(&idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
            Sol best = pop.download_solution(idx);
            printf("  [%5d]", gen_done);
            for (int i = 0; i < pcfg.num_objectives; i++)
                printf(" %.1f", best.objectives[i]);
            if (best.penalty > 0.0f) printf(" P=%.1f", best.penalty);
            printf("\n");
        }
    }
    
    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));
    
    // --- 4. Final result ---
    Sol best;
    if (use_sa) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&best, d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost));
    } else {
        find_best_kernel<<<1, 1>>>(pop.d_solutions, pop_size, oc, d_best_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
        int h_best_idx;
        CUDA_CHECK(cudaMemcpy(&h_best_idx, d_best_idx, sizeof(int), cudaMemcpyDeviceToHost));
        best = pop.download_solution(h_best_idx);
    }
    
    if (cfg.verbose) {
        const char* reason_str = stop_reason == StopReason::TimeLimit ? " [time]" :
                                 stop_reason == StopReason::Stagnation ? " [stag]" : "";
        printf("  Result:");
        for (int i = 0; i < pcfg.num_objectives; i++)
            printf(" obj%d=%.2f", i, best.objectives[i]);
        if (best.penalty > 0.0f) printf(" INFEASIBLE(%.2f)", best.penalty);
        printf("  %.0fms %dgen%s\n", elapsed_ms, gen_done, reason_str);
    }
    
    if (cfg.verbose) {
        for (int r = 0; r < pcfg.dim1; r++) {
            printf("  row[%d]:", r);
            int show = best.dim2_sizes[r] < 20 ? best.dim2_sizes[r] : 20;
            for (int c = 0; c < show; c++) printf(" %d", best.data[r][c]);
            if (best.dim2_sizes[r] > 20) printf(" ...(%d)", best.dim2_sizes[r]);
            printf("\n");
        }
    }
    
    // AOS: print final two-level weights
    if (use_aos && cfg.verbose) {
        printf("  AOS K-step weights: K1=%.3f K2=%.3f K3=%.3f\n",
               kstep.weights[0], kstep.weights[1], kstep.weights[2]);
        printf("  AOS seq weights:");
        for (int i = 0; i < seq_reg.count; i++)
            printf(" [%d]=%.3f", seq_reg.ids[i], seq_reg.weights[i]);
        printf("\n");
    }
    
    // Fill return struct
    result.best_solution = best;
    result.elapsed_ms = elapsed_ms;
    result.generations = gen_done;
    result.stop_reason = stop_reason;
    
    CUDA_CHECK(cudaFree(d_best_idx));
    if (d_global_best) CUDA_CHECK(cudaFree(d_global_best));
    if (d_aos_stats) CUDA_CHECK(cudaFree(d_aos_stats));
    if (h_aos_stats) delete[] h_aos_stats;
    if (use_relation_matrix) relation_matrix_destroy(rel_mat);
    CUDA_CHECK(cudaFree(d_params));
    if (graph_exec) CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));
    
    return result;
}
