/**
 * multi_gpu_solver.cuh - Multi-GPU cooperative solving
 * 
 * v5.0 plan B3: passive injection + GPU-agnostic design
 *   - Each GPU runs solve() independently with its own seed
 *   - Each GPU has an InjectBuffer (device memory)
 *   - A CPU coordinator thread periodically (every N seconds) collects each GPU's best and asynchronously writes to other GPUs' InjectBuffers
 *   - After migrate_kernel, each GPU checks InjectBuffer and injects if a new solution is present
 *   - Fully decoupled: GPUs need not pause; CPU writes asynchronously; CUDA stream sync ensures safety
 */

#pragma once
#include "solver.cuh"
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>

// ============================================================
// MultiGpuContext — per-GPU context
// ============================================================

template<typename Problem>
struct MultiGpuContext {
    using Sol = typename Problem::Sol;
    
    int gpu_id;                      // GPU device ID
    Problem* problem;                // Problem instance (device pointer for this GPU)
    SolverConfig config;             // Solver config (independent seed)
    
    Sol best_solution;               // Current best solution (host)
    SolveResult<Sol> solve_result;   // Full result from solve()
    std::mutex best_mutex;           // Mutex protecting best_solution
    
    InjectBuffer<Sol>* d_inject_buf; // Device-side inject buffer (allocated on this GPU)
    Sol* d_global_best;              // Device pointer to global best (exported by solve())
    
    std::atomic<bool> stop_flag;     // Stop flag
    std::atomic<bool> running;       // Running flag (for coordinator thread)
    
    MultiGpuContext(int id) : gpu_id(id), problem(nullptr), d_inject_buf(nullptr), 
                               d_global_best(nullptr), stop_flag(false), running(false) {
        best_solution = Sol{};
        best_solution.penalty = 1e30f;
        for (int i = 0; i < MAX_OBJ; i++) best_solution.objectives[i] = 1e30f;
    }
};

// ============================================================
// GPU worker thread (plan B3)
// ============================================================

template<typename Problem>
void gpu_worker(MultiGpuContext<Problem>* ctx) {
    using Sol = typename Problem::Sol;
    
    // Set GPU for this thread
    CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
    
    // Mark as running
    ctx->running.store(true);
    
    // Run solve (pass inject_buf and d_global_best_out)
    SolveResult<Sol> result = solve(*ctx->problem, ctx->config, 
                                     nullptr, 0, nullptr, ctx->d_inject_buf, &ctx->d_global_best);
    
    // Mark as finished running
    ctx->running.store(false);
    
    // Update best solution and full result
    {
        std::lock_guard<std::mutex> lock(ctx->best_mutex);
        ctx->best_solution = result.best_solution;
        ctx->solve_result = result;
    }
    
    // Mark complete
    ctx->stop_flag.store(true);
}

// ============================================================
// Coordinator thread (plan B3)
// ============================================================
// Periodically read each GPU's current best from d_global_best, compute global_best, inject to other GPUs
//
// Key design:
// 1. Read directly from each GPU's d_global_best (exported by solve())
// 2. Requires SA enabled (otherwise no d_global_best)
// 3. Light touch: solve() only exports a pointer; single-GPU path unchanged

template<typename Problem>
void coordinator_thread(std::vector<MultiGpuContext<Problem>*>& contexts,
                        float interval_sec, bool verbose) {
    using Sol = typename Problem::Sol;
    ObjConfig oc = contexts[0]->problem->obj_config();
    
    auto interval_ms = std::chrono::milliseconds(static_cast<int>(interval_sec * 1000));
    int round = 0;
    
    // Wait until all GPUs' d_global_best are ready
    bool all_ready = false;
    while (!all_ready) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        all_ready = true;
        for (auto* ctx : contexts) {
            if (ctx->d_global_best == nullptr && ctx->running.load()) {
                all_ready = false;
                break;
            }
        }
    }
    
    while (true) {
        // Wait for the configured interval
        std::this_thread::sleep_for(interval_ms);
        
        // Check whether all GPUs have stopped
        bool all_stopped = true;
        for (auto* ctx : contexts) {
            if (ctx->running.load()) {
                all_stopped = false;
                break;
            }
        }
        if (all_stopped) break;
        
        round++;
        
        // Collect each GPU's current best (from d_global_best)
        Sol global_best;
        global_best.penalty = 1e30f;
        global_best.objectives[0] = 1e30f;
        int best_gpu = -1;
        
        for (int i = 0; i < (int)contexts.size(); i++) {
            if (!contexts[i]->running.load()) continue;  // skip stopped GPUs
            if (contexts[i]->d_global_best == nullptr) continue;  // skip not ready
            
            // Read from this GPU's d_global_best
            Sol gpu_best;
            cudaSetDevice(contexts[i]->gpu_id);
            cudaMemcpy(&gpu_best, contexts[i]->d_global_best, sizeof(Sol), cudaMemcpyDeviceToHost);
            
            if (best_gpu == -1 || is_better(gpu_best, global_best, oc)) {
                global_best = gpu_best;
                best_gpu = i;
            }
        }
        
        if (best_gpu == -1) continue;  // all GPUs stopped or not ready
        
        if (verbose) {
            printf("  [Coordinator Round %d] Global best from GPU %d: obj=%.2f, penalty=%.2f\n",
                   round, best_gpu, global_best.objectives[0], global_best.penalty);
        }
        
        // Inject global_best into other GPUs (except best_gpu)
        for (int i = 0; i < (int)contexts.size(); i++) {
            if (i == best_gpu) continue;  // do not inject to self
            if (!contexts[i]->running.load()) continue;  // do not inject to stopped GPUs
            
            // Read InjectBuffer struct (device to host)
            InjectBuffer<Sol> buf;
            cudaMemcpy(&buf, contexts[i]->d_inject_buf, sizeof(InjectBuffer<Sol>), cudaMemcpyDeviceToHost);
            
            // Synchronous write (switches device as needed)
            buf.write_sync(global_best, contexts[i]->gpu_id);
        }
    }
    
    if (verbose) {
        printf("  [Coordinator] All GPUs stopped, coordinator exiting.\n");
    }
}

// ============================================================
// Multi-GPU cooperative solve entry (plan B3)
// ============================================================

template<typename Problem>
SolveResult<typename Problem::Sol> solve_multi_gpu(Problem& prob, const SolverConfig& cfg) {
    using Sol = typename Problem::Sol;
    
    if (cfg.num_gpus <= 1) {
        // Single-GPU mode: call plain solve
        return solve(prob, cfg);
    }
    
    // Check available GPU count
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "Error: No CUDA devices available\n");
        return SolveResult<Sol>{};
    }
    int actual_gpus = std::min(cfg.num_gpus, device_count);
    
    if (cfg.verbose) {
        printf("  [Multi-GPU B3] Using %d GPUs (requested %d, available %d)\n",
               actual_gpus, cfg.num_gpus, device_count);
        printf("  [Multi-GPU B3] Exchange interval: %.1fs, inject mode: %s\n",
               cfg.multi_gpu_interval_sec,
               cfg.multi_gpu_inject_mode == MultiGpuInjectMode::OneIsland ? "OneIsland" :
               cfg.multi_gpu_inject_mode == MultiGpuInjectMode::HalfIslands ? "HalfIslands" : "AllIslands");
    }
    
    // Create per-GPU contexts
    std::vector<MultiGpuContext<Problem>*> contexts;
    for (int i = 0; i < actual_gpus; i++) {
        auto* ctx = new MultiGpuContext<Problem>(i);
        ctx->config = cfg;
        ctx->config.seed = cfg.seed + i * 1000;  // distinct seed per GPU
        ctx->config.num_gpus = 1;  // run as single-GPU per device
        
        // Clone Problem onto this GPU
        ctx->problem = prob.clone_to_device(i);
        if (ctx->problem == nullptr) {
            fprintf(stderr, "Error: Failed to clone problem to GPU %d\n", i);
            for (auto* c : contexts) {
                if (c->problem) delete c->problem;
                delete c;
            }
            return SolveResult<Sol>{};
        }
        
        // Allocate InjectBuffer on this GPU
        InjectBuffer<Sol> buf = InjectBuffer<Sol>::allocate(i);
        
        // Copy InjectBuffer to device (for kernels)
        InjectBuffer<Sol>* d_buf;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_buf, sizeof(InjectBuffer<Sol>)));
        CUDA_CHECK(cudaMemcpy(d_buf, &buf, sizeof(InjectBuffer<Sol>), cudaMemcpyHostToDevice));
        ctx->d_inject_buf = d_buf;
        
        contexts.push_back(ctx);
    }
    
    // Start worker threads
    std::vector<std::thread> workers;
    for (auto* ctx : contexts) {
        workers.emplace_back(gpu_worker<Problem>, ctx);
    }
    
    // Start coordinator thread (periodic global_best injection)
    std::thread coordinator(coordinator_thread<Problem>, std::ref(contexts),
                            cfg.multi_gpu_interval_sec, cfg.verbose);
    
    // Wait for all workers to finish
    for (auto& w : workers) w.join();
    
    // Wait for coordinator to finish
    coordinator.join();
    
    // Collect final result from best GPU
    Sol final_best = contexts[0]->best_solution;
    int best_ctx = 0;
    ObjConfig oc = prob.obj_config();
    for (int i = 1; i < (int)contexts.size(); i++) {
        if (is_better(contexts[i]->best_solution, final_best, oc)) {
            final_best = contexts[i]->best_solution;
            best_ctx = i;
        }
    }
    
    // Cleanup
    for (auto* ctx : contexts) {
        // Read InjectBuffer content (for teardown)
        InjectBuffer<Sol> buf;
        CUDA_CHECK(cudaSetDevice(ctx->gpu_id));
        CUDA_CHECK(cudaMemcpy(&buf, ctx->d_inject_buf, sizeof(InjectBuffer<Sol>), cudaMemcpyDeviceToHost));
        buf.destroy();
        CUDA_CHECK(cudaFree(ctx->d_inject_buf));
        
        if (ctx->problem) delete ctx->problem;
        delete ctx;
    }
    
    // Build return value from best GPU's result
    SolveResult<Sol> result = contexts[best_ctx]->solve_result;
    result.best_solution = final_best;
    
    return result;
}
