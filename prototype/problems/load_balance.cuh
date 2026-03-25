/**
 * load_balance.cuh - discrete load balancing (Integer encoding sanity check)
 *
 * N tasks on M machines, processing time p[i] per task.
 * Decision: data[0][i] in [0, M-1] = machine for task i.
 * Objective: minimize makespan (max machine load).
 *
 * NP-hard (same as multiprocessor scheduling / load balancing).
 * LPT (longest processing time first) greedy achieves 4/3 approximation.
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct LoadBalanceProblem : ProblemBase<LoadBalanceProblem, 1, 64> {
    const float* d_proc_time;   // task processing times [N]
    int n;                      // number of tasks
    int m;                      // number of machines
    
    __device__ float calc_makespan(const Sol& sol) const {
        float load[32] = {};    // at most 32 machines
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++) {
            int machine = sol.data[0][i];
            if (machine >= 0 && machine < m)
                load[machine] += d_proc_time[i];
        }
        float max_load = 0.0f;
        for (int j = 0; j < m; j++)
            if (load[j] > max_load) max_load = load[j];
        return max_load;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},   // case 0: makespan
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_makespan(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;   // no side constraints (any assignment is feasible)
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = m - 1;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        return (size_t)n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sp = reinterpret_cast<float*>(smem);
        for (int i = tid; i < n; i += bsz) sp[i] = d_proc_time[i];
        d_proc_time = sp;
    }
    
    static LoadBalanceProblem create(const float* h_proc_time, int n, int m) {
        LoadBalanceProblem prob;
        prob.n = n; prob.m = m;
        float* dp;
        CUDA_CHECK(cudaMalloc(&dp, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(dp, h_proc_time, sizeof(float) * n, cudaMemcpyHostToDevice));
        prob.d_proc_time = dp;
        return prob;
    }
    
    void destroy() {
        if (d_proc_time) cudaFree(const_cast<float*>(d_proc_time));
        d_proc_time = nullptr;
    }
};
