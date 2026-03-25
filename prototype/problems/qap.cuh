/**
 * qap.cuh - Quadratic Assignment Problem (QAP)
 *
 * Assign N facilities to N locations (permutation encoding).
 * Decision: data[0][i] = location assigned to facility i.
 * Objective: Minimize sum(flow[i][j] * dist[perm[i]][perm[j]])
 *
 * Validation instance: custom 5x5
 *   flow: inter-facility flow
 *   dist: inter-location distances
 *   known optimum = 58
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct QAPProblem : ProblemBase<QAPProblem, 1, 32> {
    const float* d_flow;    // flow matrix [N*N] (device)
    const float* d_dist;    // distance matrix [N*N] (device)
    const float* h_flow;    // flow matrix [N*N] (host, for clone_to_device)
    const float* h_dist;    // distance matrix [N*N] (host, for clone_to_device)
    int n;
    
    __device__ float calc_cost(const Sol& sol) const {
        float cost = 0.0f;
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                cost += d_flow[i * n + j] * d_dist[sol.data[0][i] * n + sol.data[0][j]];
        return cost;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_cost(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        return 2 * (size_t)n * n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sf = reinterpret_cast<float*>(smem);
        float* sd = sf + n * n;
        int total = n * n;
        for (int i = tid; i < total; i += bsz) { sf[i] = d_flow[i]; sd[i] = d_dist[i]; }
        d_flow = sf;
        d_dist = sd;
    }
    
    static QAPProblem create(const float* h_flow_in, const float* h_dist_in, int n) {
        QAPProblem prob;
        prob.n = n;
        prob.h_flow = h_flow_in;
        prob.h_dist = h_dist_in;
        float *df, *dd;
        CUDA_CHECK(cudaMalloc(&df, sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(df, h_flow_in, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dd, h_dist_in, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        prob.d_flow = df; prob.d_dist = dd;
        return prob;
    }
    
    void destroy() {
        if (d_flow) cudaFree(const_cast<float*>(d_flow));
        if (d_dist) cudaFree(const_cast<float*>(d_dist));
        d_flow = nullptr; d_dist = nullptr;
    }
    
    // v5.0: multi-GPU — clone onto a given device
    QAPProblem* clone_to_device(int gpu_id) const override {
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        
        // Use host-side matrices directly (no D2H needed)
        CUDA_CHECK(cudaSetDevice(gpu_id));
        float *df, *dd;
        CUDA_CHECK(cudaMalloc(&df, sizeof(float) * n * n));
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(df, h_flow, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaSetDevice(orig_device));
        
        QAPProblem* new_prob = new QAPProblem();
        new_prob->n = n;
        new_prob->h_flow = h_flow;
        new_prob->h_dist = h_dist;
        new_prob->d_flow = df;
        new_prob->d_dist = dd;
        
        return new_prob;
    }
};
