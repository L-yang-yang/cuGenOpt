/**
 * tsp_large.cuh - large-scale TSP definition (up to 256 cities)
 *
 * Same logic as tsp.cuh under ProblemBase; only D2 cap differs.
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

struct TSPLargeProblem : ProblemBase<TSPLargeProblem, 1, 256> {
    const float* d_dist;
    const float* h_dist;
    int n;
    
    // ---- objective evaluation ----
    __device__ float calc_total_distance(const Sol& sol) const {
        float total = 0.0f;
        const int* route = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            total += d_dist[route[i] * n + route[(i + 1) % size]];
        return total;
    }
    
    // ---- objective defs (OBJ_DEFS must match compute_obj one-to-one) ----
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},   // case 0: calc_total_distance
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_distance(sol);   // OBJ_DEFS[0]
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
    
    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    
    size_t shared_mem_bytes() const {
        size_t need = (size_t)n * n * sizeof(float);
        return need <= SMEM_LIMIT ? need : 0;
    }
    
    // Actual distance matrix size (whether or not placed in smem)
    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sd = reinterpret_cast<float*>(smem);
        int total = n * n;
        for (int i = tid; i < total; i += bsz)
            sd[i] = d_dist[i];
        d_dist = sd;
    }
    
    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_dist || N != n) return;
        float max_d = 0.0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (h_dist[i * N + j] > max_d) max_d = h_dist[i * N + j];
        if (max_d <= 0.0f) return;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                float proximity = 1.0f - h_dist[i * N + j] / max_d;
                G[i * N + j] = proximity * 0.3f;
                O[i * N + j] = proximity * 0.1f;
            }
    }
    
    int heuristic_matrices(HeuristicMatrix* out, int max_count) const {
        if (max_count < 1 || !h_dist) return 0;
        out[0] = {h_dist, n};
        return 1;
    }
    
    static TSPLargeProblem create(const float* h_dist_ptr, int n) {
        TSPLargeProblem prob;
        prob.n = n;
        prob.h_dist = h_dist_ptr;
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * n * n));
        CUDA_CHECK(cudaMemcpy(dd, h_dist_ptr, sizeof(float) * n * n, cudaMemcpyHostToDevice));
        prob.d_dist = dd;
        return prob;
    }
    
    void destroy() {
        if (d_dist) { cudaFree(const_cast<float*>(d_dist)); d_dist = nullptr; }
        h_dist = nullptr;
    }
};
