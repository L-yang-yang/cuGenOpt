/**
 * tsp_xlarge.cuh - very large TSP definition (up to 512 cities)
 *
 * Same as tsp_large.cuh under ProblemBase, with D2=512.
 * Note: distance matrix 512×512×4B = 1MB, far above 48KB shared memory,
 *       so shared_mem_bytes() returns 0 and the matrix stays in global memory.
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

struct TSPXLargeProblem : ProblemBase<TSPXLargeProblem, 1, 512> {
    const float* d_dist;
    const float* h_dist;  // host distance matrix (for init_relation_matrix)
    int n;
    
    __device__ float calc_total_distance(const Sol& sol) const {
        float total = 0.0f;
        const int* route = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            total += d_dist[route[i] * n + route[(i + 1) % size]];
        return total;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_distance(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const { return 0.0f; }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        fill_obj_config(cfg);
        return cfg;
    }
    
    // Distance matrix too large for shared memory
    size_t shared_mem_bytes() const { return 0; }
    __device__ void load_shared(char*, int, int) {}
    
    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }
    
    // Initialize G/O priors from distances: closer → higher score
    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_dist || N != n) return;
        // Max distance for normalization
        float max_d = 0.0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (h_dist[i * N + j] > max_d) max_d = h_dist[i * N + j];
        if (max_d <= 0.0f) return;
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                // Closer → higher G (stronger grouping signal)
                float proximity = 1.0f - h_dist[i * N + j] / max_d;
                G[i * N + j] = proximity * 0.3f;  // keep initial signal moderate for EMA headroom
                // Closer → small O signal too (symmetric, no directional bias)
                O[i * N + j] = proximity * 0.1f;
            }
        }
    }
    
    int heuristic_matrices(HeuristicMatrix* out, int max_count) const {
        if (max_count < 1 || !h_dist) return 0;
        out[0] = {h_dist, n};
        return 1;
    }
    
    static TSPXLargeProblem create(const float* h_dist_ptr, int n) {
        TSPXLargeProblem prob;
        prob.n = n;
        prob.h_dist = h_dist_ptr;  // keep host pointer
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
