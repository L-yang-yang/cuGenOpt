/**
 * assignment.cuh - assignment problem
 *
 * Extends ProblemBase with ObjDef objective registration.
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include "operators.cuh"

struct AssignmentProblem : ProblemBase<AssignmentProblem, 1, 16> {
    const float* d_cost;
    const float* h_cost;  // host cost matrix (for init_relation_matrix)
    int n;
    
    // ---- objective evaluation ----
    __device__ float calc_total_cost(const Sol& sol) const {
        float total = 0.0f;
        const int* assign = sol.data[0];
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            total += d_cost[i * n + assign[i]];
        return total;
    }
    
    // ---- objective defs (OBJ_DEFS must match compute_obj one-to-one) ----
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},   // case 0: calc_total_cost
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_total_cost(sol);   // OBJ_DEFS[0]
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
    
    // ---- shared memory interface ----
    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    
    size_t shared_mem_bytes() const {
        size_t need = (size_t)n * n * sizeof(float);
        return need <= SMEM_LIMIT ? need : 0;
    }
    
    size_t working_set_bytes() const {
        return (size_t)n * n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sc = reinterpret_cast<float*>(smem);
        int total = n * n;
        for (int i = tid; i < total; i += bsz) sc[i] = d_cost[i];
        d_cost = sc;
    }
    
    // Cost prior: if tasks j and k are similarly preferred by agents, G is high
    // O matrix: low cost for task j at slot i → slightly higher O[j][k] (j tends before k)
    void init_relation_matrix(float* G, float* O, int N) const {
        if (!h_cost || N != n) return;
        // Per task, build cost vectors; cosine similarity between tasks → G
        // Simplified: correlation of cost columns
        float max_c = 0.0f;
        for (int i = 0; i < N * N; i++)
            if (h_cost[i] > max_c) max_c = h_cost[i];
        if (max_c <= 0.0f) return;
        
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) {
                if (j == k) continue;
                // G: more similar cost columns → more likely to swap tasks
                float dot = 0.0f, nj = 0.0f, nk = 0.0f;
                for (int i = 0; i < N; i++) {
                    float cj = h_cost[i * N + j] / max_c;
                    float ck = h_cost[i * N + k] / max_c;
                    dot += cj * ck;
                    nj += cj * cj;
                    nk += ck * ck;
                }
                float denom = sqrtf(nj) * sqrtf(nk);
                float sim = (denom > 1e-6f) ? dot / denom : 0.0f;
                G[j * N + k] = sim * 0.2f;
                O[j * N + k] = sim * 0.05f;
            }
    }
    
    static AssignmentProblem create(const float* hc, int n) {
        AssignmentProblem prob;
        prob.n = n;
        prob.h_cost = hc;
        float* dc;
        CUDA_CHECK(cudaMalloc(&dc, sizeof(float)*n*n));
        CUDA_CHECK(cudaMemcpy(dc, hc, sizeof(float)*n*n, cudaMemcpyHostToDevice));
        prob.d_cost = dc;
        return prob;
    }
    
    void destroy() {
        if (d_cost) { cudaFree(const_cast<float*>(d_cost)); d_cost = nullptr; }
        h_cost = nullptr;
    }
};
