/**
 * graph_color.cuh - graph coloring (Integer encoding)
 *
 * Graph on N nodes, k colors.
 * Decision: data[0][i] in [0, k-1] = color of node i.
 * Objective: minimize number of conflicting edges (adjacent same color).
 *
 * Validation instance: Petersen graph (10 nodes, 15 edges, chromatic number 3, optimal conflicts=0)
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct GraphColorProblem : ProblemBase<GraphColorProblem, 1, 64> {
    const int* d_adj;   // adjacency [N*N] (1=edge, 0=no edge)
    int n;              // number of nodes
    int k;              // number of colors
    
    __device__ float calc_conflicts(const Sol& sol) const {
        int conflicts = 0;
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++)
            for (int j = i + 1; j < size; j++)
                if (d_adj[i * n + j] && sol.data[0][i] == sol.data[0][j])
                    conflicts++;
        return (float)conflicts;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_conflicts(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = k - 1;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        return (size_t)n * n * sizeof(int);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        int* sa = reinterpret_cast<int*>(smem);
        int total = n * n;
        for (int i = tid; i < total; i += bsz) sa[i] = d_adj[i];
        d_adj = sa;
    }
    
    static GraphColorProblem create(const int* h_adj, int n, int k) {
        GraphColorProblem prob;
        prob.n = n; prob.k = k;
        int* da;
        CUDA_CHECK(cudaMalloc(&da, sizeof(int) * n * n));
        CUDA_CHECK(cudaMemcpy(da, h_adj, sizeof(int) * n * n, cudaMemcpyHostToDevice));
        prob.d_adj = da;
        return prob;
    }
    
    void destroy() {
        if (d_adj) cudaFree(const_cast<int*>(d_adj));
        d_adj = nullptr;
    }
};
