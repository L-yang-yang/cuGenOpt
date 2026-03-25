/**
 * bin_packing.cuh - one-dimensional bin packing (Integer encoding + constraints)
 *
 * N items with weights w[i], at most B bins, capacity C per bin.
 * Decision: data[0][i] in [0, B-1] = bin index for item i.
 * Objective: minimize number of bins used.
 * Constraint: bin load ≤ C; overflow contributes to penalty.
 *
 * Validation instance: 8 items weights=[7,5,3,4,6,2,8,1], C=10, optimum=4 bins
 *   bin0={7,3}=10, bin1={5,4,1}=10, bin2={6,2}=8, bin3={8}=8
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct BinPackingProblem : ProblemBase<BinPackingProblem, 1, 64> {
    const float* d_weights;
    int n;              // number of items
    int max_bins;       // max bins B
    float capacity;     // bin capacity C
    
    __device__ float calc_bins_used(const Sol& sol) const {
        bool used[32] = {};
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++) {
            int b = sol.data[0][i];
            if (b >= 0 && b < max_bins) used[b] = true;
        }
        int count = 0;
        for (int b = 0; b < max_bins; b++)
            if (used[b]) count++;
        return (float)count;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_bins_used(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        float penalty = 0.0f;
        float load[32] = {};
        int size = sol.dim2_sizes[0];
        for (int i = 0; i < size; i++) {
            int b = sol.data[0][i];
            if (b >= 0 && b < max_bins)
                load[b] += d_weights[i];
        }
        for (int b = 0; b < max_bins; b++) {
            float over = load[b] - capacity;
            if (over > 0.0f) penalty += over * 10.0f;
        }
        return penalty;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = 1;  cfg.dim2_default = n;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = max_bins - 1;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        return (size_t)n * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sw = reinterpret_cast<float*>(smem);
        for (int i = tid; i < n; i += bsz) sw[i] = d_weights[i];
        d_weights = sw;
    }
    
    static BinPackingProblem create(const float* h_weights, int n,
                                     int max_bins, float capacity) {
        BinPackingProblem prob;
        prob.n = n; prob.max_bins = max_bins; prob.capacity = capacity;
        float* dw;
        CUDA_CHECK(cudaMalloc(&dw, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(dw, h_weights, sizeof(float) * n, cudaMemcpyHostToDevice));
        prob.d_weights = dw;
        return prob;
    }
    
    void destroy() {
        if (d_weights) cudaFree(const_cast<float*>(d_weights));
        d_weights = nullptr;
    }
};
