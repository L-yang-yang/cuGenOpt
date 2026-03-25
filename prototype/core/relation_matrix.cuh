/**
 * relation_matrix.cuh - G/O relation matrix management
 *
 * G[i][j]: grouping affinity (tendency for elements i and j to be on the same row; symmetric)
 * O[i][j]: ordering affinity (tendency for element i to appear before j; asymmetric)
 *
 * Update source: statistics from historical best solutions
 *   Whenever the host obtains the current best solution, scan all element-pair relations:
 *     - Same row → strengthen G[i][j]
 *     - i before j → strengthen O[i][j]
 *   EMA decay: M[i][j] = α * M[i][j] + (1-α) * signal
 *
 * Lifecycle:
 *   1. relation_matrix_create(N)  — allocate host/device memory, initialize to 0
 *   2. relation_matrix_update(rm, sol, dim1) — update G/O from one solution (host)
 *   3. relation_matrix_upload(rm) — upload h_G/h_O to d_G/d_O
 *   4. relation_matrix_destroy(rm) — free memory
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"
#include <cstring>

// ============================================================
// Create / destroy
// ============================================================

inline RelationMatrix relation_matrix_create(int N, float decay = 0.95f) {
    RelationMatrix rm;
    rm.N = N;
    rm.decay = decay;
    rm.update_count = 0;
    
    size_t bytes = (size_t)N * N * sizeof(float);
    
    rm.h_G = new float[N * N];
    rm.h_O = new float[N * N];
    memset(rm.h_G, 0, bytes);
    memset(rm.h_O, 0, bytes);
    
    CUDA_CHECK(cudaMalloc(&rm.d_G, bytes));
    CUDA_CHECK(cudaMalloc(&rm.d_O, bytes));
    CUDA_CHECK(cudaMemset(rm.d_G, 0, bytes));
    CUDA_CHECK(cudaMemset(rm.d_O, 0, bytes));
    
    return rm;
}

inline void relation_matrix_destroy(RelationMatrix& rm) {
    delete[] rm.h_G;
    delete[] rm.h_O;
    CUDA_CHECK(cudaFree(rm.d_G));
    CUDA_CHECK(cudaFree(rm.d_O));
    rm.h_G = rm.h_O = nullptr;
    rm.d_G = rm.d_O = nullptr;
    rm.N = 0;
}

// ============================================================
// Update G/O from one solution (host)
// ============================================================
// sol: current best solution (already copied to host)
// dim1: number of rows in use
//
// Logic:
//   For each pair (val_a, val_b) in sol:
//     If on the same row → strengthen G[val_a][val_b]
//     If val_a appears before val_b → strengthen O[val_a][val_b]
//
// Note: element values val are meaningful only in [0, N)
//       For partition encoding (VRP), values are customer IDs
//       For single-row permutation (TSP), values are city IDs

template<typename Sol>
void relation_matrix_update(RelationMatrix& rm, const Sol& sol, int dim1) {
    int N = rm.N;
    float alpha = rm.decay;
    float signal_strength = 1.0f;
    
    // Decay all existing values
    for (int i = 0; i < N * N; i++) {
        rm.h_G[i] *= alpha;
        rm.h_O[i] *= alpha;
    }
    
    // Scan element-pair relations in the solution
    for (int r = 0; r < dim1; r++) {
        int sz = sol.dim2_sizes[r];
        for (int c1 = 0; c1 < sz; c1++) {
            int val_a = sol.data[r][c1];
            if (val_a < 0 || val_a >= N) continue;
            
            for (int c2 = c1 + 1; c2 < sz; c2++) {
                int val_b = sol.data[r][c2];
                if (val_b < 0 || val_b >= N) continue;
                
                // Same row → strengthen G (symmetric)
                rm.h_G[val_a * N + val_b] += (1.0f - alpha) * signal_strength;
                rm.h_G[val_b * N + val_a] += (1.0f - alpha) * signal_strength;
                
                // val_a before val_b → strengthen O[val_a][val_b]
                rm.h_O[val_a * N + val_b] += (1.0f - alpha) * signal_strength;
            }
        }
    }
    
    // Clamp to [0, 1]
    for (int i = 0; i < N * N; i++) {
        if (rm.h_G[i] > 1.0f) rm.h_G[i] = 1.0f;
        if (rm.h_O[i] > 1.0f) rm.h_O[i] = 1.0f;
    }
    
    rm.update_count++;
}

// ============================================================
// Upload to GPU
// ============================================================

inline void relation_matrix_upload(const RelationMatrix& rm) {
    size_t bytes = (size_t)rm.N * rm.N * sizeof(float);
    CUDA_CHECK(cudaMemcpy(rm.d_G, rm.h_G, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rm.d_O, rm.h_O, bytes, cudaMemcpyHostToDevice));
}
