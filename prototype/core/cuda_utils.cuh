/**
 * cuda_utils.cuh - CUDA utilities
 * 
 * Responsibilities: error checking, device info, random number utilities
 * Rule: every CUDA API call must be wrapped with CUDA_CHECK
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <curand_kernel.h>

// ============================================================
// Error checking
// ============================================================

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// Check after kernel launch (catches async errors)
#define CUDA_CHECK_LAST() do {                                      \
    cudaError_t err = cudaGetLastError();                            \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// ============================================================
// Device info
// ============================================================

inline void print_device_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU: %s\n", prop.name);
    printf("  SM count:       %d\n", prop.multiProcessorCount);
    printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Shared mem/blk: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Global mem:     %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Compute cap:    %d.%d\n", prop.major, prop.minor);
}

// ============================================================
// Random number utilities (device-side)
// ============================================================

// Initialize curand state: one per thread
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Device-side: random integer in [0, bound)
__device__ inline int rand_int(curandState* state, int bound) {
    return curand(state) % bound;
}

// Device-side: Fisher-Yates shuffle of arr[0..n-1]
__device__ inline void shuffle(int* arr, int n, curandState* state) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand_int(state, i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// ============================================================
// Kernel launch grid sizing
// ============================================================

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// Compute suitable number of blocks
inline int calc_grid_size(int n, int block_size = 256) {
    return div_ceil(n, block_size);
}
