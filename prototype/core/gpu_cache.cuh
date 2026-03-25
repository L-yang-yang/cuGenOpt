/**
 * gpu_cache.cuh - GPU global-memory hash table (generic cache component)
 * 
 * Design:
 *   - Open addressing, fixed capacity (power of 2), linear probing
 *   - key = uint64_t (hash computed by Problem)
 *   - value = float (single metric value)
 *   - Lock-free: race conditions allowed (cache semantics; occasional dirty reads OK)
 *   - Built-in atomic hit/miss counters
 * 
 * Usage:
 *   GpuCache cache = GpuCache::allocate(65536);   // host
 *   // ... pass cache as Problem member to kernels ...
 *   cache.print_stats();                           // host
 *   cache.destroy();                               // host
 * 
 * Reference: scute project LRUCache (key = metric_type + content_hash)
 */

#pragma once
#include "cuda_utils.cuh"
#include <cstdint>

// ============================================================
// Constants
// ============================================================

static constexpr uint64_t CACHE_EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;
static constexpr int CACHE_MAX_PROBE = 8;   // Max linear probing steps

// ============================================================
// GpuCache struct (POD, safe to copy to kernel)
// ============================================================

struct GpuCache {
    uint64_t* keys;             // GPU global memory
    float*    values;           // GPU global memory
    unsigned int* d_hits;       // Atomic counters (GPU)
    unsigned int* d_misses;     // Atomic counters (GPU)
    int capacity;               // Must be a power of 2
    int mask;                   // = capacity - 1
    
    // ---- Host operations ----
    
    static GpuCache allocate(int cap = 65536) {
        GpuCache c;
        c.capacity = cap;
        c.mask = cap - 1;
        CUDA_CHECK(cudaMalloc(&c.keys,     sizeof(uint64_t) * cap));
        CUDA_CHECK(cudaMalloc(&c.values,   sizeof(float) * cap));
        CUDA_CHECK(cudaMalloc(&c.d_hits,   sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&c.d_misses, sizeof(unsigned int)));
        c.clear();
        return c;
    }
    
    static GpuCache disabled() {
        GpuCache c;
        c.keys = nullptr;  c.values = nullptr;
        c.d_hits = nullptr; c.d_misses = nullptr;
        c.capacity = 0;  c.mask = 0;
        return c;
    }
    
    bool is_enabled() const { return keys != nullptr; }
    
    void clear() {
        CUDA_CHECK(cudaMemset(keys, 0xFF, sizeof(uint64_t) * capacity));
        CUDA_CHECK(cudaMemset(d_hits,   0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_misses, 0, sizeof(unsigned int)));
    }
    
    void destroy() {
        if (keys)     cudaFree(keys);
        if (values)   cudaFree(values);
        if (d_hits)   cudaFree(d_hits);
        if (d_misses) cudaFree(d_misses);
        keys = nullptr; values = nullptr;
        d_hits = nullptr; d_misses = nullptr;
    }
    
    void print_stats() const {
        if (!keys) { printf("  Cache: disabled\n"); return; }
        unsigned int h = 0, m = 0;
        CUDA_CHECK(cudaMemcpy(&h, d_hits,   sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&m, d_misses, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        unsigned int total = h + m;
        float rate = total > 0 ? (float)h / total * 100.0f : 0.0f;
        printf("  Cache: %u lookups | %u hits + %u misses | hit rate = %.1f%%\n",
               total, h, m, rate);
        printf("  Cache: capacity = %d entries (%.1f KB)\n",
               capacity, capacity * (sizeof(uint64_t) + sizeof(float)) / 1024.0f);
    }
};

// ============================================================
// Device functions: hash / lookup / insert
// ============================================================

/// FNV-1a hash over an ordered int sequence (e.g. customer IDs on a route)
__device__ inline uint64_t route_hash(const int* data, int len) {
    uint64_t h = 14695981039346656037ULL;   // FNV offset basis
    for (int i = 0; i < len; i++) {
        h ^= (uint64_t)(unsigned int)data[i];
        h *= 1099511628211ULL;               // FNV prime
    }
    return (h == CACHE_EMPTY_KEY) ? h - 1 : h;  // Avoid collision with sentinel value
}

/// Lookup: on hit returns true and writes out
__device__ inline bool cache_lookup(const GpuCache& c, uint64_t key, float& out) {
    int slot = (int)(key & (uint64_t)c.mask);
    for (int p = 0; p < CACHE_MAX_PROBE; p++) {
        int idx = (slot + p) & c.mask;
        uint64_t k = c.keys[idx];
        if (k == key) {
            out = c.values[idx];
            return true;
        }
        if (k == CACHE_EMPTY_KEY) return false;  // Empty slot -> key not present
    }
    return false;   // Probing exhausted
}

/// Insert: write key-value; same key overwrites; if probe full, evict first slot
__device__ inline void cache_insert(const GpuCache& c, uint64_t key, float value) {
    int slot = (int)(key & (uint64_t)c.mask);
    for (int p = 0; p < CACHE_MAX_PROBE; p++) {
        int idx = (slot + p) & c.mask;
        uint64_t k = c.keys[idx];
        if (k == CACHE_EMPTY_KEY || k == key) {
            c.keys[idx]   = key;
            c.values[idx] = value;
            return;
        }
    }
    // Probe full: evict first slot
    int idx = slot & c.mask;
    c.keys[idx]   = key;
    c.values[idx] = value;
}
