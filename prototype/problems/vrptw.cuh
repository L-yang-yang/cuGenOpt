/**
 * vrptw.cuh - Vehicle Routing Problem with Time Windows (VRPTW)
 *
 * CVRP plus time window constraints.
 * Encoding: multi-row perm partition (same as CVRP); data[r][j] = j-th customer on route r.
 * Objective: minimize total distance.
 * Constraints: (a) capacity, (b) time windows (arrival ≤ latest; early arrival waits).
 *
 * Validation instance: 8 customers, 3 vehicles; hand-crafted coords + windows with known feasible solution.
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

struct VRPTWProblem : ProblemBase<VRPTWProblem, 8, 64> {
    const float* d_dist;        // distance matrix [(n+1)*(n+1)] (includes depot)
    const float* d_demand;      // demand [n]
    const float* d_earliest;    // earliest service time [n+1] (includes depot)
    const float* d_latest;      // latest service time [n+1] (includes depot)
    const float* d_service;     // service time [n+1] (includes depot)
    int n;                      // number of customers (excludes depot)
    int stride;                 // n+1
    float capacity;
    int num_vehicles;
    int max_vehicles;
    
    __device__ float compute_route_dist(const int* route, int size) const {
        if (size == 0) return 0.0f;
        float dist = 0.0f;
        int prev = 0;
        for (int j = 0; j < size; j++) {
            int node = route[j] + 1;
            dist += d_dist[prev * stride + node];
            prev = node;
        }
        dist += d_dist[prev * stride + 0];
        return dist;
    }
    
    __device__ float calc_total_distance(const Sol& sol) const {
        float total = 0.0f;
        for (int r = 0; r < num_vehicles; r++)
            total += compute_route_dist(sol.data[r], sol.dim2_sizes[r]);
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
    
    __device__ float compute_penalty(const Sol& sol) const {
        float penalty = 0.0f;
        int active = 0;
        for (int r = 0; r < num_vehicles; r++) {
            int size = sol.dim2_sizes[r];
            if (size == 0) continue;
            active++;
            
            // Capacity constraint
            float load = 0.0f;
            for (int j = 0; j < size; j++)
                load += d_demand[sol.data[r][j]];
            if (load > capacity)
                penalty += (load - capacity) * 100.0f;
            
            // Time windows: simulate route travel
            float time = 0.0f;
            int prev = 0;
            for (int j = 0; j < size; j++) {
                int node = sol.data[r][j] + 1;
                float travel = d_dist[prev * stride + node];
                time += travel;
                // Wait if early
                if (time < d_earliest[node])
                    time = d_earliest[node];
                // Penalize lateness
                if (time > d_latest[node])
                    penalty += (time - d_latest[node]) * 50.0f;
                time += d_service[node];
                prev = node;
            }
            // Time window returning to depot
            float return_time = time + d_dist[prev * stride + 0];
            if (return_time > d_latest[0])
                penalty += (return_time - d_latest[0]) * 50.0f;
        }
        if (active > max_vehicles)
            penalty += (float)(active - max_vehicles) * 1000.0f;
        return penalty;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = num_vehicles;
        cfg.dim2_default = 0;
        fill_obj_config(cfg);
        cfg.cross_row_prob = 0.3f;
        cfg.row_mode = RowMode::Partition;
        cfg.total_elements = n;
        return cfg;
    }
    
    static constexpr size_t SMEM_LIMIT = 48 * 1024;
    
    size_t shared_mem_bytes() const {
        size_t dist_bytes = (size_t)stride * stride * sizeof(float);
        size_t aux_bytes  = (size_t)(n + 1) * 4 * sizeof(float);  // demand(n) + earliest/latest/service(n+1 each)
        size_t total = dist_bytes + aux_bytes;
        return total <= SMEM_LIMIT ? total : 0;
    }
    
    size_t working_set_bytes() const {
        return (size_t)stride * stride * sizeof(float) + (size_t)(n + 1) * 4 * sizeof(float);
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        float* sd = reinterpret_cast<float*>(smem);
        int dist_size = stride * stride;
        for (int i = tid; i < dist_size; i += bsz) sd[i] = d_dist[i];
        d_dist = sd;
        
        float* sdem = sd + dist_size;
        for (int i = tid; i < n; i += bsz) sdem[i] = d_demand[i];
        d_demand = sdem;
        
        float* se = sdem + n;
        int nn = n + 1;
        for (int i = tid; i < nn; i += bsz) se[i] = d_earliest[i];
        d_earliest = se;
        
        float* sl = se + nn;
        for (int i = tid; i < nn; i += bsz) sl[i] = d_latest[i];
        d_latest = sl;
        
        float* ss = sl + nn;
        for (int i = tid; i < nn; i += bsz) ss[i] = d_service[i];
        d_service = ss;
    }
    
    static VRPTWProblem create(const float* h_dist, const float* h_demand,
                                const float* h_earliest, const float* h_latest,
                                const float* h_service,
                                int n, float capacity,
                                int num_vehicles, int max_vehicles) {
        VRPTWProblem prob;
        prob.n = n;
        prob.stride = n + 1;
        prob.capacity = capacity;
        prob.num_vehicles = num_vehicles;
        prob.max_vehicles = max_vehicles;
        
        int nn = n + 1;
        float *dd, *ddem, *de, *dl, *ds;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * nn * nn));
        CUDA_CHECK(cudaMemcpy(dd, h_dist, sizeof(float) * nn * nn, cudaMemcpyHostToDevice));
        prob.d_dist = dd;
        
        CUDA_CHECK(cudaMalloc(&ddem, sizeof(float) * n));
        CUDA_CHECK(cudaMemcpy(ddem, h_demand, sizeof(float) * n, cudaMemcpyHostToDevice));
        prob.d_demand = ddem;
        
        CUDA_CHECK(cudaMalloc(&de, sizeof(float) * nn));
        CUDA_CHECK(cudaMemcpy(de, h_earliest, sizeof(float) * nn, cudaMemcpyHostToDevice));
        prob.d_earliest = de;
        
        CUDA_CHECK(cudaMalloc(&dl, sizeof(float) * nn));
        CUDA_CHECK(cudaMemcpy(dl, h_latest, sizeof(float) * nn, cudaMemcpyHostToDevice));
        prob.d_latest = dl;
        
        CUDA_CHECK(cudaMalloc(&ds, sizeof(float) * nn));
        CUDA_CHECK(cudaMemcpy(ds, h_service, sizeof(float) * nn, cudaMemcpyHostToDevice));
        prob.d_service = ds;
        
        return prob;
    }
    
    void destroy() {
        if (d_dist)     { cudaFree(const_cast<float*>(d_dist));     d_dist = nullptr; }
        if (d_demand)   { cudaFree(const_cast<float*>(d_demand));   d_demand = nullptr; }
        if (d_earliest) { cudaFree(const_cast<float*>(d_earliest)); d_earliest = nullptr; }
        if (d_latest)   { cudaFree(const_cast<float*>(d_latest));   d_latest = nullptr; }
        if (d_service)  { cudaFree(const_cast<float*>(d_service));  d_service = nullptr; }
    }
};
