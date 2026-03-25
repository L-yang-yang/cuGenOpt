/**
 * jsp.cuh - Job Shop Scheduling Problem (JSSP)
 *
 * J jobs, each with O operations; each op specifies machine and duration.
 *
 * === Encoding A: multi-row Integer (time-table encoding) ===
 * JSPProblem: data[j][i] = start time of job j's i-th operation
 *   dim1 = num_jobs, dim2_default = num_ops
 *   row_mode = Fixed (no ROW_SPLIT/ROW_MERGE)
 *   Each row is a fixed op sequence for one job; row length is fixed.
 *
 * === Encoding B: Permutation multiset (operation sequence encoding) ===
 * JSPPermProblem: data[0][k] = job id (0..J-1), length J*O
 *   Value j appears O times. Left-to-right scan: t-th occurrence of j is job j's t-th op.
 *   dim1 = 1, dim2_default = J*O, perm_repeat_count = O
 *   Standard permutation ops (swap/reverse/insert) preserve multiset structure.
 *
 * Objective: minimize makespan (max completion time over jobs).
 * Constraints:
 *   (a) Precedence: ops of the same job must run in order.
 *   (b) Machine conflict: one op per machine at a time.
 *
 * Validation instance: custom 3 jobs × 3 machines (3x3), optimal makespan = 12
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

// ============================================================
// Encoding A: multi-row Integer (time-table encoding)
// ============================================================

struct JSPProblem : ProblemBase<JSPProblem, 8, 16> {
    const int*   d_machine;     // machine per op [J*O]
    const float* d_duration;    // op duration [J*O]
    int num_jobs;               // number of jobs J
    int num_ops;                // ops per job O
    int num_machines;           // number of machines M
    int time_horizon;           // time horizon upper bound
    
    __device__ float calc_makespan(const Sol& sol) const {
        float makespan = 0.0f;
        for (int j = 0; j < num_jobs; j++) {
            int last = num_ops - 1;
            float end = (float)sol.data[j][last] + d_duration[j * num_ops + last];
            if (end > makespan) makespan = end;
        }
        return makespan;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return calc_makespan(sol);
            default: return 0.0f;
        }
    }
    
    __device__ float compute_penalty(const Sol& sol) const {
        float penalty = 0.0f;
        
        // (a) Precedence constraints
        for (int j = 0; j < num_jobs; j++) {
            for (int i = 1; i < num_ops; i++) {
                float prev_end = (float)sol.data[j][i-1] + d_duration[j * num_ops + (i-1)];
                float curr_start = (float)sol.data[j][i];
                if (curr_start < prev_end)
                    penalty += (prev_end - curr_start) * 10.0f;
            }
        }
        
        // (b) Machine conflict constraints
        int total = num_jobs * num_ops;
        for (int a = 0; a < total; a++) {
            int ja = a / num_ops, ia = a % num_ops;
            int m_a = d_machine[a];
            float s_a = (float)sol.data[ja][ia];
            float e_a = s_a + d_duration[a];
            for (int b = a + 1; b < total; b++) {
                if (d_machine[b] != m_a) continue;
                int jb = b / num_ops, ib = b % num_ops;
                float s_b = (float)sol.data[jb][ib];
                float e_b = s_b + d_duration[b];
                float overlap = fminf(e_a, e_b) - fmaxf(s_a, s_b);
                if (overlap > 0.0f)
                    penalty += overlap * 10.0f;
            }
        }
        
        return penalty;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Integer;
        cfg.dim1 = num_jobs;
        cfg.dim2_default = num_ops;
        cfg.value_lower_bound = 0;
        cfg.value_upper_bound = time_horizon - 1;
        cfg.row_mode = RowMode::Fixed;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        int total = num_jobs * num_ops;
        return (size_t)total * (sizeof(int) + sizeof(float));
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        int total = num_jobs * num_ops;
        int* sm = reinterpret_cast<int*>(smem);
        for (int i = tid; i < total; i += bsz) sm[i] = d_machine[i];
        d_machine = sm;
        
        float* sd = reinterpret_cast<float*>(sm + total);
        for (int i = tid; i < total; i += bsz) sd[i] = d_duration[i];
        d_duration = sd;
    }
    
    static JSPProblem create(const int* h_machine, const float* h_duration,
                              int num_jobs, int num_ops, int num_machines,
                              int time_horizon) {
        JSPProblem prob;
        prob.num_jobs = num_jobs;
        prob.num_ops = num_ops;
        prob.num_machines = num_machines;
        prob.time_horizon = time_horizon;
        
        int total = num_jobs * num_ops;
        int* dm;
        CUDA_CHECK(cudaMalloc(&dm, sizeof(int) * total));
        CUDA_CHECK(cudaMemcpy(dm, h_machine, sizeof(int) * total, cudaMemcpyHostToDevice));
        prob.d_machine = dm;
        
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * total));
        CUDA_CHECK(cudaMemcpy(dd, h_duration, sizeof(float) * total, cudaMemcpyHostToDevice));
        prob.d_duration = dd;
        
        return prob;
    }
    
    void destroy() {
        if (d_machine)  { cudaFree(const_cast<int*>(d_machine));     d_machine = nullptr; }
        if (d_duration) { cudaFree(const_cast<float*>(d_duration));  d_duration = nullptr; }
    }
};

// ============================================================
// Encoding B: Permutation multiset (operation sequence encoding)
// ============================================================
// data[0] is a length-J*O sequence with values in [0, J), each appearing O times.
// Left-to-right: t-th occurrence of j schedules job j's t-th operation.
// Greedy decode: each op at earliest feasible time (precedence + machine free).

struct JSPPermProblem : ProblemBase<JSPPermProblem, 1, 64> {
    const int*   d_machine;     // machine per op [J*O]
    const float* d_duration;    // op duration [J*O]
    int num_jobs;
    int num_ops;
    int num_machines;
    
    // Greedy decode: build schedule from permutation, return makespan
    __device__ float decode_and_makespan(const Sol& sol) const {
        int total = num_jobs * num_ops;
        int size = sol.dim2_sizes[0];
        if (size < total) return 1e9f;
        
        float job_avail[8];     // earliest start for next op of each job
        float mach_avail[8];    // earliest machine free time
        int   job_next_op[8];   // next op index to schedule per job
        
        for (int j = 0; j < num_jobs; j++) { job_avail[j] = 0.0f; job_next_op[j] = 0; }
        for (int m = 0; m < num_machines; m++) mach_avail[m] = 0.0f;
        
        float makespan = 0.0f;
        for (int k = 0; k < total; k++) {
            int j = sol.data[0][k];
            if (j < 0 || j >= num_jobs) return 1e9f;
            int op = job_next_op[j];
            if (op >= num_ops) continue;  // job already fully scheduled
            
            int flat = j * num_ops + op;
            int m = d_machine[flat];
            float dur = d_duration[flat];
            
            // Earliest start = max(job predecessor done, machine free)
            float start = fmaxf(job_avail[j], mach_avail[m]);
            float end = start + dur;
            
            job_avail[j] = end;
            mach_avail[m] = end;
            job_next_op[j] = op + 1;
            
            if (end > makespan) makespan = end;
        }
        
        return makespan;
    }
    
    static constexpr ObjDef OBJ_DEFS[] = {
        {ObjDir::Minimize, 1.0f, 0.0f},
    };
    __device__ float compute_obj(int idx, const Sol& sol) const {
        switch (idx) {
            case 0: return decode_and_makespan(sol);
            default: return 0.0f;
        }
    }
    
    // Greedy decode satisfies constraints; penalty is always 0
    __device__ float compute_penalty(const Sol& sol) const {
        return 0.0f;
    }
    
    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;
        cfg.dim2_default = num_jobs * num_ops;
        cfg.perm_repeat_count = num_ops;
        fill_obj_config(cfg);
        return cfg;
    }
    
    size_t shared_mem_bytes() const {
        int total = num_jobs * num_ops;
        return (size_t)total * (sizeof(int) + sizeof(float));
    }
    
    __device__ void load_shared(char* smem, int tid, int bsz) {
        int total = num_jobs * num_ops;
        int* sm = reinterpret_cast<int*>(smem);
        for (int i = tid; i < total; i += bsz) sm[i] = d_machine[i];
        d_machine = sm;
        
        float* sd = reinterpret_cast<float*>(sm + total);
        for (int i = tid; i < total; i += bsz) sd[i] = d_duration[i];
        d_duration = sd;
    }
    
    static JSPPermProblem create(const int* h_machine, const float* h_duration,
                                  int num_jobs, int num_ops, int num_machines) {
        JSPPermProblem prob;
        prob.num_jobs = num_jobs;
        prob.num_ops = num_ops;
        prob.num_machines = num_machines;
        
        int total = num_jobs * num_ops;
        int* dm;
        CUDA_CHECK(cudaMalloc(&dm, sizeof(int) * total));
        CUDA_CHECK(cudaMemcpy(dm, h_machine, sizeof(int) * total, cudaMemcpyHostToDevice));
        prob.d_machine = dm;
        
        float* dd;
        CUDA_CHECK(cudaMalloc(&dd, sizeof(float) * total));
        CUDA_CHECK(cudaMemcpy(dd, h_duration, sizeof(float) * total, cudaMemcpyHostToDevice));
        prob.d_duration = dd;
        
        return prob;
    }
    
    void destroy() {
        if (d_machine)  { cudaFree(const_cast<int*>(d_machine));     d_machine = nullptr; }
        if (d_duration) { cudaFree(const_cast<float*>(d_duration));  d_duration = nullptr; }
    }
};
