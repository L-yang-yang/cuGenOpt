/**
 * types.cuh - Core type definitions
 * 
 * Contains: encoding types, Solution template, ProblemConfig/SolverConfig,
 *           SeqRegistry (AOS sequence-level weights), KStepConfig (multi-step execution),
 *           RelationMatrix (G/O relation matrix), ProblemBase (CRTP base class)
 */

#pragma once
#include <cstdio>
#include "cuda_utils.cuh"

// ============================================================
// Compile-time constants
// ============================================================
constexpr int MAX_OBJ = 4;    // Max 4 objectives (16 bytes, not worth templatizing)
constexpr int MAX_SEQ = 32;   // Max sequences (built-in ~16 + custom ops ≤8, with margin)
constexpr int MAX_K   = 3;    // Max steps for multi-step execution (K=1,2,3)
// AOS weight bounds
constexpr float AOS_WEIGHT_FLOOR = 0.05f;  // Minimum weight floor (ensures sufficient exploration)
constexpr float AOS_WEIGHT_CAP   = 0.35f;  // Maximum weight cap (prevents winner-take-all)

// ============================================================
// Enum types
// ============================================================

enum class EncodingType {
    Permutation,    // Permutation: elements are unique
    Binary,         // 0-1: flip is the main operator
    Integer         // Bounded integers
};

enum class RowMode {
    Single,     // dim1=1, single row (most problems: TSP/QAP/Knapsack, etc.)
    Fixed,      // dim1>1, equal row lengths fixed (JSP-Int/Schedule; SPLIT/MERGE disallowed)
    Partition   // dim1>1, elements partitioned across rows, variable row lengths (CVRP/VRPTW)
};

enum class ObjDir {
    Minimize,
    Maximize
};

// Multi-objective comparison mode
enum class CompareMode {
    Weighted,       // Weighted sum: sum(weight[i] * obj[i]), lower is better
    Lexicographic   // Lexicographic: compare objectives by priority order
};

enum class MigrateStrategy {
    Ring,       // Ring: each island's best → neighbor's worst (slow spread, high diversity)
    TopN,       // Global Top-N round-robin (fast spread, strong convergence)
    Hybrid      // Hybrid: Top-N replaces worst + Ring replaces second-worst
};

// v5.0: multi-GPU coordination — solution injection mode
enum class MultiGpuInjectMode {
    OneIsland,   // Inject into worst of 1 island (conservative, preserves diversity)
    HalfIslands, // Inject into worst on num_islands/2 islands (balanced)
    AllIslands   // Inject into worst on all islands (aggressive, fast spread)
};

// v5.0 option B3: InjectBuffer — passive injection buffer
// GPU has no awareness; CPU writes synchronously; GPU checks and applies in migrate_kernel
// Design notes:
// 1. Use synchronous cudaMemcpy to avoid conflicts with solve() stream/Graph
// 2. Write order: solution first, then flag; GPU atomic flag read ensures consistency
// 3. Fully decoupled: does not depend on any internal state of solve()
template<typename Sol>
struct InjectBuffer {
    Sol*  d_solution = nullptr;  // Device solution buffer (single solution)
    int*  d_flag     = nullptr;  // Device flag: 0=empty, 1=new solution
    int   owner_gpu  = 0;       // GPU that owns the allocation
    
    // Allocate InjectBuffer (on given GPU)
    static InjectBuffer<Sol> allocate(int gpu_id) {
        InjectBuffer<Sol> buf;
        buf.owner_gpu = gpu_id;
        
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        CUDA_CHECK(cudaSetDevice(gpu_id));
        
        CUDA_CHECK(cudaMalloc(&buf.d_solution, sizeof(Sol)));
        CUDA_CHECK(cudaMalloc(&buf.d_flag, sizeof(int)));
        
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(buf.d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaSetDevice(orig_device));
        
        return buf;
    }
    
    // Free InjectBuffer (switches to owner GPU before freeing)
    void destroy() {
        if (d_solution || d_flag) {
            int orig_device;
            cudaGetDevice(&orig_device);
            cudaSetDevice(owner_gpu);
            if (d_solution) { cudaFree(d_solution); d_solution = nullptr; }
            if (d_flag)     { cudaFree(d_flag);     d_flag = nullptr;     }
            cudaSetDevice(orig_device);
        }
    }
    
    // CPU-side write of new solution
    // Note: synchronous cudaMemcpy avoids stream conflicts with solve()
    // Order: write solution first, then flag (GPU atomic flag read avoids half-written reads)
    void write_sync(const Sol& sol, int target_gpu) {
        int orig_device;
        CUDA_CHECK(cudaGetDevice(&orig_device));
        CUDA_CHECK(cudaSetDevice(target_gpu));
        
        CUDA_CHECK(cudaMemcpy(d_solution, &sol, sizeof(Sol), cudaMemcpyHostToDevice));
        int flag = 1;
        CUDA_CHECK(cudaMemcpy(d_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaSetDevice(orig_device));
    }
};


// ============================================================
// SeqID — unified OperationSequence IDs
// ============================================================
// Each SeqID maps to one concrete search operation (atomic or multi-step)
// AOS weight granularity = SeqID (independent weight per sequence)
//
// Naming: SEQ_{encoding}_{operation}
// Row-level ops shared across encodings use unified numbering

namespace seq {

// --- Permutation in-row (element-level) ---
constexpr int SEQ_PERM_SWAP           = 0;   // swap two positions
constexpr int SEQ_PERM_REVERSE        = 1;   // 2-opt (reverse segment)
constexpr int SEQ_PERM_INSERT         = 2;   // insert (move to new position)
constexpr int SEQ_PERM_3OPT           = 3;   // 3-opt (reconnect after 3 edges)

// --- Permutation in-row (segment-level) ---
constexpr int SEQ_PERM_OR_OPT         = 4;   // or-opt (move k consecutive elements)

// --- Permutation in-row (combo-level) ---
constexpr int SEQ_PERM_DOUBLE_SWAP    = 30;  // two consecutive swaps (same row)
constexpr int SEQ_PERM_TRIPLE_SWAP    = 31;  // three consecutive swaps (same row)

// --- Permutation cross-row (element-level) ---
constexpr int SEQ_PERM_CROSS_RELOCATE = 5;   // single element moves row
constexpr int SEQ_PERM_CROSS_SWAP     = 6;   // single element swaps rows

// --- Permutation cross-row (segment-level) ---
constexpr int SEQ_PERM_SEG_RELOCATE   = 7;   // segment moves row
constexpr int SEQ_PERM_SEG_SWAP       = 8;   // segment swaps rows (2-opt*)
constexpr int SEQ_PERM_CROSS_EXCHANGE = 9;   // segment exchange (order preserved)

// --- Binary in-row (element-level) ---
constexpr int SEQ_BIN_FLIP            = 0;   // flip one bit
constexpr int SEQ_BIN_SWAP            = 1;   // swap two bits

// --- Binary in-row (segment-level) ---
constexpr int SEQ_BIN_SEG_FLIP        = 2;   // flip k consecutive bits
constexpr int SEQ_BIN_K_FLIP          = 3;   // flip k random bits at once

// --- Binary cross-row ---
constexpr int SEQ_BIN_CROSS_SWAP      = 4;   // swap one bit per row across two rows
constexpr int SEQ_BIN_SEG_CROSS_SWAP  = 5;   // swap a segment from each row

// --- Shared: row-level (encoding-agnostic) ---
constexpr int SEQ_ROW_SWAP            = 10;  // swap two rows
constexpr int SEQ_ROW_REVERSE         = 11;  // reverse row order
constexpr int SEQ_ROW_SPLIT           = 12;  // split one row into two
constexpr int SEQ_ROW_MERGE           = 13;  // merge two rows

// --- Special ---
constexpr int SEQ_PERTURBATION        = 14;  // perturbation (multi-step, irreversible)

// --- Integer in-row (element-level) ---
constexpr int SEQ_INT_RANDOM_RESET    = 0;   // reset one position to random in [lb, ub]
constexpr int SEQ_INT_DELTA           = 1;   // one position ±k (clamped to [lb, ub])
constexpr int SEQ_INT_SWAP            = 2;   // swap values at two positions

// --- Integer in-row (segment-level) ---
constexpr int SEQ_INT_SEG_RESET       = 3;   // reset k consecutive positions
constexpr int SEQ_INT_K_DELTA         = 4;   // k positions each ±1 at random

// --- Integer cross-row ---
constexpr int SEQ_INT_CROSS_SWAP      = 5;   // swap one position per row across two rows

// --- LNS (large neighborhood search) ---
constexpr int SEQ_LNS_SEGMENT_SHUFFLE = 20;  // shuffle a contiguous segment
constexpr int SEQ_LNS_SCATTER_SHUFFLE = 21;  // shuffle a scattered set of positions
constexpr int SEQ_LNS_GUIDED_REBUILD  = 22;  // guided rebuild from relation matrix

}  // namespace seq

// ============================================================
// RelationMatrix — G/O relation matrix (GPU global memory)
// ============================================================
// G[i][j]: grouping tendency of elements i and j (symmetric; higher → more same-group)
// O[i][j]: tendency for element i to precede j (asymmetric)
// Stored as a 1D row-major array [N * N]
// For small N<200 use dense directly; P2 may add sparsification
//
// Updated on: host, between batches
// Read in: kernel for SEQ_LNS_GUIDED_REBUILD

struct RelationMatrix {
    float* d_G;           // G matrix on GPU [N * N]
    float* d_O;           // O matrix on GPU [N * N]
    float* h_G;           // G matrix on host [N * N] (for upload after update)
    float* h_O;           // O matrix on host [N * N]
    int    N;             // total number of elements
    float  decay;         // decay factor α (default 0.95)
    int    update_count;  // number of updates so far (for cold-start logic)
};

// ============================================================
// SeqRegistry — runtime-available sequence registry
// ============================================================
// Which sequences are available is determined from EncodingType and dim1
// Passed to GPU for sample_sequence()

enum class SeqCategory : int {
    InRow    = 0,   // within-row operators (swap, reverse, insert, ...)
    CrossRow = 1,   // cross-row operators (cross_relocate, cross_swap, seg_relocate, ...)
    RowLevel = 2,   // row-level operators (row_swap, row_reverse, split, merge)
    LNS      = 3,   // large neighborhood search
};

struct SeqRegistry {
    int   ids[MAX_SEQ];       // SeqID list of available sequences
    int   count;              // number of available sequences
    float weights[MAX_SEQ];   // current weight per sequence (unnormalized; lazy normalization)
    float weights_sum;        // sum of weights (cached for lazy normalization)
    float max_w[MAX_SEQ];     // per-sequence weight cap (0 = unlimited, use global cap)
    SeqCategory categories[MAX_SEQ];  // category per sequence (for constraint-directed mode)
};

// ============================================================
// KStepConfig — step-count selection for multi-step execution
// ============================================================
// K=1: single step (current behavior); K=2/3: run several sequences then evaluate
// First layer of the two-level weight system
//
// Adaptive policy:
//   - Initially K=1 has large weight (conservative), K>1 small
//   - If K>1 yields improvement → increase that K's weight
//   - Long stagnation → reset / boost K>1 weights (escape local optima)

struct KStepConfig {
    float weights[MAX_K];     // sampling weights for K=1,2,3 (normalized)
    int   stagnation_count;   // consecutive batches without improvement (triggers reset)
    int   stagnation_limit;   // threshold to trigger reset (default 5 batches)
};

// Build default K-step configuration
inline KStepConfig build_kstep_config() {
    KStepConfig kc;
    kc.weights[0] = 0.80f;   // K=1: dominates initially
    kc.weights[1] = 0.15f;   // K=2: little exploration
    kc.weights[2] = 0.05f;   // K=3: minimal exploration
    kc.stagnation_count = 0;
    kc.stagnation_limit = 5;
    return kc;
};

// ============================================================
// ProblemProfile — problem profile inferred from structural features
// ============================================================
// Layer 1: structure-only inference (no semantics), drives operator registration and initial weights
// Future layer 2: finer profiles (e.g. multi-attribute, high constraint)

enum class ScaleClass  { Small, Medium, Large };
enum class StructClass { SingleSeq, MultiFixed, MultiPartition };

struct ProblemProfile {
    EncodingType  encoding;
    ScaleClass    scale;
    StructClass   structure;
    float         cross_row_prob;
};

// classify_problem() is defined after ProblemConfig

// ============================================================
// Weight presets — driven by ScaleClass
// ============================================================

struct WeightPreset {
    float w_cubic;
    float w_quadratic;
    float w_lns;
    float lns_cap;
};

inline WeightPreset get_weight_preset(ScaleClass scale) {
    switch (scale) {
        case ScaleClass::Small:  return { 0.50f, 0.80f, 0.006f, 0.01f };
        case ScaleClass::Medium: return { 0.30f, 0.70f, 0.004f, 0.01f };
        case ScaleClass::Large:  return { 0.05f, 0.30f, 0.001f, 0.01f };
    }
    return { 0.50f, 0.80f, 0.006f, 0.01f };
}

// classify_problem() and build_seq_registry() are defined after ProblemConfig

// ============================================================
// Solution<D1, D2> — templated solution representation
// ============================================================
// D1: max number of rows (TSP=1, VRP≤16, Schedule≤8)
// D2: max columns per row (TSP≤64, knapsack≤32)
// Each Problem picks the smallest sufficient D1/D2; compiler emits a compact layout

template<int D1, int D2>
struct Solution {
    static constexpr int DIM1 = D1;   // compile-time max rows
    static constexpr int DIM2 = D2;   // compile-time max columns per row
    int   data[D1][D2];               // D1×D2×4 bytes
    int   dim2_sizes[D1];             // D1×4 bytes
    float objectives[MAX_OBJ];        // 16 bytes (fixed)
    float penalty;                    // 4 bytes
};

// ============================================================
// ProblemConfig — runtime metadata for a problem
// ============================================================

struct ProblemConfig {
    EncodingType encoding;
    int   dim1;                       // actual number of rows used (≤ D1)
    int   dim2_default;               // actual number of columns used (≤ D2)
    int   num_objectives;
    ObjDir obj_dirs[MAX_OBJ];
    float obj_weights[MAX_OBJ];       // weights in Weighted mode
    // Multi-objective comparison
    CompareMode compare_mode = CompareMode::Weighted;
    int   obj_priority[MAX_OBJ] = {0, 1, 2, 3};  // comparison order in Lexicographic mode (indices)
    float obj_tolerance[MAX_OBJ] = {0.0f, 0.0f, 0.0f, 0.0f};  // lexicographic tolerance: |diff| ≤ tol ⇒ tie
    int   value_lower_bound;
    int   value_upper_bound;
    // v3.4: unified row mode
    RowMode row_mode      = RowMode::Single;  // row mode (Single/Fixed/Partition)
    float cross_row_prob  = 0.0f;     // probability of cross-row moves (0 = within-row only)
    int   total_elements  = 0;        // total elements in Partition mode
    int   perm_repeat_count = 1;      // repeats per value in permutation (1 = standard; >1 = multiset)
};

// ============================================================
// SolverConfig — solver parameters
// ============================================================

struct SolverConfig {
    int   pop_size         = 0;       // population size (0 = auto to max GPU parallelism)
    int   max_gen          = 1000;
    float mutation_rate    = 0.1f;
    unsigned seed          = 42;
    bool  verbose          = true;
    int   print_every      = 100;
    // Island model
    int   num_islands      = 1;       // 0 = adaptive, 1 = pure hill climbing (no islands), >1 = island model
    int   migrate_interval = 100;     // migrate every this many generations
    MigrateStrategy migrate_strategy = MigrateStrategy::Hybrid;
    // Simulated annealing
    float sa_temp_init     = 0.0f;    // initial temperature (0 = disable SA, hill climb only)
    float sa_alpha         = 0.998f;  // cooling rate (multiply by alpha each generation)
    // v1.0: crossover
    float crossover_rate   = 0.1f;    // probability of crossover per generation (vs mutation)
    // v2.0: adaptive operator selection
    bool  use_aos          = false;   // enable AOS (update operator weights between batches)
    float aos_weight_floor = AOS_WEIGHT_FLOOR;  // runtime-overridable floor
    float aos_weight_cap   = AOS_WEIGHT_CAP;    // runtime-overridable cap
    // v2.1: initial solution strategy
    int   init_oversample  = 4;       // oversampling factor (1 = no sampling selection, pure random)
    float init_random_ratio = 0.3f;   // fraction of purely random solutions (diversity floor)
    // v3.0: engineering usability
    float time_limit_sec   = 0.0f;   // time limit in seconds (0 = none, run to max_gen)
    int   stagnation_limit = 0;      // convergence: reheat after this many batches without improvement (0 = off)
    float reheat_ratio     = 0.5f;   // on reheat, fraction of initial temperature to restore
    // v3.5: CUDA Graph
    bool  use_cuda_graph   = false;  // enable CUDA Graph (fewer kernel launch overheads)
    // v3.6: AOS update frequency
    int   aos_update_interval = 10;  // update AOS weights every this many batches (lower cudaMemcpy sync rate)
    // v4.0: constraint-directed + phased search
    bool  use_constraint_directed = false;  // constraint-directed mode (scale cross-row weights by penalty ratio)
    bool  use_phased_search       = false;  // phased search (adjust global floor/cap by progress)
    // Phased search: three-phase thresholds
    float phase_explore_end  = 0.30f;  // end of exploration phase (progress fraction)
    float phase_refine_start = 0.70f;  // start of refinement phase (progress fraction)
    // Constraint-directed parameters
    float constraint_boost_max = 2.5f; // max multiplier boost for cross-row cap under high constraint
    // v5.0: multi-GPU cooperation
    int   num_gpus             = 1;    // number of GPUs (1 = single GPU, >1 = multi-GPU)
    float multi_gpu_interval_sec = 10.0f;  // interval in seconds to exchange best solutions across GPUs
    MultiGpuInjectMode multi_gpu_inject_mode = MultiGpuInjectMode::HalfIslands;  // injection mode
};

// ============================================================
// classify_problem — infer problem profile from ProblemConfig
// ============================================================

inline ProblemProfile classify_problem(const ProblemConfig& pcfg) {
    ProblemProfile p;
    p.encoding = pcfg.encoding;

    if      (pcfg.dim2_default <= 100) p.scale = ScaleClass::Small;
    else if (pcfg.dim2_default <= 250) p.scale = ScaleClass::Medium;
    else                               p.scale = ScaleClass::Large;

    if (pcfg.dim1 <= 1)
        p.structure = StructClass::SingleSeq;
    else if (pcfg.row_mode == RowMode::Partition)
        p.structure = StructClass::MultiPartition;
    else
        p.structure = StructClass::MultiFixed;

    p.cross_row_prob = pcfg.cross_row_prob;
    return p;
}

// ============================================================
// build_seq_registry — operator registration driven by ProblemProfile
// ============================================================

inline SeqRegistry build_seq_registry(const ProblemProfile& prof) {
    SeqRegistry reg;
    reg.count = 0;
    for (int i = 0; i < MAX_SEQ; i++) {
        reg.ids[i] = -1; reg.weights[i] = 0.0f;
        reg.max_w[i] = 0.0f; reg.categories[i] = SeqCategory::InRow;
    }

    auto add = [&](int id, float w, SeqCategory cat, float cap = 0.0f) {
        if (reg.count >= MAX_SEQ) {
            printf("[WARN] SeqRegistry full (MAX_SEQ=%d), ignoring SeqID %d\n", MAX_SEQ, id);
            return;
        }
        reg.ids[reg.count] = id;
        reg.weights[reg.count] = w;
        reg.max_w[reg.count] = cap;
        reg.categories[reg.count] = cat;
        reg.count++;
    };

    WeightPreset wp = get_weight_preset(prof.scale);
    bool multi_row = (prof.structure != StructClass::SingleSeq);
    float cr = prof.cross_row_prob;

    if (prof.encoding == EncodingType::Permutation) {
        add(seq::SEQ_PERM_SWAP,    1.0f, SeqCategory::InRow);
        add(seq::SEQ_PERM_REVERSE, 1.0f, SeqCategory::InRow);
        add(seq::SEQ_PERM_INSERT,  1.0f, SeqCategory::InRow);
        add(seq::SEQ_PERM_DOUBLE_SWAP, 0.5f, SeqCategory::InRow);
        add(seq::SEQ_PERM_TRIPLE_SWAP, 0.3f, SeqCategory::InRow);

        add(seq::SEQ_PERM_3OPT,   wp.w_cubic,     SeqCategory::InRow);
        add(seq::SEQ_PERM_OR_OPT, wp.w_quadratic,  SeqCategory::InRow);

        if (multi_row && cr > 0.0f) {
            add(seq::SEQ_PERM_CROSS_RELOCATE, 0.6f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_CROSS_SWAP,     0.6f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_SEG_RELOCATE,   0.5f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_SEG_SWAP,       0.5f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_PERM_CROSS_EXCHANGE,  0.4f * cr, SeqCategory::CrossRow);
        }
        if (multi_row) {
            add(seq::SEQ_ROW_SWAP,    0.3f, SeqCategory::RowLevel);
            add(seq::SEQ_ROW_REVERSE, 0.2f, SeqCategory::RowLevel);
            if (prof.structure == StructClass::MultiPartition) {
                add(seq::SEQ_ROW_SPLIT,  0.2f, SeqCategory::RowLevel);
                add(seq::SEQ_ROW_MERGE,  0.2f, SeqCategory::RowLevel);
            }
        }
        add(seq::SEQ_LNS_SEGMENT_SHUFFLE, wp.w_lns, SeqCategory::LNS, wp.lns_cap);
        add(seq::SEQ_LNS_SCATTER_SHUFFLE, wp.w_lns, SeqCategory::LNS, wp.lns_cap);
        add(seq::SEQ_LNS_GUIDED_REBUILD,  wp.w_lns, SeqCategory::LNS, wp.lns_cap);
    }
    else if (prof.encoding == EncodingType::Binary) {
        add(seq::SEQ_BIN_FLIP, 1.0f, SeqCategory::InRow);
        add(seq::SEQ_BIN_SWAP, 0.8f, SeqCategory::InRow);
        add(seq::SEQ_BIN_SEG_FLIP, 0.6f, SeqCategory::InRow);
        add(seq::SEQ_BIN_K_FLIP,   0.6f, SeqCategory::InRow);
        if (multi_row && cr > 0.0f) {
            add(seq::SEQ_BIN_CROSS_SWAP,     0.5f * cr, SeqCategory::CrossRow);
            add(seq::SEQ_BIN_SEG_CROSS_SWAP, 0.4f * cr, SeqCategory::CrossRow);
        }
        if (multi_row) {
            add(seq::SEQ_ROW_SWAP,    0.3f, SeqCategory::RowLevel);
            add(seq::SEQ_ROW_REVERSE, 0.2f, SeqCategory::RowLevel);
            if (prof.structure == StructClass::MultiPartition) {
                add(seq::SEQ_ROW_SPLIT,  0.2f, SeqCategory::RowLevel);
                add(seq::SEQ_ROW_MERGE,  0.2f, SeqCategory::RowLevel);
            }
        }
    }
    else if (prof.encoding == EncodingType::Integer) {
        add(seq::SEQ_INT_RANDOM_RESET, 1.0f, SeqCategory::InRow);
        add(seq::SEQ_INT_DELTA,        1.0f, SeqCategory::InRow);
        add(seq::SEQ_INT_SWAP,         0.8f, SeqCategory::InRow);
        add(seq::SEQ_INT_SEG_RESET,    0.6f, SeqCategory::InRow);
        add(seq::SEQ_INT_K_DELTA,      0.6f, SeqCategory::InRow);
        if (multi_row && cr > 0.0f) {
            add(seq::SEQ_INT_CROSS_SWAP, 0.5f * cr, SeqCategory::CrossRow);
        }
        if (multi_row) {
            add(seq::SEQ_ROW_SWAP,    0.3f, SeqCategory::RowLevel);
            add(seq::SEQ_ROW_REVERSE, 0.2f, SeqCategory::RowLevel);
            if (prof.structure == StructClass::MultiPartition) {
                add(seq::SEQ_ROW_SPLIT,  0.2f, SeqCategory::RowLevel);
                add(seq::SEQ_ROW_MERGE,  0.2f, SeqCategory::RowLevel);
            }
        }
    }

    // Lazy normalization: only sum weights; do not normalize here
    reg.weights_sum = 0.0f;
    for (int i = 0; i < reg.count; i++) {
        reg.weights_sum += reg.weights[i];
    }
    return reg;
}

// ============================================================
// ObjConfig — compact objective comparison config for GPU
// ============================================================

struct ObjConfig {
    int         num_obj;
    CompareMode mode;
    ObjDir      dirs[MAX_OBJ];       // direction per objective
    float       weights[MAX_OBJ];    // weights in Weighted mode
    int         priority[MAX_OBJ];   // comparison order in Lexicographic mode
    float       tolerance[MAX_OBJ];  // tolerance in Lexicographic mode
};

// Build ObjConfig from ProblemConfig (CPU side)
inline ObjConfig make_obj_config(const ProblemConfig& pcfg) {
    ObjConfig oc;
    oc.num_obj = pcfg.num_objectives;
    oc.mode = pcfg.compare_mode;
    for (int i = 0; i < MAX_OBJ; i++) {
        oc.dirs[i]      = pcfg.obj_dirs[i];
        oc.weights[i]   = pcfg.obj_weights[i];
        oc.priority[i]  = pcfg.obj_priority[i];
        oc.tolerance[i] = pcfg.obj_tolerance[i];
    }
    return oc;
}

// ============================================================
// SolveResult — return value of solve()
// ============================================================

enum class StopReason { MaxGen, TimeLimit, Stagnation };

template<typename Sol>
struct SolveResult {
    Sol         best_solution;
    float       elapsed_ms     = 0.0f;
    int         generations    = 0;
    StopReason  stop_reason    = StopReason::MaxGen;
};

// ============================================================
// Objective importance mapping — unified importance for Weighted / Lexicographic
// ============================================================
// Used for initial selection (NSGA-II weighted crowding + core-object slots)
// Weighted:      importance[i] = weight[i] / Σweight
// Lexicographic: importance[i] = 0.5^rank[i] / Σ(0.5^rank)
//   → first priority ~57%, second ~29%, third ~14%

inline void compute_importance(const ObjConfig& oc, float* importance) {
    float sum = 0.0f;
    for (int i = 0; i < oc.num_obj; i++) {
        if (oc.mode == CompareMode::Weighted) {
            importance[i] = oc.weights[i];
        } else {
            int rank = oc.priority[i];
            importance[i] = 1.0f;
            for (int r = 0; r < rank; r++) importance[i] *= 0.5f;  // 0.5^rank
        }
        sum += importance[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < oc.num_obj; i++)
            importance[i] /= sum;
    }
}

// ============================================================
// Comparison utilities — Weighted / Lexicographic
// ============================================================

// Normalize objectives to "smaller is better": negate Maximize objectives
__device__ __host__ inline float normalize_obj(float val, ObjDir dir) {
    return (dir == ObjDir::Maximize) ? -val : val;
}

// Core comparison: whether a is better than b
// v5.0: add __host__ so multi-GPU can compare solutions on CPU
template<typename Sol>
__device__ __host__ inline bool is_better(const Sol& a, const Sol& b,
                                  const ObjConfig& oc) {
    // Penalty first: feasible beats infeasible
    if (a.penalty <= 0.0f && b.penalty > 0.0f) return true;
    if (a.penalty > 0.0f && b.penalty <= 0.0f) return false;
    if (a.penalty > 0.0f && b.penalty > 0.0f) return a.penalty < b.penalty;
    
    if (oc.mode == CompareMode::Weighted) {
        // Weighted sum (weights may encode direction: negative for Maximize, or use normalize_obj)
        float sum_a = 0.0f, sum_b = 0.0f;
        for (int i = 0; i < oc.num_obj; i++) {
            float na = normalize_obj(a.objectives[i], oc.dirs[i]);
            float nb = normalize_obj(b.objectives[i], oc.dirs[i]);
            sum_a += oc.weights[i] * na;
            sum_b += oc.weights[i] * nb;
        }
        return sum_a < sum_b;
    } else {
        // Lexicographic: compare objectives in priority order
        for (int p = 0; p < oc.num_obj; p++) {
            int idx = oc.priority[p];
            if (idx < 0 || idx >= oc.num_obj) continue;
            float va = normalize_obj(a.objectives[idx], oc.dirs[idx]);
            float vb = normalize_obj(b.objectives[idx], oc.dirs[idx]);
            float diff = va - vb;
            if (diff < -oc.tolerance[idx]) return true;   // a clearly better
            if (diff >  oc.tolerance[idx]) return false;  // b clearly better
            // Within tolerance → tie, continue to next objective
        }
        return false;  // all objectives tied within tolerance
    }
}

// Scalarization (for SA acceptance): smaller is better
template<typename Sol>
__device__ __host__ inline float scalar_objective(const Sol& sol,
                                                    const ObjConfig& oc) {
    if (oc.mode == CompareMode::Weighted) {
        float sum = 0.0f;
        for (int i = 0; i < oc.num_obj; i++)
            sum += oc.weights[i] * normalize_obj(sol.objectives[i], oc.dirs[i]);
        return sum;
    } else {
        // Under lexicographic SA, use first-priority objective as scalar
        int idx = oc.priority[0];
        if (idx < 0 || idx >= oc.num_obj) idx = 0;
        return normalize_obj(sol.objectives[idx], oc.dirs[idx]);
    }
}

// Lightweight comparison: operate on float[] objectives (avoid copying full Sol)
__device__ inline bool obj_is_better(const float* new_objs, const float* old_objs,
                                      const ObjConfig& oc) {
    if (oc.mode == CompareMode::Weighted) {
        float sum_new = 0.0f, sum_old = 0.0f;
        for (int i = 0; i < oc.num_obj; i++) {
            sum_new += oc.weights[i] * normalize_obj(new_objs[i], oc.dirs[i]);
            sum_old += oc.weights[i] * normalize_obj(old_objs[i], oc.dirs[i]);
        }
        return sum_new < sum_old;
    } else {
        for (int p = 0; p < oc.num_obj; p++) {
            int idx = oc.priority[p];
            if (idx < 0 || idx >= oc.num_obj) continue;
            float va = normalize_obj(new_objs[idx], oc.dirs[idx]);
            float vb = normalize_obj(old_objs[idx], oc.dirs[idx]);
            float diff = va - vb;
            if (diff < -oc.tolerance[idx]) return true;
            if (diff >  oc.tolerance[idx]) return false;
        }
        return false;
    }
}

// Lightweight scalarization: operate on float[] objectives
__device__ __host__ inline float obj_scalar(const float* objs, const ObjConfig& oc) {
    if (oc.mode == CompareMode::Weighted) {
        float sum = 0.0f;
        for (int i = 0; i < oc.num_obj; i++)
            sum += oc.weights[i] * normalize_obj(objs[i], oc.dirs[i]);
        return sum;
    } else {
        int idx = oc.priority[0];
        if (idx < 0 || idx >= oc.num_obj) idx = 0;
        return normalize_obj(objs[idx], oc.dirs[idx]);
    }
}

// ============================================================
// AOSStats — adaptive operator selection stats (one per block)
// ============================================================
// v3.0: granularity from 3 layers → MAX_SEQ sequences
// Records per-sequence usage and improvement counts
// Host aggregates after each batch and updates SeqRegistry weights

struct AOSStats {
    // Operator-level stats (second layer)
    int usage[MAX_SEQ];       // per-sequence usage counts
    int improvement[MAX_SEQ]; // per-sequence improvements (delta < 0 and accepted)
    // K-step layer stats (first layer)
    int k_usage[MAX_K];       // usage counts for K=1,2,3
    int k_improvement[MAX_K]; // improvement counts for K=1,2,3
};

// ============================================================
// ObjDef — single-objective definition (compile-time constant)
// ============================================================

struct ObjDef {
    ObjDir dir;           // optimization direction
    float  weight;        // weight in Weighted mode
    float  tolerance;     // tolerance in Lexicographic mode
};

// ============================================================
// HeuristicMatrix — data matrix descriptor for heuristic initial solutions
// ============================================================

struct HeuristicMatrix {
    const float* data;   // N×N matrix on host
    int N;               // dimension
};

// ============================================================
// ProblemBase<Derived, D1, D2> — CRTP base class
//
// Users inherit this base and provide:
//   static constexpr ObjDef OBJ_DEFS[] = {...};   — objective metadata
//   __device__ float compute_obj(int idx, ...) const;  — objective dispatch
//   __device__ float compute_penalty(...) const;
//
// Convention: OBJ_DEFS and compute_obj stay aligned; case N maps to OBJ_DEFS[N]
// NUM_OBJ is derived from sizeof(OBJ_DEFS); no manual count
//
// Base class provides:
//   evaluate(sol)           — loop objectives and call compute_obj
//   fill_obj_config(cfg)    — fill ProblemConfig from OBJ_DEFS
//   obj_config()            — build ObjConfig directly
// ============================================================

template<typename Derived, int D1_, int D2_>
struct ProblemBase {
    static constexpr int D1 = D1_;
    static constexpr int D2 = D2_;
    using Sol = Solution<D1, D2>;
    
    // NUM_OBJ derived from OBJ_DEFS array size
    static constexpr int NUM_OBJ = sizeof(Derived::OBJ_DEFS) / sizeof(ObjDef);
    
    // Automatic evaluation: iterate objectives
    __device__ void evaluate(Sol& sol) const {
        const auto& self = static_cast<const Derived&>(*this);
        constexpr int n = sizeof(Derived::OBJ_DEFS) / sizeof(ObjDef);
        for (int i = 0; i < n; i++)
            sol.objectives[i] = self.compute_obj(i, sol);
        sol.penalty = self.compute_penalty(sol);
    }
    
    // Fill objective fields of ProblemConfig from OBJ_DEFS
    void fill_obj_config(ProblemConfig& cfg) const {
        constexpr int n = sizeof(Derived::OBJ_DEFS) / sizeof(ObjDef);
        cfg.num_objectives = n;
        for (int i = 0; i < n; i++) {
            cfg.obj_dirs[i]      = Derived::OBJ_DEFS[i].dir;
            cfg.obj_weights[i]   = Derived::OBJ_DEFS[i].weight;
            cfg.obj_tolerance[i] = Derived::OBJ_DEFS[i].tolerance;
            cfg.obj_priority[i]  = i;  // list order is priority order
        }
    }
    
    // Build ObjConfig directly (for solver)
    ObjConfig obj_config() const {
        ProblemConfig pcfg;
        fill_obj_config(pcfg);
        return make_obj_config(pcfg);
    }
    
    // Optional: shared memory requirement (bytes)
    // Default 0 (no shared memory)
    // Override if problem data fits in shared memory; return actual size
    size_t shared_mem_bytes() const {
        return 0;
    }
    
    // Optional: load problem data into shared memory
    // Default no-op (no shared memory)
    // Override if shared_mem_bytes() > 0 to implement loading
    __device__ void load_shared(char* smem, int tid, int bsz) {
        (void)smem; (void)tid; (void)bsz;  // default: no-op
    }
    
    // Hot working-set size in global memory per block (bytes)
    // Used for auto pop_size L2 cache pressure estimate
    // Default = shared_mem_bytes() (when data is in smem, gmem working set is 0)
    // Override when shared_mem_bytes() is 0 (data does not fit in smem):
    //           return actual data size (e.g. distance matrix n*n*sizeof(float))
    size_t working_set_bytes() const {
        return static_cast<const Derived&>(*this).shared_mem_bytes();
    }
    
    // Optional: initialize G/O relation matrix (prior for GUIDED_REBUILD)
    // G[i*N+j]: grouping tendency of i and j (symmetric, [0,1]; higher → same group)
    // O[i*N+j]: tendency for i before j (asymmetric, [0,1])
    // Default none (zeros); EMA accumulates from good solutions during search
    // Example override: close distance → high G and O
    void init_relation_matrix(float* h_G, float* h_O, int N) const {
        (void)h_G; (void)h_O; (void)N;  // default: no-op (keep zeros)
    }
    
    // Optional: host-side data matrices for heuristic initial solutions
    // Default 0 (none); override to fill out[] and return count
    int heuristic_matrices(HeuristicMatrix* out, int max_count) const {
        (void)out; (void)max_count;
        return 0;
    }
    
    // v5.0: multi-GPU — clone Problem to a given GPU
    // Subclasses implement: cudaSetDevice(gpu_id) + device alloc + copy
    // Returns new Problem* on host; internal device pointers target gpu_id
    virtual Derived* clone_to_device(int gpu_id) const {
        (void)gpu_id;
        fprintf(stderr, "Error: clone_to_device() not implemented for this Problem type\n");
        return nullptr;
    }
};
