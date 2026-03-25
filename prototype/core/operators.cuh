/**
 * operators.cuh - Four-layer search operator hierarchy (device side)
 *
 * v1.0: Full operator hierarchy for 2D universal encoding
 *
 * Hierarchy (all operators only see data[D1][D2] + dim2_sizes, no problem semantics):
 *
 *   Layer 1 - Element: operate on single elements
 *     Within row: swap, reverse(2-opt), insert, flip
 *     Cross-row: cross_relocate (move one element across rows), cross_swap (swap one element per row)
 *
 *   Layer 2 - Segment: operate on contiguous segments
 *     Within row: or_opt (move contiguous k elements to a new position in the row)
 *     Cross-row: seg_relocate (move a segment from one row to another)
 *            seg_swap (swap two segments from two rows each, i.e. 2-opt*)
 *
 *   Layer 3 - Row: operate on whole rows
 *     row_swap (swap full contents and lengths of two rows)
 *     row_reverse (reverse row order)
 *     row_split (split one row into two)
 *     row_merge (merge two rows into one)
 *
 *   Layer 4 - Crossover: combine two solutions
 *     row_crossover (child takes some rows from parent A and B)
 *     uniform_crossover (pick per element from two parents)
 *
 * Move descriptor:
 *   row, row2: row indices (row2=-1 means within-row)
 *   op:        operation code
 *   pos1, pos2: position parameters
 *   seg_len:   segment length (used by layer 2)
 *
 * Design principles:
 *   - All operators are problem-agnostic; they only manipulate a 2D array
 *   - Each operator has a corresponding undo
 *   - Empty-row safe: automatically degrades to no-op
 *   - Encoding type determines the available operator set
 */

#pragma once
#include "types.cuh"
#include "cuda_utils.cuh"

namespace ops {

// ============================================================
// Op code constants — numbered by layer to avoid collisions
// ============================================================

// General
constexpr int OP_NOOP             = -1;

// --- Layer 1: element ---
// Permutation within row
constexpr int PERM_SWAP           = 0;   // swap two positions
constexpr int PERM_REVERSE        = 1;   // reverse interval (2-opt)
constexpr int PERM_INSERT         = 2;   // move one element to a new position
// Permutation cross-row
constexpr int PERM_CROSS_RELOCATE = 3;   // move one element from one row to another
constexpr int PERM_CROSS_SWAP     = 4;   // swap one element per row between two rows
// Binary within row
constexpr int BIN_FLIP            = 0;   // flip one bit
constexpr int BIN_SWAP            = 1;   // swap two bits
// Binary cross-row
constexpr int BIN_CROSS_SWAP      = 2;   // swap one bit per row between two rows

// --- Layer 1 (cont.): permutation within row ---
constexpr int PERM_3OPT           = 5;   // 3-opt: break 3 edges and reconnect

// --- Layer 2: segment ---
constexpr int PERM_OR_OPT         = 10;  // within row: move contiguous k elements
constexpr int PERM_SEG_RELOCATE   = 11;  // cross-row: move segment from one row to another
constexpr int PERM_SEG_SWAP       = 12;  // cross-row: swap two segments from two rows each (2-opt*)
constexpr int PERM_CROSS_EXCHANGE = 15;  // cross-row: swap two segments (preserve internal order each)
constexpr int BIN_SEG_FLIP        = 13;  // within row: flip contiguous k bits
constexpr int BIN_SEG_CROSS_SWAP  = 14;  // cross-row: swap two segments from two rows each
constexpr int BIN_K_FLIP          = 16;  // within row: flip k random bits at once

// --- Layer 3: row ---
constexpr int ROW_SWAP            = 20;  // swap full contents of two rows
constexpr int ROW_REVERSE         = 21;  // reverse row order (row index permutation)
constexpr int ROW_SPLIT           = 22;  // split one row into two
constexpr int ROW_MERGE           = 23;  // merge two rows into one

// --- Special: perturbation (multi-step moves, no undo, escape local optima) ---
constexpr int PERTURBATION        = 40;

// --- Layer 4: crossover ---
constexpr int CROSS_ROW           = 30;  // row crossover: take some rows from each parent
constexpr int CROSS_UNIFORM       = 31;  // uniform crossover: pick per element from two parents

// ============================================================
// Move descriptor — encoding-level change description
// ============================================================

struct Move {
    int row;            // source row (or first row)
    int row2;           // target row (-1 = within-row)
    int op;             // operation code
    int pos1, pos2;     // position parameters
    int seg_len;        // segment length (layer 2; 0 for other layers)
};

}  // namespace ops

namespace ops {

// ============================================================
// Layer 1: element-level primitives
// ============================================================

// --- Permutation within row ---

__device__ inline void perm_swap(int* row, int i, int j) {
    int tmp = row[i]; row[i] = row[j]; row[j] = tmp;
}

__device__ inline void perm_reverse(int* row, int i, int j) {
    while (i < j) { perm_swap(row, i, j); i++; j--; }
}

__device__ inline void perm_insert(int* row, int from, int to, int size) {
    int val = row[from];
    if (from < to) { for (int k = from; k < to; k++) row[k] = row[k+1]; }
    else           { for (int k = from; k > to; k--) row[k] = row[k-1]; }
    row[to] = val;
}

// --- Permutation cross-row ---

/// cross_relocate: take element from src_row[src_pos], insert at dst_row[dst_pos]
__device__ inline void perm_cross_relocate(int* src_row, int& src_size,
                                            int* dst_row, int& dst_size,
                                            int src_pos, int dst_pos) {
    int val = src_row[src_pos];
    for (int k = src_pos; k < src_size - 1; k++)
        src_row[k] = src_row[k + 1];
    src_size--;
    for (int k = dst_size; k > dst_pos; k--)
        dst_row[k] = dst_row[k - 1];
    dst_row[dst_pos] = val;
    dst_size++;
}

/// cross_swap: swap rowA[posA] and rowB[posB]
__device__ inline void cross_swap_elem(int* rowA, int posA, int* rowB, int posB) {
    int tmp = rowA[posA]; rowA[posA] = rowB[posB]; rowB[posB] = tmp;
}

// --- Permutation within row: 3-opt ---
// Break 3 edges and pick a reconnection (8 combinations; pick one random non-identity)
// Args: three breakpoints i < j < k, route splits seg0=[0,i] seg1=[i+1,j] seg2=[j+1,k] seg3=[k+1,end]
// Impl: random reconnection (reverse seg1, reverse seg2, or both)
// pos1=i, pos2=j, seg_len encodes k
__device__ inline void perm_3opt(int* row, int size, int i, int j, int k) {
    // 3-opt has several reconnections; here we use the most common non-identity variants:
    //   type 1: reverse [i+1, j]                    — same as 2-opt(i+1, j)
    //   type 2: reverse [j+1, k]                    — same as 2-opt(j+1, k)
    //   type 3: reverse [i+1, j] + reverse [j+1, k] — true 3-opt move
    //   type 4: swap seg1 and seg2 (no reverse)     — generalization of or-opt
    // We would randomize type 3 or 4 (types 1/2 are covered by 2-opt)
    // Here we fix type 3 (double reverse) as the only new neighborhood 2-opt cannot reach
    // reverse [i+1, j]
    int lo = i + 1, hi = j;
    while (lo < hi) { int t = row[lo]; row[lo] = row[hi]; row[hi] = t; lo++; hi--; }
    // reverse [j+1, k]
    lo = j + 1; hi = k;
    while (lo < hi) { int t = row[lo]; row[lo] = row[hi]; row[hi] = t; lo++; hi--; }
}

// 3-opt undo: repeat the same move to restore (double reverse is self-inverse)
__device__ inline void perm_3opt_undo(int* row, int size, int i, int j, int k) {
    perm_3opt(row, size, i, j, k);  // self-inverse
}

// --- Binary within row ---

__device__ inline void bin_flip(int* row, int i) { row[i] = 1 - row[i]; }

__device__ inline void bin_swap(int* row, int i, int j) {
    int tmp = row[i]; row[i] = row[j]; row[j] = tmp;
}

// ============================================================
// Layer 2: segment-level primitives
// ============================================================

/// or_opt: within row, move contiguous seg_len elements (starting at from) to position to
/// Same as: take [from, from+seg_len), insert before to
/// Constraints: from + seg_len <= size, to not in [from, from+seg_len)
__device__ inline void perm_or_opt(int* row, int size, int from, int to, int seg_len) {
    // Temp buffer (max segment length limited by registers; seg_len usually <= 4)
    int buf[8];  // enough for typical seg_len
    int actual_len = (seg_len > 8) ? 8 : seg_len;
    
    // Save segment
    for (int i = 0; i < actual_len; i++) buf[i] = row[from + i];
    
    // Remove segment (shift left to close gap)
    int new_size = size - actual_len;
    for (int k = from; k < new_size; k++) row[k] = row[k + actual_len];
    
    // Insert position after removal (coords after removal)
    int ins = (to > from) ? to - actual_len : to;
    if (ins < 0) ins = 0;
    if (ins > new_size) ins = new_size;
    
    // Insert segment (shift right to make room)
    for (int k = new_size - 1; k >= ins; k--) row[k + actual_len] = row[k];
    for (int i = 0; i < actual_len; i++) row[ins + i] = buf[i];
}

/// seg_relocate: take contiguous seg_len elements from src_row, insert at dst_pos in dst_row
/// src_size -= seg_len, dst_size += seg_len
__device__ inline void perm_seg_relocate(int* src_row, int& src_size,
                                          int* dst_row, int& dst_size,
                                          int src_pos, int dst_pos, int seg_len) {
    int buf[8];
    int actual_len = (seg_len > 8) ? 8 : seg_len;
    
    // Save segment
    for (int i = 0; i < actual_len; i++) buf[i] = src_row[src_pos + i];
    
    // Source row: remove (shift left)
    for (int k = src_pos; k < src_size - actual_len; k++)
        src_row[k] = src_row[k + actual_len];
    src_size -= actual_len;
    
    // Destination row: insert (shift right)
    for (int k = dst_size - 1; k >= dst_pos; k--)
        dst_row[k + actual_len] = dst_row[k];
    for (int i = 0; i < actual_len; i++)
        dst_row[dst_pos + i] = buf[i];
    dst_size += actual_len;
}

/// seg_swap: swap one segment from each row (general 2-opt*)
/// rowA[posA..posA+lenA) <-> rowB[posB..posB+lenB)
/// Row lengths: sizeA += (lenB - lenA), sizeB += (lenA - lenB)
__device__ inline void perm_seg_swap(int* rowA, int& sizeA, int posA, int lenA,
                                      int* rowB, int& sizeB, int posB, int lenB) {
    int bufA[8], bufB[8];
    int aLen = (lenA > 8) ? 8 : lenA;
    int bLen = (lenB > 8) ? 8 : lenB;
    
    // Save both segments
    for (int i = 0; i < aLen; i++) bufA[i] = rowA[posA + i];
    for (int i = 0; i < bLen; i++) bufB[i] = rowB[posB + i];
    
    // Remove segA from rowA to make room for segB
    // Remove first
    int newSizeA = sizeA - aLen;
    for (int k = posA; k < newSizeA; k++) rowA[k] = rowA[k + aLen];
    // Then insert segB
    for (int k = newSizeA - 1; k >= posA; k--) rowA[k + bLen] = rowA[k];
    for (int i = 0; i < bLen; i++) rowA[posA + i] = bufB[i];
    sizeA = newSizeA + bLen;
    
    // Remove segB from rowB to make room for segA
    int newSizeB = sizeB - bLen;
    for (int k = posB; k < newSizeB; k++) rowB[k] = rowB[k + bLen];
    for (int k = newSizeB - 1; k >= posB; k--) rowB[k + aLen] = rowB[k];
    for (int i = 0; i < aLen; i++) rowB[posB + i] = bufA[i];
    sizeB = newSizeB + aLen;
}

/// cross_exchange: swap one segment from each row, preserving internal order each
/// Unlike seg_swap: seg_swap is equal-length swap; cross_exchange allows unequal lengths
/// rowA[posA..posA+lenA) <-> rowB[posB..posB+lenB)
/// Row lengths: sizeA += (lenB - lenA), sizeB += (lenA - lenB)
__device__ inline void perm_cross_exchange(int* rowA, int& sizeA, int posA, int lenA,
                                            int* rowB, int& sizeB, int posB, int lenB) {
    int bufA[8], bufB[8];
    int aLen = (lenA > 8) ? 8 : lenA;
    int bLen = (lenB > 8) ? 8 : lenB;
    
    for (int i = 0; i < aLen; i++) bufA[i] = rowA[posA + i];
    for (int i = 0; i < bLen; i++) bufB[i] = rowB[posB + i];
    
    // rowA: remove segA, insert segB
    int newSizeA = sizeA - aLen;
    for (int k = posA; k < newSizeA; k++) rowA[k] = rowA[k + aLen];
    for (int k = newSizeA - 1; k >= posA; k--) rowA[k + bLen] = rowA[k];
    for (int i = 0; i < bLen; i++) rowA[posA + i] = bufB[i];
    sizeA = newSizeA + bLen;
    
    // rowB: remove segB, insert segA
    int newSizeB = sizeB - bLen;
    for (int k = posB; k < newSizeB; k++) rowB[k] = rowB[k + bLen];
    for (int k = newSizeB - 1; k >= posB; k--) rowB[k + aLen] = rowB[k];
    for (int i = 0; i < aLen; i++) rowB[posB + i] = bufA[i];
    sizeB = newSizeB + aLen;
}

/// k-bit flip: flip k random bits at once (Binary encoding)
/// positions array holds indices to flip; k = number of flips
__device__ inline void bin_k_flip(int* row, int size, int k, curandState* rng) {
    for (int i = 0; i < k; i++) {
        int pos = rand_int(rng, size);
        row[pos] = 1 - row[pos];
    }
}

/// seg_flip: flip contiguous seg_len bits within row (Binary encoding)
__device__ inline void bin_seg_flip(int* row, int pos, int seg_len) {
    for (int i = 0; i < seg_len; i++) row[pos + i] = 1 - row[pos + i];
}

/// seg_cross_swap: swap one segment from each row (Binary encoding, equal length)
__device__ inline void bin_seg_cross_swap(int* rowA, int posA,
                                           int* rowB, int posB, int seg_len) {
    for (int i = 0; i < seg_len; i++) {
        int tmp = rowA[posA + i];
        rowA[posA + i] = rowB[posB + i];
        rowB[posB + i] = tmp;
    }
}

// ============================================================
// Integer encoding primitives
// ============================================================

/// int_clamp: clamp value to [lb, ub]
__device__ inline int int_clamp(int v, int lb, int ub) {
    if (v < lb) return lb;
    if (v > ub) return ub;
    return v;
}

/// int_random_reset: reset one random position to uniform random in [lb, ub]
__device__ inline void int_random_reset(int* row, int pos, int lb, int ub,
                                         curandState* rng) {
    row[pos] = lb + (curand(rng) % (ub - lb + 1));
}

/// int_delta: random position, add ±k (clamped to [lb, ub])
__device__ inline void int_delta(int* row, int pos, int lb, int ub,
                                  curandState* rng) {
    int range = ub - lb + 1;
    int max_step = (range < 5) ? range : 5;
    int step = 1 + (curand(rng) % max_step);
    if (curand(rng) & 1) step = -step;
    row[pos] = int_clamp(row[pos] + step, lb, ub);
}

/// int_seg_reset: reset k contiguous positions to uniform random in [lb, ub]
__device__ inline void int_seg_reset(int* row, int pos, int seg_len,
                                      int lb, int ub, curandState* rng) {
    int range = ub - lb + 1;
    for (int i = 0; i < seg_len; i++)
        row[pos + i] = lb + (curand(rng) % range);
}

/// int_k_delta: k random positions, each ±1
__device__ inline void int_k_delta(int* row, int size, int k,
                                    int lb, int ub, curandState* rng) {
    for (int i = 0; i < k; i++) {
        int pos = rand_int(rng, size);
        int step = (curand(rng) & 1) ? 1 : -1;
        row[pos] = int_clamp(row[pos] + step, lb, ub);
    }
}

// ============================================================
// Layer 3: row-level primitives
// ============================================================

/// row_swap: swap full contents and lengths of two rows
template<typename Sol>
__device__ inline void row_swap(Sol& sol, int r1, int r2) {
    // Swap lengths
    int tmp_size = sol.dim2_sizes[r1];
    sol.dim2_sizes[r1] = sol.dim2_sizes[r2];
    sol.dim2_sizes[r2] = tmp_size;
    // Swap data (use the longer of the two row lengths)
    int max_len = (sol.dim2_sizes[r1] > sol.dim2_sizes[r2]) 
                  ? sol.dim2_sizes[r1] : sol.dim2_sizes[r2];
    // After swap, r1 has old r2 length and r2 has old r1 length
    // So swap max(old r1 len, old r2 len) elements
    max_len = (tmp_size > max_len) ? tmp_size : max_len;
    for (int c = 0; c < max_len; c++) {
        int tmp = sol.data[r1][c];
        sol.data[r1][c] = sol.data[r2][c];
        sol.data[r2][c] = tmp;
    }
}

/// row_reverse: reverse row order in [r1, r2]
/// e.g. row_reverse(sol, 1, 4) turns rows 1,2,3,4 into 4,3,2,1
template<typename Sol>
__device__ inline void row_reverse_range(Sol& sol, int r1, int r2) {
    while (r1 < r2) {
        row_swap(sol, r1, r2);
        r1++; r2--;
    }
}

/// row_split: split row at split_pos into two rows
/// row keeps [0, split_pos), empty_row gets [split_pos, size)
/// requires empty_row empty or with enough space
template<typename Sol>
__device__ inline void row_split(Sol& sol, int row, int empty_row, int split_pos) {
    int orig_size = sol.dim2_sizes[row];
    int move_count = orig_size - split_pos;
    // Copy tail to empty_row
    for (int i = 0; i < move_count; i++)
        sol.data[empty_row][i] = sol.data[row][split_pos + i];
    sol.dim2_sizes[empty_row] = move_count;
    sol.dim2_sizes[row] = split_pos;
}

/// row_merge: append full contents of src_row to end of dst_row
/// src_row cleared, dst_row length increased
/// requires dst_size + src_size <= DIM2
template<typename Sol>
__device__ inline void row_merge(Sol& sol, int dst_row, int src_row) {
    int dst_size = sol.dim2_sizes[dst_row];
    int src_size = sol.dim2_sizes[src_row];
    for (int i = 0; i < src_size; i++)
        sol.data[dst_row][dst_size + i] = sol.data[src_row][i];
    sol.dim2_sizes[dst_row] = dst_size + src_size;
    sol.dim2_sizes[src_row] = 0;
}

// ============================================================
// Layer 4: crossover primitives
// ============================================================
//
// Permutation encoding: OX family (unified framework)
//   Core: mark "kept" positions from A; fill gaps in B's global order
//   Three variants differ only in how the keep set is chosen; fill logic is shared
//   Uniqueness: take from B in order elements not in keep set, no duplicates
//   Row lengths unchanged (= A's row lengths), row boundaries unchanged
//
// Binary encoding: uniform_crossover (random pick per element)
//
// ============================================================

// ---- OX core fill logic ----
// keep[r][c] = true means child[r][c] keeps A's value; false = gap to fill
// Gaps filled in order of appearance of elements in B (row-major scan)
// Requires: child copied from A, dim2_sizes set to A's row lengths
//
// total_elements: total elements in partitioned mode; in non-partitioned = single row length
//   Used to bound the scan range in B

template<typename Sol>
__device__ inline void ox_fill_from_b(Sol& child, const Sol& parentB,
                                       const bool* keep_flat,
                                       int dim1, int total_elements) {
    // Count occurrences of each value at kept positions in A (multiset permutations)
    // keep_flat is row-major flat: keep_flat[r * DIM2 + c]
    int keep_count[512];
    for (int i = 0; i < total_elements; i++) keep_count[i] = 0;
    
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            if (keep_flat[r * Sol::DIM2 + c]) {
                int v = child.data[r][c];
                if (v >= 0 && v < total_elements) keep_count[v]++;
            }
    
    // Collect from B in row scan order: take only as many of each value as needed to fill
    // Standard permutation: at most 1 of each value; multiset: up to repeat_count each
    int fill_buf[512];
    int fill_count = 0;
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < parentB.dim2_sizes[r]; c++) {
            int val = parentB.data[r][c];
            if (val >= 0 && val < total_elements && keep_count[val] > 0) {
                keep_count[val]--;  // consume one kept slot
            } else if (val >= 0 && val < total_elements) {
                fill_buf[fill_count++] = val;
            }
        }
    
    // Fill gaps in order (row by row, left to right)
    int fi = 0;
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            if (!keep_flat[r * Sol::DIM2 + c] && fi < fill_count)
                child.data[r][c] = fill_buf[fi++];
}

// ---- Variant 1: OX-interval ----
// Per row, random contiguous interval kept; preserves adjacency
template<typename Sol>
__device__ inline void ox_interval(Sol& child, const Sol& parentA, const Sol& parentB,
                                    int dim1, int total_elements, curandState* rng) {
    bool keep[Sol::DIM1 * Sol::DIM2];
    for (int i = 0; i < Sol::DIM1 * Sol::DIM2; i++) keep[i] = false;
    
    // child = A, mark each row's kept interval
    for (int r = 0; r < dim1; r++) {
        int sz = parentA.dim2_sizes[r];
        child.dim2_sizes[r] = sz;
        for (int c = 0; c < sz; c++) child.data[r][c] = parentA.data[r][c];
        
        if (sz < 2) {
            // length 0 or 1: keep all
            for (int c = 0; c < sz; c++) keep[r * Sol::DIM2 + c] = true;
            continue;
        }
        // Random interval [lo, hi]
        int lo = rand_int(rng, sz);
        int hi = rand_int(rng, sz);
        if (lo > hi) { int tmp = lo; lo = hi; hi = tmp; }
        for (int c = lo; c <= hi; c++) keep[r * Sol::DIM2 + c] = true;
    }
    
    ox_fill_from_b(child, parentB, keep, dim1, total_elements);
}

// ---- Variant 2: OX-subset ----
// Randomly keep ~50% of positions at their A values; most general
template<typename Sol>
__device__ inline void ox_subset(Sol& child, const Sol& parentA, const Sol& parentB,
                                  int dim1, int total_elements, curandState* rng) {
    bool keep[Sol::DIM1 * Sol::DIM2];
    for (int i = 0; i < Sol::DIM1 * Sol::DIM2; i++) keep[i] = false;
    
    // child = A
    for (int r = 0; r < dim1; r++) {
        child.dim2_sizes[r] = parentA.dim2_sizes[r];
        for (int c = 0; c < parentA.dim2_sizes[r]; c++)
            child.data[r][c] = parentA.data[r][c];
    }
    
    // 50% keep per position
    for (int r = 0; r < dim1; r++)
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            keep[r * Sol::DIM2 + c] = (curand_uniform(rng) < 0.5f);
    
    ox_fill_from_b(child, parentB, keep, dim1, total_elements);
}

// ---- Variant 3: OX-row ----
// Randomly keep whole rows; refill non-kept rows from B's order
// Preserves full route structure; good for VRP
template<typename Sol>
__device__ inline void ox_row(Sol& child, const Sol& parentA, const Sol& parentB,
                               int dim1, int total_elements, curandState* rng) {
    bool keep[Sol::DIM1 * Sol::DIM2];
    for (int i = 0; i < Sol::DIM1 * Sol::DIM2; i++) keep[i] = false;
    
    // child = A
    for (int r = 0; r < dim1; r++) {
        child.dim2_sizes[r] = parentA.dim2_sizes[r];
        for (int c = 0; c < parentA.dim2_sizes[r]; c++)
            child.data[r][c] = parentA.data[r][c];
    }
    
    // 50% chance to keep whole row
    int kept = 0;
    for (int r = 0; r < dim1; r++) {
        if (curand_uniform(rng) < 0.5f) {
            for (int c = 0; c < child.dim2_sizes[r]; c++)
                keep[r * Sol::DIM2 + c] = true;
            kept++;
        }
    }
    // Ensure not all-kept or all-unkept
    if (kept == 0) {
        int r = rand_int(rng, dim1);
        // No keep marks → full refill (at least one row not kept)
        // kept==0 means full refill; valid (child gets B's order into A's structure)
    }
    if (kept == dim1 && dim1 > 1) {
        // All kept → randomly un-keep one row
        int r = rand_int(rng, dim1);
        for (int c = 0; c < child.dim2_sizes[r]; c++)
            keep[r * Sol::DIM2 + c] = false;
    }
    
    ox_fill_from_b(child, parentB, keep, dim1, total_elements);
}

// ---- OX unified entry ----
// Pick one variant at random
// When dim1==1 use only interval and subset (row variant useless)
template<typename Sol>
__device__ inline void perm_ox_crossover(Sol& child, const Sol& parentA, const Sol& parentB,
                                          int dim1, int total_elements, curandState* rng) {
    int n_variants = (dim1 > 1) ? 3 : 2;
    int variant = rand_int(rng, n_variants);  // 0: interval, 1: subset, [2: row]
    switch (variant) {
        case 0: ox_interval(child, parentA, parentB, dim1, total_elements, rng); break;
        case 1: ox_subset(child, parentA, parentB, dim1, total_elements, rng); break;
        case 2: ox_row(child, parentA, parentB, dim1, total_elements, rng); break;
    }
}

/// uniform_crossover: random parent choice per element
/// Suitable for Binary encoding (does not break permutation constraints)
template<typename Sol>
__device__ inline void uniform_crossover(Sol& child, const Sol& parentA, const Sol& parentB,
                                          int dim1, curandState* rng) {
    for (int r = 0; r < dim1; r++) {
        int sizeA = parentA.dim2_sizes[r];
        int sizeB = parentB.dim2_sizes[r];
        int size = (sizeA < sizeB) ? sizeA : sizeB;
        child.dim2_sizes[r] = size;
        for (int c = 0; c < size; c++) {
            child.data[r][c] = (curand_uniform(rng) < 0.5f)
                               ? parentA.data[r][c] : parentB.data[r][c];
        }
    }
}

// [removed] generate_move_for_seq / sample_and_generate / apply_move / undo_move
// After P0 refactor the main path uses execute_sequence; old Move gen/apply/undo path removed

// ============================================================
// execute_sequence — unified API: generate params and execute directly (no Move returned)
// ============================================================
// Returns true if sol modified, false if NOOP
// d_G, d_O, rel_N: optional relation matrices (for SEQ_LNS_GUIDED_REBUILD)
// val_lb, val_ub: Integer encoding value range (ignored for other encodings)

template<typename Sol>
__device__ inline bool execute_sequence(int seq_id, Sol& sol, int dim1,
                                         EncodingType encoding, curandState* rng,
                                         const float* d_G = nullptr,
                                         const float* d_O = nullptr,
                                         int rel_N = 0,
                                         int val_lb = 0,
                                         int val_ub = 1,
                                         const void* prob_data = nullptr) {
    // ============================================================
    // Permutation sequences
    // ============================================================
    if (encoding == EncodingType::Permutation) {
        switch (seq_id) {
        case seq::SEQ_PERM_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            perm_swap(sol.data[row], pos1, pos2);
            return true;
        }
        case seq::SEQ_PERM_REVERSE: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            if (pos1 > pos2) { int t = pos1; pos1 = pos2; pos2 = t; }
            perm_reverse(sol.data[row], pos1, pos2);
            return true;
        }
        case seq::SEQ_PERM_INSERT: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            perm_insert(sol.data[row], pos1, pos2, sz);
            return true;
        }
        case seq::SEQ_PERM_3OPT: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 5) return false;
            int a = rand_int(rng, sz);
            int b = rand_int(rng, sz - 1); if (b >= a) b++;
            int c = rand_int(rng, sz - 2); if (c >= a) c++; if (c >= b) c++;
            if (a > b) { int t = a; a = b; b = t; }
            if (b > c) { int t = b; b = c; c = t; }
            if (a > b) { int t = a; a = b; b = t; }
            perm_3opt(sol.data[row], sz, a, b, c);
            return true;
        }
        case seq::SEQ_PERM_OR_OPT: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 3) return false;
            int max_seg = (sz < 4) ? sz - 1 : 4;
            int seg_len = 1 + rand_int(rng, max_seg);
            int pos1 = rand_int(rng, sz - seg_len + 1);
            int avail = sz - seg_len;
            if (avail < 1) return false;
            int pos2 = rand_int(rng, avail);
            if (pos2 >= pos1) pos2 += seg_len;
            perm_or_opt(sol.data[row], sz, pos1, pos2, seg_len);
            return true;
        }
        case seq::SEQ_PERM_CROSS_RELOCATE: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 && dst_sz > 0) {
                int t = row; row = row2; row2 = t;
                src_sz = sol.dim2_sizes[row]; dst_sz = sol.dim2_sizes[row2];
            }
            if (src_sz == 0 || dst_sz >= Sol::DIM2) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz + 1);
            perm_cross_relocate(sol.data[row], sol.dim2_sizes[row],
                               sol.data[row2], sol.dim2_sizes[row2],
                               pos1, pos2);
            return true;
        }
        case seq::SEQ_PERM_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 || dst_sz == 0) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz);
            cross_swap_elem(sol.data[row], pos1, sol.data[row2], pos2);
            return true;
        }
        case seq::SEQ_PERM_SEG_RELOCATE: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 && dst_sz > 0) {
                int t = row; row = row2; row2 = t;
                src_sz = sol.dim2_sizes[row]; dst_sz = sol.dim2_sizes[row2];
            }
            if (src_sz < 2) return false;
            int max_seg = (src_sz < 4) ? src_sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > src_sz) seg_len = src_sz;
            if (dst_sz + seg_len > Sol::DIM2) return false;
            int pos1 = rand_int(rng, src_sz - seg_len + 1);
            int pos2 = rand_int(rng, dst_sz + 1);
            perm_seg_relocate(sol.data[row], sol.dim2_sizes[row],
                             sol.data[row2], sol.dim2_sizes[row2],
                             pos1, pos2, seg_len);
            return true;
        }
        case seq::SEQ_PERM_SEG_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz < 2 || dst_sz < 2) return false;
            int max_seg = (src_sz < 4) ? src_sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > src_sz) seg_len = src_sz;
            if (dst_sz < seg_len) return false;
            int pos1 = rand_int(rng, src_sz - seg_len + 1);
            int avail = dst_sz - seg_len + 1;
            if (avail <= 0) return false;
            int pos2 = rand_int(rng, avail);
            perm_seg_swap(sol.data[row], sol.dim2_sizes[row], pos1, seg_len,
                         sol.data[row2], sol.dim2_sizes[row2], pos2, seg_len);
            return true;
        }
        case seq::SEQ_PERM_CROSS_EXCHANGE: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz < 2 || dst_sz < 2) return false;
            int max_segA = (src_sz < 4) ? src_sz : 4;
            int lenA = 2 + rand_int(rng, max_segA - 1);
            if (lenA > src_sz) lenA = src_sz;
            int dst_max_seg = (dst_sz < 4) ? dst_sz : 4;
            int lenB = 1 + rand_int(rng, dst_max_seg);
            if (src_sz - lenA + lenB > Sol::DIM2 || dst_sz - lenB + lenA > Sol::DIM2)
                return false;
            int pos1 = rand_int(rng, src_sz - lenA + 1);
            int pos2 = rand_int(rng, dst_sz - lenB + 1);
            perm_cross_exchange(sol.data[row], sol.dim2_sizes[row], pos1, lenA,
                               sol.data[row2], sol.dim2_sizes[row2], pos2, lenB);
            return true;
        }
        case seq::SEQ_PERM_DOUBLE_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 3) return false;
            int a1 = rand_int(rng, sz), a2 = rand_int(rng, sz - 1); if (a2 >= a1) a2++;
            perm_swap(sol.data[row], a1, a2);
            int b1 = rand_int(rng, sz), b2 = rand_int(rng, sz - 1); if (b2 >= b1) b2++;
            perm_swap(sol.data[row], b1, b2);
            return true;
        }
        case seq::SEQ_PERM_TRIPLE_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            for (int s = 0; s < 3; s++) {
                int p1 = rand_int(rng, sz), p2 = rand_int(rng, sz - 1); if (p2 >= p1) p2++;
                perm_swap(sol.data[row], p1, p2);
            }
            return true;
        }
        case seq::SEQ_LNS_SEGMENT_SHUFFLE: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            int count = sz / 5;
            if (count < 2) count = 2;
            if (count > sz) count = sz;
            int start = rand_int(rng, sz);
            for (int i = count - 1; i > 0; i--) {
                int j = rand_int(rng, i + 1);
                int pi = (start + i) % sz;
                int pj = (start + j) % sz;
                int tmp = sol.data[row][pi];
                sol.data[row][pi] = sol.data[row][pj];
                sol.data[row][pj] = tmp;
            }
            return true;
        }
        case seq::SEQ_LNS_SCATTER_SHUFFLE: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            int count = sz / 5;
            if (count < 2) count = 2;
            if (count > 16) count = 16;
            int positions[16];
            for (int i = 0; i < count; i++) {
                positions[i] = rand_int(rng, sz);
                for (int j = 0; j < i; j++) {
                    if (positions[i] == positions[j]) {
                        positions[i] = rand_int(rng, sz);
                        j = -1;
                    }
                }
            }
            for (int i = count - 1; i > 0; i--) {
                int j = rand_int(rng, i + 1);
                int tmp = sol.data[row][positions[i]];
                sol.data[row][positions[i]] = sol.data[row][positions[j]];
                sol.data[row][positions[j]] = tmp;
            }
            return true;
        }
        case seq::SEQ_LNS_GUIDED_REBUILD: {
            // Relation-matrix guided rebuild:
            //   1. Pick random seed element seed
            //   2. Look up G[seed] for K elements with strongest grouping affinity
            //   3. Find positions of these elements in the solution
            //   4. Reorder these positions by order guided by O matrix
            //
            // Without relation matrices (cold start), fall back to scatter_shuffle
            if (!d_G || !d_O || rel_N <= 0) {
                // Fallback: random scatter shuffle
                int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
                int sz = sol.dim2_sizes[row];
                if (sz < 4) return false;
                int count = sz / 5;
                if (count < 2) count = 2;
                if (count > 12) count = 12;
                int positions[12];
                for (int i = 0; i < count; i++) {
                    positions[i] = rand_int(rng, sz);
                    for (int j = 0; j < i; j++) {
                        if (positions[i] == positions[j]) { positions[i] = rand_int(rng, sz); j = -1; }
                    }
                }
                for (int i = count - 1; i > 0; i--) {
                    int j = rand_int(rng, i + 1);
                    int tmp = sol.data[row][positions[i]];
                    sol.data[row][positions[i]] = sol.data[row][positions[j]];
                    sol.data[row][positions[j]] = tmp;
                }
                return true;
            }
            
            // --- With relation matrices: guided rebuild ---
            // Generic strategy (problem-agnostic):
            //   G matrix → which elements (weak grouping with seed = likely misplaced)
            //   O matrix → how to order (ordering affinity guides reorder)
            //   Together: G picks, O orders
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 4) return false;
            
            // Pick seed element
            int seed_pos = rand_int(rng, sz);
            int seed_val = sol.data[row][seed_pos];
            if (seed_val < 0 || seed_val >= rel_N) return false;
            
            // Check matrices have enough signal (either G or O)
            float max_signal = 0.0f;
            for (int c = 0; c < sz; c++) {
                int v = sol.data[row][c];
                if (v >= 0 && v < rel_N && v != seed_val) {
                    float g = d_G[seed_val * rel_N + v];
                    float o = d_O[seed_val * rel_N + v];
                    if (g > max_signal) max_signal = g;
                    if (o > max_signal) max_signal = o;
                }
            }
            if (max_signal < 0.05f) return false;  // insufficient signal, skip
            
            // Destroy: tournament pick low-G elements (t=2)
            // Low G = weak grouping with seed = likely misplaced
            // Tournament: draw 2 at random, take lower G, repeat count times
            constexpr int MAX_REBUILD = 10;
            constexpr int TOUR_SIZE = 2;
            int count = sz / 5;  // ~20%
            if (count < 3) count = 3;
            if (count > MAX_REBUILD) count = MAX_REBUILD;
            if (count > sz) count = sz;
            
            int sel_pos[MAX_REBUILD];
            int sel_val[MAX_REBUILD];
            bool used[128] = {};  // mark chosen positions to avoid duplicates
            int picked = 0;
            int max_attempts = count * 4;  // avoid infinite loop
            
            for (int attempt = 0; attempt < max_attempts && picked < count; attempt++) {
                // Tournament: draw TOUR_SIZE candidates at random, take lowest G
                int best_c = -1;
                float best_g = 1e30f;
                for (int t = 0; t < TOUR_SIZE; t++) {
                    int c = rand_int(rng, sz);
                    if (used[c]) continue;
                    int v = sol.data[row][c];
                    if (v < 0 || v >= rel_N) continue;
                    float g = d_G[seed_val * rel_N + v];
                    if (g < best_g) { best_g = g; best_c = c; }
                }
                if (best_c < 0 || used[best_c]) continue;
                used[best_c] = true;
                sel_pos[picked] = best_c;
                sel_val[picked] = sol.data[row][best_c];
                picked++;
            }
            if (picked < 2) return false;
            count = picked;
            
            // Repair: tournament sort (O-guided + random noise)
            // Insertion sort with noisy comparison: high O tends to go first, not guaranteed
            for (int i = 1; i < count; i++) {
                int key = sel_val[i];
                int j = i - 1;
                while (j >= 0) {
                    float o_key_before = d_O[key * rel_N + sel_val[j]];
                    float o_j_before   = d_O[sel_val[j] * rel_N + key];
                    // Noise scale 0.05: if O gap >0.05 mostly deterministic, else random
                    float noise = (curand_uniform(rng) - 0.5f) * 0.1f;
                    if (o_key_before + noise > o_j_before) {
                        sel_val[j + 1] = sel_val[j];
                        j--;
                    } else {
                        break;
                    }
                }
                sel_val[j + 1] = key;
            }
            
            // Sort sel_pos ascending so write-back order is stable
            for (int i = 1; i < count; i++) {
                int key = sel_pos[i];
                int j = i - 1;
                while (j >= 0 && sel_pos[j] > key) {
                    sel_pos[j + 1] = sel_pos[j];
                    j--;
                }
                sel_pos[j + 1] = key;
            }
            
            // Check whether permutation actually changed
            bool any_change = false;
            for (int i = 0; i < count; i++) {
                if (sol.data[row][sel_pos[i]] != sel_val[i]) {
                    any_change = true;
                    break;
                }
            }
            if (!any_change) return false;
            
            // Write back
            for (int i = 0; i < count; i++) {
                sol.data[row][sel_pos[i]] = sel_val[i];
            }
            
            return true;
        }
        default: break;
        }
    }

    // ============================================================
    // Binary sequences
    // ============================================================
    if (encoding == EncodingType::Binary) {
        switch (seq_id) {
        case seq::SEQ_BIN_FLIP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 1) return false;
            int pos1 = rand_int(rng, sz);
            bin_flip(sol.data[row], pos1);
            return true;
        }
        case seq::SEQ_BIN_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            bin_swap(sol.data[row], pos1, pos2);
            return true;
        }
        case seq::SEQ_BIN_SEG_FLIP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_seg = (sz < 4) ? sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > sz) seg_len = sz;
            int pos1 = rand_int(rng, sz - seg_len + 1);
            bin_seg_flip(sol.data[row], pos1, seg_len);
            return true;
        }
        case seq::SEQ_BIN_K_FLIP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_k = (sz < 5) ? sz : 5;
            int k = 2 + rand_int(rng, max_k - 1);
            bin_k_flip(sol.data[row], sz, k, rng);
            return true;
        }
        case seq::SEQ_BIN_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 || dst_sz == 0) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz);
            cross_swap_elem(sol.data[row], pos1, sol.data[row2], pos2);
            return true;
        }
        case seq::SEQ_BIN_SEG_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz < 2 || dst_sz < 2) return false;
            int max_seg = (src_sz < 4) ? src_sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > src_sz) seg_len = src_sz;
            if (dst_sz < seg_len) return false;
            int pos1 = rand_int(rng, src_sz - seg_len + 1);
            int pos2 = rand_int(rng, dst_sz - seg_len + 1);
            bin_seg_cross_swap(sol.data[row], pos1, sol.data[row2], pos2, seg_len);
            return true;
        }
        default: break;
        }
    }

    // ============================================================
    // Integer sequences
    // ============================================================
    if (encoding == EncodingType::Integer) {
        switch (seq_id) {
        case seq::SEQ_INT_RANDOM_RESET: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 1) return false;
            int pos = rand_int(rng, sz);
            int_random_reset(sol.data[row], pos, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_DELTA: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 1) return false;
            int pos = rand_int(rng, sz);
            int_delta(sol.data[row], pos, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_SWAP: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int pos1 = rand_int(rng, sz);
            int pos2 = rand_int(rng, sz - 1); if (pos2 >= pos1) pos2++;
            int tmp = sol.data[row][pos1];
            sol.data[row][pos1] = sol.data[row][pos2];
            sol.data[row][pos2] = tmp;
            return true;
        }
        case seq::SEQ_INT_SEG_RESET: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_seg = (sz < 4) ? sz : 4;
            int seg_len = 2 + rand_int(rng, max_seg - 1);
            if (seg_len > sz) seg_len = sz;
            int pos = rand_int(rng, sz - seg_len + 1);
            int_seg_reset(sol.data[row], pos, seg_len, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_K_DELTA: {
            int row = (dim1 > 1) ? rand_int(rng, dim1) : 0;
            int sz = sol.dim2_sizes[row];
            if (sz < 2) return false;
            int max_k = (sz < 5) ? sz : 5;
            int k = 2 + rand_int(rng, max_k - 1);
            int_k_delta(sol.data[row], sz, k, val_lb, val_ub, rng);
            return true;
        }
        case seq::SEQ_INT_CROSS_SWAP: {
            if (dim1 < 2) return false;
            int row = rand_int(rng, dim1);
            int row2 = rand_int(rng, dim1 - 1); if (row2 >= row) row2++;
            int src_sz = sol.dim2_sizes[row];
            int dst_sz = sol.dim2_sizes[row2];
            if (src_sz == 0 || dst_sz == 0) return false;
            int pos1 = rand_int(rng, src_sz);
            int pos2 = rand_int(rng, dst_sz);
            cross_swap_elem(sol.data[row], pos1, sol.data[row2], pos2);
            return true;
        }
        default: break;
        }
    }

    // ============================================================
    // Shared: row-level sequences (encoding-agnostic)
    // ============================================================
    switch (seq_id) {
    case seq::SEQ_ROW_SWAP: {
        if (dim1 < 2) return false;
        int r1 = rand_int(rng, dim1);
        int r2 = rand_int(rng, dim1 - 1); if (r2 >= r1) r2++;
        row_swap(sol, r1, r2);
        return true;
    }
    case seq::SEQ_ROW_REVERSE: {
        if (dim1 < 2) return false;
        int r1 = rand_int(rng, dim1);
        int r2 = rand_int(rng, dim1 - 1); if (r2 >= r1) r2++;
        if (r1 > r2) { int t = r1; r1 = r2; r2 = t; }
        row_reverse_range(sol, r1, r2);
        return true;
    }
    case seq::SEQ_ROW_SPLIT: {
        if (dim1 < 2) return false;
        int attempts = 0;
        int row;
        do { row = rand_int(rng, dim1); attempts++; }
        while (sol.dim2_sizes[row] < 2 && attempts < dim1 * 2);
        if (sol.dim2_sizes[row] < 2) return false;
        int empty_row = -1;
        for (int r = 0; r < dim1; r++) {
            if (r != row && sol.dim2_sizes[r] == 0) { empty_row = r; break; }
        }
        if (empty_row < 0) return false;
        int split_pos = 1 + rand_int(rng, sol.dim2_sizes[row] - 1);
        row_split(sol, row, empty_row, split_pos);
        return true;
    }
    case seq::SEQ_ROW_MERGE: {
        if (dim1 < 2) return false;
        int non_empty[64]; int cnt = 0;
        for (int r = 0; r < dim1 && cnt < 64; r++)
            if (sol.dim2_sizes[r] > 0) non_empty[cnt++] = r;
        if (cnt < 2) return false;
        int i1 = rand_int(rng, cnt);
        int i2 = rand_int(rng, cnt - 1); if (i2 >= i1) i2++;
        int dst_row = non_empty[i1];
        int src_row = non_empty[i2];
        if (sol.dim2_sizes[dst_row] + sol.dim2_sizes[src_row] > Sol::DIM2)
            return false;
        row_merge(sol, dst_row, src_row);
        return true;
    }
    default:
        break;
    }

    // Custom operator hook: if seq_id >= 100, delegate to user-defined function
    // (defined via JIT template or left as default no-op)
#ifdef CUGENOPT_HAS_CUSTOM_OPS
    return execute_custom_op(seq_id, sol, dim1, encoding, rng, val_lb, val_ub, prob_data);
#else
    return false;
#endif
}

// ============================================================
// sample_and_execute — sample from SeqRegistry by weight and execute directly
// ============================================================
// Returns true if sol modified, false if NOOP
// out_seq_idx: index of sampled sequence in registry
// d_G, d_O, rel_N: optional relation matrices (passed to execute_sequence)

template<typename Sol>
__device__ inline bool sample_and_execute(const SeqRegistry& reg,
                                          Sol& sol, int dim1,
                                          EncodingType encoding,
                                          curandState* rng,
                                          int& out_seq_idx,
                                          const float* d_G = nullptr,
                                          const float* d_O = nullptr,
                                          int rel_N = 0,
                                          int val_lb = 0,
                                          int val_ub = 1,
                                          const void* prob_data = nullptr) {
    // Lazy normalization: use cached weights_sum
    float r = curand_uniform(rng) * reg.weights_sum;  // r ∈ [0, weights_sum)
    float cumsum = 0.0f;
    out_seq_idx = reg.count - 1;
    for (int i = 0; i < reg.count; i++) {
        cumsum += reg.weights[i];
        if (r < cumsum) { out_seq_idx = i; break; }
    }
    int seq_id = reg.ids[out_seq_idx];
    return execute_sequence(seq_id, sol, dim1, encoding, rng, d_G, d_O, rel_N,
                            val_lb, val_ub, prob_data);
}



}  // namespace ops
