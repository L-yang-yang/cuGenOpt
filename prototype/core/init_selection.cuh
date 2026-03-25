/**
 * init_selection.cuh - Initial-solution sampling and NSGA-II selection
 *
 * Host-side logic; called once during solver initialization.
 * Selects pop_size individuals from K × pop_size candidates as the initial population.
 *
 * Selection strategy:
 *   1. Reserve slots for core objectives (by importance)
 *   2. NSGA-II selection (non-dominated sort + weighted crowding)
 *   3. Pure random fallback (diversity)
 *
 * Single-objective case automatically reduces to top-N sorting; no extra branching.
 */

#pragma once
#include "types.cuh"
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>

namespace init_sel {

// ============================================================
// Per-candidate objective info (used on host after download from GPU)
// ============================================================
struct CandidateInfo {
    int   idx;           // Original index in the candidate array
    float objs[MAX_OBJ]; // Normalized objectives (lower is better)
    float penalty;
    int   rank;          // Non-dominated sort front (0 = Pareto front)
    float crowding;      // Crowding distance
    bool  selected;      // Whether already selected
};

// ============================================================
// Non-dominated sort (Fast Non-dominated Sort)
// ============================================================
// Complexity: O(M × N²), M = number of objectives, N = number of candidates
// Acceptable for initialization (N up to a few thousand, M ≤ 4)

inline void fast_nondominated_sort(std::vector<CandidateInfo>& cands,
                                    int num_obj,
                                    std::vector<std::vector<int>>& fronts) {
    int n = (int)cands.size();
    std::vector<int> dom_count(n, 0);        // How many solutions dominate this one
    std::vector<std::vector<int>> dom_set(n); // Which solutions this one dominates
    
    // Whether a dominates b: a ≤ b on all objectives, and strictly < on at least one
    // Handle penalty first: feasible dominates infeasible
    auto dominates = [&](int a, int b) -> bool {
        const auto& ca = cands[a];
        const auto& cb = cands[b];
        // Penalty handling
        if (ca.penalty <= 0.0f && cb.penalty > 0.0f) return true;
        if (ca.penalty > 0.0f && cb.penalty <= 0.0f) return false;
        if (ca.penalty > 0.0f && cb.penalty > 0.0f) return ca.penalty < cb.penalty;
        
        bool all_leq = true;
        bool any_lt = false;
        for (int m = 0; m < num_obj; m++) {
            if (ca.objs[m] > cb.objs[m]) { all_leq = false; break; }
            if (ca.objs[m] < cb.objs[m]) any_lt = true;
        }
        return all_leq && any_lt;
    };
    
    // Compute dominance relations
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dominates(i, j)) {
                dom_set[i].push_back(j);
                dom_count[j]++;
            } else if (dominates(j, i)) {
                dom_set[j].push_back(i);
                dom_count[i]++;
            }
        }
    }
    
    // Extract each front layer
    fronts.clear();
    std::vector<int> current_front;
    for (int i = 0; i < n; i++) {
        if (dom_count[i] == 0) {
            cands[i].rank = 0;
            current_front.push_back(i);
        }
    }
    
    int front_idx = 0;
    while (!current_front.empty()) {
        fronts.push_back(current_front);
        std::vector<int> next_front;
        for (int i : current_front) {
            for (int j : dom_set[i]) {
                dom_count[j]--;
                if (dom_count[j] == 0) {
                    cands[j].rank = front_idx + 1;
                    next_front.push_back(j);
                }
            }
        }
        current_front = next_front;
        front_idx++;
    }
}

// ============================================================
// Weighted crowding distance
// ============================================================
// Standard crowding + importance weighting: larger gap contribution on core objectives

inline void weighted_crowding_distance(std::vector<CandidateInfo>& cands,
                                        const std::vector<int>& front,
                                        int num_obj,
                                        const float* importance) {
    int n = (int)front.size();
    if (n <= 2) {
        for (int i : front) cands[i].crowding = 1e18f;  // Boundary solutions: infinite
        return;
    }
    
    for (int i : front) cands[i].crowding = 0.0f;
    
    std::vector<int> sorted_idx(front.begin(), front.end());
    
    for (int m = 0; m < num_obj; m++) {
        // Sort by objective m
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return cands[a].objs[m] < cands[b].objs[m]; });
        
        float range = cands[sorted_idx[n-1]].objs[m] - cands[sorted_idx[0]].objs[m];
        if (range < 1e-12f) continue;  // No spread on this objective
        
        // Boundary solutions: infinite crowding
        cands[sorted_idx[0]].crowding += 1e18f;
        cands[sorted_idx[n-1]].crowding += 1e18f;
        
        // Interior: neighbor gap × importance weight
        float w = importance[m];
        for (int i = 1; i < n - 1; i++) {
            float gap = cands[sorted_idx[i+1]].objs[m] - cands[sorted_idx[i-1]].objs[m];
            cands[sorted_idx[i]].crowding += w * (gap / range);
        }
    }
}

// ============================================================
// Main selection: pick target candidates from N
// ============================================================
// Returns indices of selected candidates

inline std::vector<int> nsga2_select(std::vector<CandidateInfo>& cands,
                                      int num_obj,
                                      const float* importance,
                                      int target,
                                      int num_reserved_random) {
    // --- 1. Reserve slots for core objectives ---
    int num_reserve_total = target - num_reserved_random;
    // Reserve ratio: importance[i] × 30% of slots (remaining 70% for NSGA-II)
    float reserve_ratio = 0.3f;
    
    std::vector<int> selected;
    selected.reserve(target);
    
    // For each objective, sort by that objective and take top
    for (int m = 0; m < num_obj; m++) {
        int quota = (int)(num_reserve_total * importance[m] * reserve_ratio);
        if (quota < 1 && num_obj > 1) quota = 1;  // At least one per objective
        
        // Sort by objective m (lower is better)
        std::vector<int> by_obj(cands.size());
        for (int i = 0; i < (int)cands.size(); i++) by_obj[i] = i;
        std::sort(by_obj.begin(), by_obj.end(),
                  [&](int a, int b) { return cands[a].objs[m] < cands[b].objs[m]; });
        
        int added = 0;
        for (int i = 0; i < (int)by_obj.size() && added < quota; i++) {
            int idx = by_obj[i];
            if (!cands[idx].selected) {
                cands[idx].selected = true;
                selected.push_back(idx);
                added++;
            }
        }
    }
    
    // --- 2. NSGA-II fills remaining slots ---
    int remaining = target - num_reserved_random - (int)selected.size();
    
    if (remaining > 0) {
        // Non-dominated sort
        std::vector<std::vector<int>> fronts;
        fast_nondominated_sort(cands, num_obj, fronts);
        
        for (auto& front : fronts) {
            if (remaining <= 0) break;
            
            // Filter out already selected
            std::vector<int> available;
            for (int i : front) {
                if (!cands[i].selected) available.push_back(i);
            }
            
            if ((int)available.size() <= remaining) {
                // Take the whole front
                for (int i : available) {
                    cands[i].selected = true;
                    selected.push_back(i);
                    remaining--;
                }
            } else {
                // Truncate this front: pick by weighted crowding
                weighted_crowding_distance(cands, available, num_obj, importance);
                std::sort(available.begin(), available.end(),
                          [&](int a, int b) { return cands[a].crowding > cands[b].crowding; });
                for (int i = 0; i < remaining; i++) {
                    cands[available[i]].selected = true;
                    selected.push_back(available[i]);
                }
                remaining = 0;
            }
        }
    }
    
    return selected;
}

// ============================================================
// Single-objective fast path: scalar sort and take top
// ============================================================
inline std::vector<int> top_n_select(std::vector<CandidateInfo>& cands,
                                      int target,
                                      int num_reserved_random) {
    int to_select = target - num_reserved_random;
    
    // Prefer lower penalty, then objs[0] (normalized, lower is better)
    std::vector<int> indices(cands.size());
    for (int i = 0; i < (int)cands.size(); i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        if (cands[a].penalty <= 0.0f && cands[b].penalty > 0.0f) return true;
        if (cands[a].penalty > 0.0f && cands[b].penalty <= 0.0f) return false;
        if (cands[a].penalty > 0.0f && cands[b].penalty > 0.0f)
            return cands[a].penalty < cands[b].penalty;
        return cands[a].objs[0] < cands[b].objs[0];
    });
    
    std::vector<int> selected;
    selected.reserve(to_select);
    for (int i = 0; i < to_select && i < (int)indices.size(); i++) {
        selected.push_back(indices[i]);
        cands[indices[i]].selected = true;
    }
    return selected;
}

} // namespace init_sel
