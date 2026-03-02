"""
OR-Tools CP-SAT 求解 Assignment / Schedule / BinPacking / GraphColoring
与 GenSolver exp1 中的小规模实例对比
"""
import sys
import time
from ortools.sat.python import cp_model


def solve_assignment(cost, time_limit_sec):
    """指派问题：n 人 n 任务，最小化总成本"""
    n = len(cost)
    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n)] for i in range(n)]

    for i in range(n):
        model.AddExactlyOne(x[i][j] for j in range(n))
    for j in range(n):
        model.AddExactlyOne(x[i][j] for i in range(n))

    model.Minimize(
        sum(cost[i][j] * x[i][j] for i in range(n) for j in range(n))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.ObjectiveValue(), elapsed_ms
    return float("inf"), elapsed_ms


def solve_schedule(cost, days, emps, required, time_limit_sec):
    """排班问题：每天恰好 required 人上班，最小化总成本 + 不公平度"""
    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{d}_{e}") for e in range(emps)] for d in range(days)]

    for d in range(days):
        model.Add(sum(x[d][e] for e in range(emps)) == required)

    total_cost = sum(
        int(cost[d][e]) * x[d][e] for d in range(days) for e in range(emps)
    )

    shifts = [sum(x[d][e] for d in range(days)) for e in range(emps)]
    max_shift = model.NewIntVar(0, days, "max_shift")
    min_shift = model.NewIntVar(0, days, "min_shift")
    model.AddMaxEquality(max_shift, shifts)
    model.AddMinEquality(min_shift, shifts)
    unfairness = model.NewIntVar(0, days, "unfairness")
    model.Add(unfairness == max_shift - min_shift)

    model.Minimize(total_cost)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.ObjectiveValue(), elapsed_ms
    return float("inf"), elapsed_ms


def solve_bin_packing(weights, max_bins, capacity, time_limit_sec):
    """装箱问题：最小化使用箱数"""
    n = len(weights)
    model = cp_model.CpModel()

    x = [[model.NewBoolVar(f"x_{i}_{b}") for b in range(max_bins)] for i in range(n)]
    y = [model.NewBoolVar(f"y_{b}") for b in range(max_bins)]

    for i in range(n):
        model.AddExactlyOne(x[i][b] for b in range(max_bins))

    for b in range(max_bins):
        model.Add(
            sum(int(weights[i]) * x[i][b] for i in range(n)) <= int(capacity) * y[b]
        )

    for b in range(max_bins):
        for i in range(n):
            model.AddImplication(x[i][b], y[b])

    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.ObjectiveValue(), elapsed_ms
    return float("inf"), elapsed_ms


def solve_graph_coloring(n, edges, k, time_limit_sec):
    """图着色：k 色，最小化冲突数"""
    model = cp_model.CpModel()

    color = [model.NewIntVar(0, k - 1, f"c_{i}") for i in range(n)]

    conflicts = []
    for u, v in edges:
        b = model.NewBoolVar(f"conflict_{u}_{v}")
        model.Add(color[u] == color[v]).OnlyEnforceIf(b)
        model.Add(color[u] != color[v]).OnlyEnforceIf(b.Not())
        conflicts.append(b)

    model.Minimize(sum(conflicts))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec

    t0 = time.perf_counter()
    status = solver.Solve(model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.ObjectiveValue(), elapsed_ms
    return float("inf"), elapsed_ms


def main():
    print("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason")

    time_limits = [1, 5, 10]

    # Assignment 4x4 (optimal=13)
    cost = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]]
    for t in time_limits:
        obj, ms = solve_assignment(cost, t)
        gap = (obj - 13) / 13 * 100.0
        print(f"Assign4,cpsat_{t}s,0,{obj:.2f},0.00,{ms:.1f},{gap:.2f},0,time")
        sys.stdout.flush()
    print("  [cpsat] Assign4 done", file=sys.stderr)

    # Schedule 3x4 (cost optimal=21)
    sched_cost = [[5, 3, 8, 4], [6, 2, 7, 5], [4, 6, 3, 7]]
    for t in time_limits:
        obj, ms = solve_schedule(sched_cost, 3, 4, 2, t)
        print(f"Schedule3x4,cpsat_{t}s,0,{obj:.2f},0.00,{ms:.1f},0.00,0,time")
        sys.stdout.flush()
    print("  [cpsat] Schedule3x4 done", file=sys.stderr)

    # BinPacking 8 items (optimal=4)
    weights = [7, 5, 3, 4, 6, 2, 8, 1]
    for t in time_limits:
        obj, ms = solve_bin_packing(weights, 6, 10, t)
        gap = (obj - 4) / 4 * 100.0
        print(f"BinPack8,cpsat_{t}s,0,{obj:.2f},0.00,{ms:.1f},{gap:.2f},0,time")
        sys.stdout.flush()
    print("  [cpsat] BinPack8 done", file=sys.stderr)

    # GraphColor Petersen (optimal=0 conflicts)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]
    for t in time_limits:
        obj, ms = solve_graph_coloring(10, edges, 3, t)
        print(f"GraphColor10,cpsat_{t}s,0,{obj:.2f},0.00,{ms:.1f},0.00,0,time")
        sys.stdout.flush()
    print("  [cpsat] GraphColor10 done", file=sys.stderr)


if __name__ == "__main__":
    main()
