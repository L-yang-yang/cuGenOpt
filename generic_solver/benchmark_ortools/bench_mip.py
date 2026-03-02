"""
MIP (SCIP/CBC) 求解 TSP 和 BinPacking
展示 MIP 建模复杂度 vs GenSolver 的低门槛
TSP 用 MTZ 子回路消除，BinPacking 用标准指派模型
"""
import sys
import time
from ortools.linear_solver import pywraplp
from instances import load_tsp, euc2d_dist_matrix, TSP_INSTANCES


def solve_tsp_mip(dist, n, time_limit_sec, solver_id="SCIP"):
    """TSP MTZ 公式：x_ij + u_i 子回路消除"""
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if not solver:
        return float("inf"), 0.0

    INF = solver.infinity()

    x = [[solver.IntVar(0, 1, f"x_{i}_{j}") for j in range(n)] for i in range(n)]
    u = [solver.IntVar(0, n - 1, f"u_{i}") for i in range(n)]

    for i in range(n):
        solver.Add(x[i][i] == 0)

    for i in range(n):
        solver.Add(sum(x[i][j] for j in range(n)) == 1)
    for j in range(n):
        solver.Add(sum(x[i][j] for i in range(n)) == 1)

    # MTZ 子回路消除
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                solver.Add(u[i] - u[j] + n * x[i][j] <= n - 1)

    solver.Minimize(
        sum(dist[i][j] * x[i][j] for i in range(n) for j in range(n))
    )

    solver.SetTimeLimit(int(time_limit_sec * 1000))

    t0 = time.perf_counter()
    status = solver.Solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return solver.Objective().Value(), elapsed_ms
    return float("inf"), elapsed_ms


def solve_binpacking_mip(weights, max_bins, capacity, time_limit_sec, solver_id="SCIP"):
    """BinPacking 标准 MIP"""
    n = len(weights)
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if not solver:
        return float("inf"), 0.0

    x = [[solver.IntVar(0, 1, f"x_{i}_{b}") for b in range(max_bins)] for i in range(n)]
    y = [solver.IntVar(0, 1, f"y_{b}") for b in range(max_bins)]

    for i in range(n):
        solver.Add(sum(x[i][b] for b in range(max_bins)) == 1)

    for b in range(max_bins):
        solver.Add(sum(weights[i] * x[i][b] for i in range(n)) <= capacity * y[b])

    solver.Minimize(sum(y))
    solver.SetTimeLimit(int(time_limit_sec * 1000))

    t0 = time.perf_counter()
    status = solver.Solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return solver.Objective().Value(), elapsed_ms
    return float("inf"), elapsed_ms


def main():
    print("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason")

    # TSP — 只跑小/中规模（MTZ 在大规模上变量数 O(n^2) 很慢）
    tsp_small = [e for e in TSP_INSTANCES if e["optimal"] <= 21282]  # eil51, eil76, kroA100
    time_limits = [10, 60]

    for entry in tsp_small:
        inst = load_tsp(entry)
        print(f"  [mip] TSP {inst['name']} (n={inst['n']})", file=sys.stderr)
        dist = euc2d_dist_matrix(inst["coords"])

        for solver_id in ["SCIP", "CBC"]:
            for t in time_limits:
                config_name = f"mip_{solver_id}_{t}s"
                obj, ms = solve_tsp_mip(dist, inst["n"], t, solver_id)
                if obj == float("inf"):
                    gap = float("inf")
                    print(f"{inst['name']},{config_name},0,inf,0.00,{ms:.1f},inf,0,time")
                else:
                    gap = (obj - inst["optimal"]) / inst["optimal"] * 100.0
                    print(f"{inst['name']},{config_name},0,{obj:.2f},0.00,{ms:.1f},{gap:.2f},0,time")
                sys.stdout.flush()

    # BinPacking
    weights = [7, 5, 3, 4, 6, 2, 8, 1]
    print("  [mip] BinPacking", file=sys.stderr)
    for solver_id in ["SCIP", "CBC"]:
        for t in [1, 10]:
            config_name = f"mip_{solver_id}_{t}s"
            obj, ms = solve_binpacking_mip(weights, 6, 10, t, solver_id)
            if obj == float("inf"):
                print(f"BinPack8,{config_name},0,inf,0.00,{ms:.1f},inf,0,time")
            else:
                gap = (obj - 4) / 4 * 100.0
                print(f"BinPack8,{config_name},0,{obj:.2f},0.00,{ms:.1f},{gap:.2f},0,time")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
