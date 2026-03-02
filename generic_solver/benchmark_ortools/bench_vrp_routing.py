"""
OR-Tools Routing (GLS) 求解 CVRP — 与 GenSolver exp3 对比
"""
import sys
import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from instances import load_vrp, euc2d_dist_matrix, VRP_INSTANCES

TIME_BUDGETS = [1, 5, 10, 30]


def solve_cvrp_routing(dist, demands, n, n_vehicles, capacity, time_limit_sec):
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return dist[from_node][to_node]

    transit_id = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

    def demand_callback(idx):
        node = manager.IndexToNode(idx)
        return demands[node]

    demand_id = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_id, 0, [capacity] * n_vehicles, True, "Capacity"
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.seconds = time_limit_sec

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if solution:
        obj = solution.ObjectiveValue()
    else:
        obj = float("inf")

    return obj, elapsed_ms


def main():
    print("instance,config,seed,obj,penalty,time_ms,gap_pct,generations,stop_reason")

    for entry in VRP_INSTANCES:
        inst = load_vrp(entry)
        print(f"  [vrp_routing] {inst['name']} (n={inst['n']})", file=sys.stderr)
        dist = euc2d_dist_matrix(inst["coords"])

        for t in TIME_BUDGETS:
            config_name = f"routing_GLS_{t}s"
            obj, elapsed_ms = solve_cvrp_routing(
                dist, inst["demands"], inst["n"],
                inst["n_vehicles"], inst["capacity"], t
            )
            gap = (obj - inst["optimal"]) / inst["optimal"] * 100.0 if inst["optimal"] > 0 else 0.0
            print(f"{inst['name']},{config_name},0,{obj:.2f},0.00,{elapsed_ms:.1f},{gap:.2f},0,time")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
