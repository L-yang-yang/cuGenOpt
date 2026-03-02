# GenSolver

A CUDA-based general-purpose heuristic optimization framework that is **problem-agnostic**.

## Why GenSolver?

Most optimization solvers are designed for a single problem type. GenSolver takes a different approach: **solve any combinatorial optimization problem with a single unified framework**.

| Feature | Description |
|---------|-------------|
| **Problem-Agnostic** | One framework for TSP, VRP, Knapsack, QAP, JSP, Scheduling, and more |
| **Zero Modeling** | Just define your data and evaluation function — no search algorithm expertise needed |
| **GPU-Accelerated** | Block-level neighborhood parallelism + population-level parallelism |
| **Adaptive** | Auto-tuning population size, island model, and operator selection (AOS) |
| **Production-Ready** | CUDA Graphs, Relation Matrix for LNS, comprehensive benchmark suite |

## Quick Example

```cpp
// Define your problem: TSP with distance matrix
struct TSPProblem : ProblemBase<TSPProblem, 1, 64> {
    const float* d_dist;
    int n;

    __device__ float compute_obj(int idx, const Sol& sol) const {
        float total = 0.0f;
        const int* route = sol.data[0];
        for (int i = 0; i < sol.dim2_sizes[0]; i++)
            total += d_dist[route[i] * n + route[(i + 1) % n]];
        return total;
    }

    ProblemConfig config() const {
        ProblemConfig cfg;
        cfg.encoding = EncodingType::Permutation;
        cfg.dim1 = 1;
        cfg.dim2_default = n;
        return cfg;
    }
};

// Solve it — framework handles all the rest
auto result = solver.solve(problem, config);
```

## Supported Problem Types

| Encoding | Problems |
|----------|----------|
| **Permutation** | TSP, QAP, Assignment, JSP, Scheduling |
| **Partition** | CVRP, VRPTW, Multiple TSP |
| **Binary** | Knapsack, Bin Packing |
| **Integer** | Graph Coloring, Load Balancing |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Layer                        │
│  Problem Data  │  Evaluator  │  (Optional) Custom  │
└────────┬────────┴──────┬───────┴────────┬──────────┘
         │               │                │
         ▼               ▼                ▼
┌─────────────────────────────────────────────────────┐
│                  Framework Core                     │
│  Solution   │  Comparator  │  Operators  │  Solver │
│  Storage    │  (Weighted/  │  (24+ ops)   │  Engine │
│  (2D Array) │   Lexicographic)│            │         │
└─────────────┴──────────────┴──────────────┴─────────┘
         │               │                │
         ▼               ▼                ▼
┌─────────────────────────────────────────────────────┐
│                   GPU Execution                     │
│  Memory Manager  │  Kernel Launch  │  Stream Mgmt  │
└─────────────────────────────────────────────────────┘
```

## Key Features

### Adaptive Search
- **Auto Population**: Automatically sizes population based on GPU resources
- **Island Model**: Adaptive island count based on population size
- **AOS (Adaptive Operator Selection)**: Two-layer weights with EMA updates

### Parallel Strategy
- **Block-Level**: 1 block = 1 solution, multiple threads search neighborhoods in parallel
- **Population-Level**: Multiple islands evolve independently with periodic migration

### Large Neighborhood Search (LNS)
- **Relation Matrix**: G/O matrices capture element groupings and ordering tendencies
- **Guided Rebuild**: Use learned patterns to reconstruct solutions

## Build & Run

```bash
cd generic_solver/prototype
make bench_tsp51
./bin/bench_tsp51
```

## Documentation

- [00 Overview](generic_solver/design/00_overview.md)
- [01 Solution Encoding](generic_solver/design/01_solution_encoding.md)
- [02 Objective System](generic_solver/design/02_objective_system.md)
- [04 Operators](generic_solver/design/04_operators.md)
- [06 Population](generic_solver/design/06_population.md)
- [07 Parallel Strategy](generic_solver/design/07_parallel_strategy.md)
