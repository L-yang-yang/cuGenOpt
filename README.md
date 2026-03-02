**GenSolver** is a CUDA-based general-purpose heuristic optimization framework. It is **problem-agnostic** (does not depend on specific problem types such as TSP, VRP, assignment, scheduling, etc.). Instead, it provides:

* **Generic solution encoding structure**: represented as a 2D array
* **Generic search operators**: swap, reverse, insert, relocate, etc.
* **Flexible multi-objective comparison strategies**: weighted sum, lexicographic order, feasibility-first, etc.
* **Efficient GPU parallel scheduling**: population-level + neighborhood-level parallelism

The user only needs to define:

1. **Problem data** (distance matrix, constraints, etc.)
2. **Evaluation function** (array → objective value)

The framework handles all GPU parallelization, population management, and search scheduling. 
