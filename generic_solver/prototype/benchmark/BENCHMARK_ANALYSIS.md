# Exp1: Generality Validation (default_g2k)

| Instance | Best obj | Avg obj | Gap % | All 5 agree? | Hit optimal? |
|----------|----------|---------|-------|---------------|--------------|
| Assign4 | 13.0 | 13.00 | -7.14 | Yes | No |
| BinPack8 | 4.0 | 4.00 | 0.00 | Yes | Yes |
| CVRP10 | 200.0 | 200.00 | 0.00 | Yes | Yes |
| GraphColor10 | 0.0 | 0.00 | 0.00 | Yes | Yes |
| JSP3x3_Int | 12.0 | 12.00 | 0.00 | Yes | Yes |
| JSP3x3_Perm | 12.0 | 12.00 | 0.00 | Yes | Yes |
| Knapsack6 | 30.0 | 30.00 | 0.00 | Yes | Yes |
| LoadBal8 | 14.0 | 14.00 | 0.00 | Yes | Yes |
| QAP5 | 58.0 | 58.00 | 16.00 | Yes | No |
| Schedule3x4 | 21.0 | 21.00 | 0.00 | Yes | Yes |
| TSP5 | 18.0 | 18.00 | 0.00 | Yes | Yes |
| VRPTW8 | 162.0 | 162.00 | 0.00 | Yes | Yes |

**Instances hitting optimal: 10/12** (Assign4 found 13, better than ref 14)

# Exp2: TSP Quality (time budgets)

| Instance | Time | Best obj | Avg obj | Best gap% | Avg gap% |
|----------|------|----------|---------|----------|----------|
| eil51 | t1s | 439.0 | 459.0 | 3.05 | 7.75 |
| eil51 | t5s | 426.0 | 426.4 | 0.00 | 0.09 |
| eil51 | t10s | 426.0 | 426.4 | 0.00 | 0.09 |
| eil51 | t30s | 426.0 | 426.4 | 0.00 | 0.09 |
| eil51 | t60s | 426.0 | 426.4 | 0.00 | 0.09 |
| kroA100 | t1s | 21282.0 | 21599.2 | 0.00 | 1.49 |
| kroA100 | t5s | 21282.0 | 21567.2 | 0.00 | 1.34 |
| kroA100 | t10s | 21282.0 | 21567.2 | 0.00 | 1.34 |
| kroA100 | t30s | 21282.0 | 21559.6 | 0.00 | 1.30 |
| kroA100 | t60s | 21282.0 | 21559.6 | 0.00 | 1.30 |
| ch150 | t1s | 7163.0 | 7314.2 | 9.73 | 12.04 |
| ch150 | t5s | 6690.0 | 6765.2 | 2.48 | 3.63 |
| ch150 | t10s | 6690.0 | 6765.2 | 2.48 | 3.63 |
| ch150 | t30s | 6690.0 | 6760.4 | 2.48 | 3.56 |
| ch150 | t60s | 6690.0 | 6757.2 | 2.48 | 3.51 |
| tsp225 | t1s | 7057.0 | 7179.4 | 80.21 | 83.34 |
| tsp225 | t5s | 4110.0 | 4144.6 | 4.95 | 5.84 |
| tsp225 | t10s | 4046.0 | 4083.0 | 3.32 | 4.26 |
| tsp225 | t30s | 4046.0 | 4082.4 | 3.32 | 4.25 |
| tsp225 | t60s | 4046.0 | 4082.4 | 3.32 | 4.25 |
| lin318 | t1s | 123743.0 | 126919.6 | 194.42 | 201.98 |
| lin318 | t5s | 52415.0 | 53394.0 | 24.71 | 27.04 |
| lin318 | t10s | 44008.0 | 44933.4 | 4.71 | 6.91 |
| lin318 | t30s | 43342.0 | 44036.8 | 3.12 | 4.78 |
| lin318 | t60s | 43342.0 | 44036.8 | 3.12 | 4.78 |
| pcb442 | t1s | 522526.0 | 527389.0 | 929.04 | 938.62 |
| pcb442 | t5s | 522526.0 | 527389.0 | 929.04 | 938.62 |
| pcb442 | t10s | 522526.0 | 527389.0 | 929.04 | 938.62 |
| pcb442 | t30s | 522526.0 | 527389.0 | 929.04 | 938.62 |
| pcb442 | t60s | 522526.0 | 527389.0 | 929.04 | 938.62 |

# Exp3: VRP Quality (A-n32-k5)

| Time | Best obj | Avg obj | Best gap% | Avg gap% |
|------|----------|---------|----------|----------|
| t1s | 784.0 | 784.0 | 0.00 | 0.00 |
| t5s | 784.0 | 784.0 | 0.00 | 0.00 |
| t10s | 784.0 | 784.0 | 0.00 | 0.00 |
| t30s | 784.0 | 784.0 | 0.00 | 0.00 |

# Exp4: Scalability (gap vs problem size N)

| N | Instance | 5s gap% | 10s gap% | 30s gap% |
|---|----------|---------|----------|----------|
| 51 | eil51 | 0.09 | 0.09 | 0.09 |
| 100 | kroA100 | 1.34 | 1.34 | 1.30 |
| 150 | ch150 | 3.63 | 3.63 | 3.56 |
| 225 | tsp225 | 5.84 | 4.26 | 4.25 |
| 318 | lin318 | 27.04 | 6.91 | 4.78 |
| 442 | pcb442 | 938.62 | 938.62 | 938.62 |

## Exp5: kroA100 (opt=21282)

### Additive: HC → SA → SA_Isl4 → SA_Isl4_CX → Full

| Config | Best obj | Avg obj | Best gap% | Avg gap% |
|--------|----------|---------|-----------|----------|
| HC | 21305.0 | 21357.40 | 0.11 | 0.35 |
| SA | 21282.0 | 21314.40 | 0.00 | 0.15 |
| SA_Isl4 | 21282.0 | 21457.80 | 0.00 | 0.83 |
| SA_Isl4_CX | 21282.0 | 21457.80 | 0.00 | 0.83 |
| Full | 21282.0 | 21567.20 | 0.00 | 1.34 |

### Leave-one-out: Full vs Full_noX

| Config | Best obj | Avg obj | Best gap% | Avg gap% | vs Full (avg gap) |
|--------|----------|---------|-----------|----------|-------------------|
| Full | 21282.0 | 21567.20 | 0.00 | 1.34 | — |
| Full_noSA | 21292.0 | 21408.40 | 0.05 | 0.59 | -0.75 |
| Full_noIsl | 21282.0 | 21324.20 | 0.00 | 0.20 | -1.14 |
| Full_noCX | 21378.0 | 21464.00 | 0.45 | 0.86 | -0.48 |
| Full_noAOS | 21282.0 | 21457.80 | 0.00 | 0.83 | -0.51 |

## Exp5: ch150 (opt=6528)

### Additive: HC → SA → SA_Isl4 → SA_Isl4_CX → Full

| Config | Best obj | Avg obj | Best gap% | Avg gap% |
|--------|----------|---------|-----------|----------|
| HC | 6612.0 | 6646.00 | 1.29 | 1.81 |
| SA | 6546.0 | 6570.00 | 0.28 | 0.64 |
| SA_Isl4 | 6543.0 | 6580.40 | 0.23 | 0.80 |
| SA_Isl4_CX | 6543.0 | 6580.40 | 0.23 | 0.80 |
| Full | 6690.0 | 6765.20 | 2.48 | 3.63 |

### Leave-one-out: Full vs Full_noX

| Config | Best obj | Avg obj | Best gap% | Avg gap% | vs Full (avg gap) |
|--------|----------|---------|-----------|----------|-------------------|
| Full | 6690.0 | 6765.20 | 2.48 | 3.63 | — |
| Full_noSA | 6580.0 | 6712.80 | 0.80 | 2.83 | -0.80 |
| Full_noIsl | 6576.0 | 6617.00 | 0.74 | 1.36 | -2.27 |
| Full_noCX | 6593.0 | 6717.20 | 1.00 | 2.90 | -0.74 |
| Full_noAOS | 6543.0 | 6580.40 | 0.23 | 0.80 | -2.83 |

## Exp5: BinPack20 (opt=7)

### Additive: HC → SA → SA_Isl4 → SA_Isl4_CX → Full

| Config | Best obj | Avg obj | Best gap% | Avg gap% |
|--------|----------|---------|-----------|----------|
| HC | 7.0 | 7.00 | 0.00 | 0.00 |
| SA | 7.0 | 7.00 | 0.00 | 0.00 |
| SA_Isl4 | 7.0 | 7.00 | 0.00 | 0.00 |
| SA_Isl4_CX | 7.0 | 7.00 | 0.00 | 0.00 |
| Full | 7.0 | 7.00 | 0.00 | 0.00 |

### Leave-one-out: Full vs Full_noX

| Config | Best obj | Avg obj | Best gap% | Avg gap% | vs Full (avg gap) |
|--------|----------|---------|-----------|----------|-------------------|
| Full | 7.0 | 7.00 | 0.00 | 0.00 | — |
| Full_noSA | 7.0 | 7.00 | 0.00 | 0.00 | +0.00 |
| Full_noIsl | 7.0 | 7.00 | 0.00 | 0.00 | +0.00 |
| Full_noCX | 7.0 | 7.00 | 0.00 | 0.00 | +0.00 |
| Full_noAOS | 7.0 | 7.00 | 0.00 | 0.00 | +0.00 |

## Exp5: GraphColor20 (opt=0)

### Additive: HC → SA → SA_Isl4 → SA_Isl4_CX → Full

| Config | Best obj | Avg obj | Best gap% | Avg gap% |
|--------|----------|---------|-----------|----------|
| HC | 0.0 | 0.00 | 0.00 | 0.00 |
| SA | 0.0 | 0.00 | 0.00 | 0.00 |
| SA_Isl4 | 0.0 | 0.00 | 0.00 | 0.00 |
| SA_Isl4_CX | 0.0 | 0.00 | 0.00 | 0.00 |
| Full | 0.0 | 0.00 | 0.00 | 0.00 |

### Leave-one-out: Full vs Full_noX

| Config | Best obj | Avg obj | Best gap% | Avg gap% | vs Full (avg gap) |
|--------|----------|---------|-----------|----------|-------------------|
| Full | 0.0 | 0.00 | 0.00 | 0.00 | — |
| Full_noSA | 0.0 | 0.00 | 0.00 | 0.00 | +0.00 |
| Full_noIsl | 0.0 | 0.00 | 0.00 | 0.00 | +0.00 |
| Full_noCX | 0.0 | 0.00 | 0.00 | 0.00 | +0.00 |
| Full_noAOS | 0.0 | 0.00 | 0.00 | 0.00 | +0.00 |

## Exp5: Schedule5x6 (opt=48)

### Additive: HC → SA → SA_Isl4 → SA_Isl4_CX → Full

| Config | Best obj | Avg obj | Best gap% | Avg gap% |
|--------|----------|---------|-----------|----------|
| HC | 48.0 | 48.00 | 0.00 | 0.00 |
| SA | 48.0 | 48.00 | 0.00 | 0.00 |
| SA_Isl4 | 48.0 | 48.00 | 0.00 | 0.00 |
| SA_Isl4_CX | 48.0 | 48.00 | 0.00 | 0.00 |
| Full | 48.0 | 48.00 | 0.00 | 0.00 |

### Leave-one-out: Full vs Full_noX

| Config | Best obj | Avg obj | Best gap% | Avg gap% | vs Full (avg gap) |
|--------|----------|---------|-----------|----------|-------------------|
| Full | 48.0 | 48.00 | 0.00 | 0.00 | — |
| Full_noSA | 48.0 | 48.00 | 0.00 | 0.00 | +0.00 |
| Full_noIsl | 48.0 | 48.00 | 0.00 | 0.00 | +0.00 |
| Full_noCX | 48.0 | 48.00 | 0.00 | 0.00 | +0.00 |
| Full_noAOS | 48.0 | 48.00 | 0.00 | 0.00 | +0.00 |

## Exp5: JSP4x3_Perm (opt=16)

### Additive: HC → SA → SA_Isl4 → SA_Isl4_CX → Full

| Config | Best obj | Avg obj | Best gap% | Avg gap% |
|--------|----------|---------|-----------|----------|
| HC | 16.0 | 16.00 | 0.00 | 0.00 |
| SA | 16.0 | 16.00 | 0.00 | 0.00 |
| SA_Isl4 | 16.0 | 16.00 | 0.00 | 0.00 |
| SA_Isl4_CX | 16.0 | 16.00 | 0.00 | 0.00 |
| Full | 16.0 | 16.00 | 0.00 | 0.00 |

### Leave-one-out: Full vs Full_noX

| Config | Best obj | Avg obj | Best gap% | Avg gap% | vs Full (avg gap) |
|--------|----------|---------|-----------|----------|-------------------|
| Full | 16.0 | 16.00 | 0.00 | 0.00 | — |
| Full_noSA | 16.0 | 16.00 | 0.00 | 0.00 | +0.00 |
| Full_noIsl | 16.0 | 16.00 | 0.00 | 0.00 | +0.00 |
| Full_noCX | 16.0 | 16.00 | 0.00 | 0.00 | +0.00 |
| Full_noAOS | 16.0 | 16.00 | 0.00 | 0.00 | +0.00 |

### AOS Impact (ch150): Full vs Full_noAOS
- **Full (with AOS)**: avg obj=6765.2, avg gap=3.63%
- **Full_noAOS**: avg obj=6580.4, avg gap=0.80%
- **Conclusion**: With AOS fix, Full is WORSE (gap 3.63% vs 0.80%). AOS still harmful on ch150.

### AOS Impact (kroA100): Full vs Full_noAOS
- **Full (with AOS)**: avg obj=21567.2, avg gap=1.34%
- **Full_noAOS**: avg obj=21457.8, avg gap=0.83%
- **Conclusion**: With AOS, Full is WORSE on kroA100 too (gap 1.34% vs 0.83%).

# Exp6: Time-Quality Curves

## kroA100

| Time | Best obj | Avg obj | Best gap% | Avg gap% |
|------|----------|---------|----------|----------|
| t0.5s | 21282.0 | 21599.2 | 0.00 | 1.49 |
| t1.0s | 21282.0 | 21599.2 | 0.00 | 1.49 |
| t2.0s | 21282.0 | 21567.2 | 0.00 | 1.34 |
| t5.0s | 21282.0 | 21567.2 | 0.00 | 1.34 |
| t10.0s | 21282.0 | 21567.2 | 0.00 | 1.34 |
| t20.0s | 21282.0 | 21559.6 | 0.00 | 1.30 |
| t30.0s | 21282.0 | 21559.6 | 0.00 | 1.30 |
| t60.0s | 21282.0 | 21559.6 | 0.00 | 1.30 |

## ch150

| Time | Best obj | Avg obj | Best gap% | Avg gap% |
|------|----------|---------|----------|----------|
| t0.5s | 8614.0 | 9046.8 | 31.95 | 38.58 |
| t1.0s | 7196.0 | 7320.8 | 10.23 | 12.14 |
| t2.0s | 6705.0 | 6814.0 | 2.71 | 4.38 |
| t5.0s | 6690.0 | 6765.2 | 2.48 | 3.63 |
| t10.0s | 6690.0 | 6765.2 | 2.48 | 3.63 |
| t20.0s | 6690.0 | 6765.2 | 2.48 | 3.63 |
| t30.0s | 6690.0 | 6760.4 | 2.48 | 3.56 |
| t60.0s | 6690.0 | 6757.2 | 2.48 | 3.51 |

# Exp5: Component Summary

| Problem | SA helps? | Islands help? | CX helps? | AOS helps? |
|---------|-----------|---------------|-----------|-----------|
| kroA100 | No | No | No | No |
| ch150 | No | No | No | No |
| BinPack20 | — | — | — | — |
| GraphColor20 | — | — | — | — |
| Schedule5x6 | — | — | — | — |
| JSP4x3_Perm | — | — | — | — |

*— = All configs hit optimal; no signal. For kroA100 and ch150, removing any component (SA, Islands, CX, or AOS) improves avg gap.*

---

## Exp5: Component Interpretation (TSP / permutation problems)

**kroA100 (permutation TSP):**
- **Additive**: SA improves over HC (0.35%→0.15% avg gap). Adding Islands+CX+AOS degrades (0.15%→0.83%→1.34%).
- **Leave-one-out**: Removing Islands gives largest improvement (-1.14 pp). Removing AOS (-0.51 pp), SA (-0.75 pp), or CX (-0.48 pp) also helps.
- **Conclusion**: SA_Isl4_CX (no AOS) is best; Full with AOS is worst.

**ch150 (permutation TSP):**
- **Additive**: SA improves over HC (1.81%→0.64%). SA_Isl4_CX matches SA_Isl4 at 0.80%. Full degrades to 3.63%.
- **Leave-one-out**: Removing AOS gives largest improvement (-2.83 pp). Removing Islands (-2.27 pp), SA (-0.80 pp), or CX (-0.74 pp) also helps.
- **Conclusion**: AOS is strongly harmful (0.80%→3.63%). Islands also hurt. SA_Isl4_CX or Full_noAOS is best.

**AOS parameter fix verdict:** The old report showed AOS harmful (ch150: 0.80%→2.87%). The new data shows **AOS still harmful** (ch150: 0.80%→3.63%). The recent AOS parameter tuning (higher floor, lower cap, faster EMA) did **not** fix the issue—Full with AOS remains worse than Full_noAOS on both kroA100 and ch150.
