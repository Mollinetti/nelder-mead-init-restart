# Analytic suite

`9450` runs. Config: `{"seeds": 15, "budget": 10000, "dims": [2, 5, 10], "initializations": {"pfeffer": ["pfefferSimplex", 1.0], "spendley_full": ["spendleySimplex", 1.0], "spendley_10pct": ["spendleySimplex", 0.1], "spendley_2pct": ["spendleySimplex", 0.02], "uniform": ["uniform", 1.0], "gaussian": ["gaussian", 1.0], "positive_basis": ["minimalPositiveBasis", 1.0]}, "triggers": {"none": [], "size": ["small_simplex"], "conditioning": ["degenerate_simplex"], "gbnm": ["small_simplex", "degenerate_simplex"], "flat": ["flat_simplex"], "progress": []}, "restart_construction": "gaussian_best"}`

## Main effect of initialization, and whether the trigger changes it

Mean rank across problems (1 = best of 7 initializers), ranked within
(problem, dim, seed). `spread` is max-min mean rank: it is how much the
choice of initializer still matters once that trigger is switched on.

| trigger | pfeffer | spendley_full | spendley_10pct | spendley_2pct | uniform | gaussian | positive_basis | spread | Friedman p | cells |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 3.02 | 3.03 | 2.48 | 2.53 | 5.59 | 6.28 | 5.07 | **3.80** | 8e-182 | 225 |
| size | 2.90 | 2.82 | 2.70 | 2.55 | 5.47 | 6.35 | 5.22 | **3.80** | 7.1e-186 | 225 |
| conditioning | 3.10 | 3.08 | 3.11 | 3.03 | 5.20 | 5.95 | 4.52 | **2.92** | 4.5e-146 | 225 |
| gbnm | 2.83 | 2.82 | 2.74 | 2.55 | 5.51 | 6.24 | 5.32 | **3.69** | 4.7e-185 | 225 |
| flat | 2.63 | 2.79 | 2.80 | 2.72 | 5.65 | 6.22 | 5.20 | **3.60** | 1.1e-185 | 225 |
| progress | 2.78 | 3.08 | 2.53 | 2.46 | 5.70 | 6.28 | 5.16 | **3.82** | 6.3e-195 | 225 |

### Is the initializer ranking stable across triggers?

Kendall tau between each trigger's initializer ranking and the
single-descent (`none`) ranking. tau near 1 means the trigger left the
initializer ordering intact (they compound); tau near 0 means the
trigger scrambled it (they substitute).

| trigger | Kendall tau vs none | p |
|---|---|---|
| size | +0.810 | 0.0107 |
| conditioning | +0.619 | 0.069 |
| gbnm | +0.810 | 0.0107 |
| flat | +0.619 | 0.069 |
| progress | +0.905 | 0.00278 |

## GBNM trigger attribution

Under the combined `gbnm` trigger (small OR degenerate), which test
actually fires? Luersen & Le Riche report the tolerances are hard to tune
because a shrinking simplex is tagged degenerate before it is tagged small.

| dim | restarts/run | by degeneracy | by size | degeneracy share |
|---|---|---|---|---|
| 2 | 115.8 | 6 | 60803 | **0.0%** |
| 5 | 36.8 | 4532 | 14808 | **23.4%** |
| 10 | 24.2 | 6473 | 6216 | **51.0%** |

## Conditioning

Median over runs of the per-run **final** Hadamard ratio |det L|/prod||e||.
1.0 = orthogonal equal-length edges; 0 = collapsed to a subspace.

| init | none | size | conditioning | gbnm | flat | progress |
|---|---|---|---|---|---|---|
| pfeffer | 1.40e-01 | 2.18e-01 | 1.62e-01 | 2.05e-01 | 2.30e-01 | 2.51e-01 |
| spendley_full | 2.51e-01 | 2.04e-01 | 1.48e-01 | 2.09e-01 | 1.94e-01 | 9.80e-01 |
| spendley_10pct | 2.48e-01 | 1.97e-01 | 1.66e-01 | 2.04e-01 | 2.47e-01 | 7.55e-01 |
| spendley_2pct | 1.96e-01 | 2.24e-01 | 1.52e-01 | 2.24e-01 | 1.89e-01 | 2.62e-01 |
| uniform | 1.34e-01 | 1.71e-01 | 1.64e-01 | 1.71e-01 | 1.82e-01 | 2.11e-01 |
| gaussian | 9.51e-02 | 1.65e-01 | 7.65e-02 | 1.65e-01 | 1.42e-01 | 1.22e-01 |
| positive_basis | 1.65e-01 | 1.96e-01 | 1.27e-01 | 1.97e-01 | 2.54e-01 | 2.03e-01 |

### Conditioning over the whole run

Median Hadamard ratio across the trace, and the fraction of iterations
spent below 1e-6 -- i.e. inside GBNM's default degeneracy tolerance.

| init | median Hadamard | frac. iters below 1e-6 |
|---|---|---|
| pfeffer | 0.274 | 0.000 |
| spendley_full | 0.268 | 0.000 |
| spendley_10pct | 0.275 | 0.000 |
| spendley_2pct | 0.264 | 0.000 |
| uniform | 0.216 | 0.000 |
| gaussian | 0.147 | 0.000 |
| positive_basis | 0.245 | 0.000 |

## Does conditioning explain the initialization effect?

Spearman rho between a run's median Hadamard ratio and its final
objective value, computed within each (problem, dim, trigger) cell and
then pooled. Negative rho means better-conditioned runs ended lower,
i.e. conditioning mediates the effect. A rho near zero would say
conditioning is a bystander and the mechanism story is wrong.

| dim | median rho | fraction of cells with rho < 0 | cells |
|---|---|---|---|
| 2 | +0.153 | 0.39 | 28 |
| 5 | -0.760 | 0.83 | 30 |
| 10 | -0.665 | 0.97 | 30 |

---

# COCO/BBOB suite

`33600` runs. Config: `{"merged_from": ["bbob_factorial.json", "bbob_factorial_d20.json"], "base": {"suite": "bbob", "functions": "1-24", "dims": [2, 5, 10], "instances": 5, "budget_per_dim": 2000, "seed": 101, "constructions": ["gaussian_best", "oriented"], "reference": "f_L = best over all arms (More-Wild convention)"}, "extra": {"suite": "bbob", "functions": "1-24", "dims": [20], "instances": 5, "budget_per_dim": 2000, "seed": 101, "constructions": ["gaussian_best", "oriented"], "reference": "f_L = best over all arms (More-Wild convention)"}}`

## Main effect of initialization, and whether the trigger changes it

Mean rank across problems (1 = best of 7 initializers), ranked within
(problem, dim, seed). `spread` is max-min mean rank: it is how much the
choice of initializer still matters once that trigger is switched on.

| trigger | gaussian | pfeffer | positive_basis | spendley_10pct | spendley_2pct | spendley_full | uniform | spread | Friedman p | cells |
|---|---|---|---|---|---|---|---|---|---|---|
| conditioning | 4.60 | 4.68 | 3.43 | 3.84 | 4.32 | 3.19 | 3.94 | **1.49** | 5.9e-46 | 480 |
| flat | 4.38 | 5.36 | 3.38 | 3.75 | 4.62 | 2.81 | 3.69 | **2.55** | 1.3e-98 | 480 |
| gbnm | 4.61 | 4.63 | 3.41 | 3.87 | 4.35 | 3.17 | 3.94 | **1.46** | 2.3e-45 | 480 |
| none | 4.57 | 5.56 | 3.26 | 3.69 | 4.54 | 2.88 | 3.49 | **2.68** | 2e-124 | 480 |
| progress | 4.57 | 4.63 | 3.55 | 3.86 | 4.37 | 3.19 | 3.84 | **1.44** | 3.6e-40 | 480 |
| size | 4.44 | 5.14 | 3.40 | 3.84 | 4.65 | 2.97 | 3.56 | **2.17** | 5.7e-93 | 480 |

### Is the initializer ranking stable across triggers?

Kendall tau between each trigger's initializer ranking and the
single-descent (`none`) ranking. tau near 1 means the trigger left the
initializer ordering intact (they compound); tau near 0 means the
trigger scrambled it (they substitute).

| trigger | Kendall tau vs none | p |
|---|---|---|
| conditioning | +0.905 | 0.00278 |
| flat | +0.905 | 0.00278 |
| gbnm | +0.905 | 0.00278 |
| progress | +1.000 | 0.000397 |
| size | +0.905 | 0.00278 |

## GBNM trigger attribution

Under the combined `gbnm` trigger (small OR degenerate), which test
actually fires? Luersen & Le Riche report the tolerances are hard to tune
because a shrinking simplex is tagged degenerate before it is tagged small.

| dim | restarts/run | by degeneracy | by size | degeneracy share |
|---|---|---|---|---|
| 2 | 79.9 | 40699 | 26379 | **60.7%** |
| 5 | 93.5 | 66696 | 11869 | **84.9%** |
| 10 | 131.1 | 102611 | 7480 | **93.2%** |
| 20 | 463.3 | 389150 | 11 | **100.0%** |

## Conditioning

Median over runs of the per-run **final** Hadamard ratio |det L|/prod||e||.
1.0 = orthogonal equal-length edges; 0 = collapsed to a subspace.

| init | conditioning | flat | gbnm | none | progress | size |
|---|---|---|---|---|---|---|
| gaussian | 1.34e-03 | 1.38e-03 | 4.30e-03 | 4.57e-08 | 1.38e-03 | 2.70e-04 |
| pfeffer | 1.26e-03 | 5.89e-04 | 4.41e-03 | 1.77e-09 | 9.70e-04 | 2.42e-04 |
| positive_basis | 1.48e-03 | 1.88e-03 | 4.91e-03 | 1.99e-06 | 5.02e-03 | 5.04e-04 |
| spendley_10pct | 1.64e-03 | 5.59e-03 | 5.02e-03 | 2.62e-05 | 1.39e-03 | 3.18e-04 |
| spendley_2pct | 1.39e-03 | 1.95e-03 | 4.81e-03 | 2.32e-05 | 1.38e-03 | 3.32e-04 |
| spendley_full | 1.49e-03 | 3.11e-03 | 6.23e-03 | 2.30e-06 | 3.41e-03 | 3.42e-04 |
| uniform | 1.38e-03 | 1.13e-03 | 3.89e-03 | 3.07e-07 | 1.49e-03 | 2.46e-04 |

### Conditioning over the whole run

Median Hadamard ratio across the trace, and the fraction of iterations
spent below 1e-6 -- i.e. inside GBNM's default degeneracy tolerance.

| init | median Hadamard | frac. iters below 1e-6 |
|---|---|---|
| gaussian | 0.001 | 0.024 |
| pfeffer | 0.001 | 0.025 |
| positive_basis | 0.001 | 0.019 |
| spendley_10pct | 0.001 | 0.018 |
| spendley_2pct | 0.001 | 0.018 |
| spendley_full | 0.001 | 0.018 |
| uniform | 0.001 | 0.022 |

## Does conditioning explain the initialization effect?

Spearman rho between a run's median Hadamard ratio and its final
objective value, computed within each (problem, dim, trigger) cell and
then pooled. Negative rho means better-conditioned runs ended lower,
i.e. conditioning mediates the effect. A rho near zero would say
conditioning is a bystander and the mechanism story is wrong.

| dim | median rho | fraction of cells with rho < 0 | cells |
|---|---|---|---|
| 2 | -0.063 | 0.61 | 140 |
| 5 | -0.016 | 0.58 | 142 |
| 10 | -0.031 | 0.56 | 144 |
| 20 | -0.101 | 0.67 | 144 |

## Trigger x construction (the confound, resolved)

Fraction solved at tau = 1e-7. Reading *down* a column compares
triggers at fixed construction; reading *across* a row isolates the
re-seeding. `progress` (Kelley) exists only in its published pairing.

| trigger | gaussian_best | oriented | delta |
|---|---|---|---|
| conditioning | 34.7% | 27.7% | -7.0 |
| flat | 48.5% | 35.4% | -13.1 |
| gbnm | 36.2% | 27.4% | -8.8 |
| none | 31.1% | - | - |
| progress | - | 34.3% | - |
| size | 45.3% | 36.2% | -9.1 |

Restarts per run, same layout:

| trigger | gaussian_best | oriented |
|---|---|---|
| conditioning | 184.9 | 275.9 |
| flat | 33.4 | 672.6 |
| gbnm | 191.9 | 746.6 |
| none | 0.0 | - |
| progress | - | 18.5 |
| size | 21.8 | 517.6 |

## Fraction solved (More-Wild, tau = 1e-3), by BBOB group

| trigger | high conditioning unimodal | low moderate conditioning | multimodal global structure | multimodal weak structure | separable |
|---|---|---|---|---|---|
| conditioning | 83% | 69% | 20% | 36% | 49% |
| flat | 93% | 77% | 23% | 38% | 59% |
| gbnm | 83% | 70% | 22% | 38% | 51% |
| none | 82% | 64% | 12% | 32% | 52% |
| progress | 86% | 72% | 9% | 29% | 52% |
| size | 93% | 75% | 21% | 38% | 58% |