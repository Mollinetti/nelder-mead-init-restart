# Initialization and restarts in the Nelder–Mead method

Reproducibility artifact for **“Restarts substitute for initialization in the
Nelder–Mead method, and which restart trigger wins depends on dimension.”**
Paper source in [`paper/`](paper/).

## What the study found

Seven initialization procedures crossed with six restart triggers and two restart
constructions, on the COCO/BBOB noiseless suite at *n* ∈ {2, 5, 10, 20}, plus an
unshifted analytic control. **43,050 runs.**

- **Restarts substitute for initialization.** The spread in initializer mean rank
  falls from 2.68 under a single descent to 0.89–1.24 once any trigger is
  enabled, and the MATLAB `fminsearch` initializer — worst of seven on its own —
  becomes unremarkable. The substitution erodes with dimension.
- **The best trigger depends on *n*,** and the two published detectors sit at
  opposite ends of that dependence. GBNM's simplex-degeneracy test degrades from
  74.5% of problems solved at *n*=2 to 4.2% at *n*=20 (against 11.4% for not
  restarting at all), while Kelley's sufficient-decrease test moves the other
  way. Both trends are significant; the individual point comparisons at *n*=20
  are not, and the paper says so.
- **Two trivial triggers** — restart when the simplex is small, or when its
  vertex values are flat — are the most reliable across the range.
- A sweep of GBNM's tolerances over 40 orders of magnitude shows its default is
  badly mis-set at high dimension, but that **no setting makes the criterion
  better than a plain size test**, and that its edge-ratio tolerance is inert in
  11 of 12 cells.

## Layout

| Path | Contents |
|---|---|
| `NM-new/src/nelder_mead/` | Nelder–Mead and variants (adaptive, Kelley oriented-restart, safeguarded, gradient), 7 initializers, restart strategies and constructions, barrier methods |
| `NM-new/tests/` | 1163 tests, including McKinnon-stagnation regression tests for the Kelley implementation |
| `NM-new/benchmarks/scripts/` | Experiment runners and analysis |
| `NM-new/benchmarks/results/` | Raw per-run records and generated analyses |
| `paper/` | LaTeX source, bibliography, figures, and a ledger of deferred work and known limitations |

## Reproducing

```bash
python -m venv venv
./venv/bin/pip install -e NM-new
./venv/bin/pip install coco-experiment      # its meson build needs `python` on PATH
./venv/bin/python -m pytest NM-new/tests -q # 1163 tests
```

Experiments, in the order they were run:

```bash
cd NM-new
python benchmarks/scripts/run_init_restart_factorial.py --seeds 15   #  9,450 runs
python benchmarks/scripts/run_bbob_factorial.py --dims 2 5 10        # 25,200 runs
python benchmarks/scripts/run_bbob_factorial.py --dims 20 \
       --out benchmarks/results/bbob_factorial_d20.json              #  8,400 runs
python benchmarks/scripts/merge_and_validate.py                      # merge, then re-test 8 claims
python benchmarks/scripts/run_tolerance_calibration.py               #  7,200 runs
python benchmarks/scripts/run_budget_sensitivity.py                  #  2,880 runs

python benchmarks/scripts/analyze_init_restart.py
python benchmarks/scripts/analyze_tolerance_calibration.py
python benchmarks/scripts/paper_statistics.py
python benchmarks/scripts/make_paper_figures.py
```

`bbob_factorial_merged.json` is **derived** and not committed. `merge_and_validate.py`
regenerates it and re-tests the eight headline claims against the merged data,
printing HOLDS or BROKEN for each — which is how the *n*=20 additions were
checked, and how three earlier claims were found to have broken.

## Notes on method

Each experiment script states its acceptance criteria in its docstring **before**
the results, so the verdicts are not chosen after the fact. Several were caught
that way: a tolerance grid that never reached the disabled end, a harness bug
that silently logged zero restarts, and a gate written against the wrong metric.

The analysis reports two definitions of “solved” — the Moré–Wild criterion
relative to the best value any arm reached, and an absolute target against
`f_opt` recovered from COCO — because they disagree in sign at *n*=20. Both are
reported rather than one being chosen.

## Environment

Python 3.14.6, NumPy 2.x, SciPy 1.x, `coco-experiment` 2.8.2, macOS/arm64.
Every algorithm run uses seed 101.

## License

See [`NM-new/LICENSE`](NM-new/LICENSE).
