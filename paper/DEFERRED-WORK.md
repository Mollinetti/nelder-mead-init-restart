# Deferred work — initialization × restart paper

Opened 2026-09-01; item 1 closed 2026-09-02. Everything here was **consciously
deferred**, not overlooked. Item 2 is now the one a referee is most likely to
raise. Nothing here blocks drafting, and all of it blocks submission except where
marked optional.

---

## 1. Factor C — restart *construction* ✅ RESOLVED 2026-09-02

**Was:** every arm re-seeded with `gaussian_best` except `progress`, which used
Kelley's oriented restart, so the Kelley arm confounded trigger with construction.

**Done.** `oriented_restart()` extracted to
[simplex_operations.py](../NM-new/src/nelder_mead/core/simplex_operations.py) as a
shared construction; `KelleyNelderMead` now delegates to it (McKinnon
single-restart regression test still passes). Construction is a real factor:
`gaussian_best` × `oriented` crossed with the four shared triggers, 70 arms,
25,200 BBOB runs.

**Outcome: the confound resolved in favour of the original claim.** At *both*
constructions the trivial triggers beat the principled detectors — by 9–15 points
at `gaussian_best` and 7–12 at `oriented`. Construction has a large main effect
of its own (`oriented` costs every trigger 11.5–16.7 points) but does not reorder
the triggers.

**Bonus finding.** The oriented restart shrinks the simplex, so pairing it with a
size or flatness trigger re-fires immediately: 615–798 restarts/run against 27–41
for `gaussian_best`. Pairing it with the degeneracy test does not (186), because
an oriented restart has Hadamard ratio 1 by construction. This is why Kelley pairs
his construction with a decrease-based detector — it is immune to both loops.

**Still open on this axis:** GBNM's own probabilistic re-seeding (Gaussian-kernel
density over past search points, Luersen & Le Riche §2.1) is not implemented and
would be the natural third construction level.

---

## 2. Dimension and instance coverage — HALF RESOLVED 2026-09-08

### ✅ Done: n = 20

Added n = 20, instances 1–5, all 70 arms (8,400 runs, ~19 h). Merged to **33,600
BBOB records** over n = 2/5/10/20. It was worth it: **three of eight claims broke**
and the paper's spine changed from "simple triggers beat principled ones" to
"the best trigger depends on dimension".

What the fourth point revealed:
- The degeneracy trigger degrades to **worse than no restart** (4.2% vs 11.4% at
  τ=1e-7) and its firing share hits **100%** — GBNM at n=20 *is* its degeneracy
  test. Claim strengthened.
- **Kelley reverses**: −5.5 / −3.5 / +6.9 / **+15.0** vs no-restart across
  n=2/5/10/20. The earlier "indistinguishable from doing nothing" was an artifact
  of stopping at n=10.
- Absorption is **not monotone** (81/79/60/67%); the paper now reports absolute
  spreads instead of the ratio, which is monotone and not an artifact.
- A **metric disagreement** appeared at n=20 only (see item 6).

### ❌ Still open: instances 6–15

Every dimension still uses **one instance seed per problem** (instances 1–5, one
seed each). Estimated ~14 h across all four dimensions. This is now the largest
remaining exposure, and the paper states it as the first limitation.

**Also still open:** n = 40. Deliberately skipped — at a 2000n budget the
no-restart arm is already down to 11.4% at n=20, so at n=40 the spread statistic
likely stops being interpretable (uniform failure compresses ranks, the same
artifact flagged for the degeneracy arms at n=20). Worth doing only with a larger
budget, which changes the comparison.

---

## 3. Calibrate εs3 / εs4 ✅ RESOLVED 2026-09-10

7,200 runs. Verdict **A1**: no tolerance is significantly better than a plain size
test at any dimension (exact binomial, Bonferroni over 21 populated cells).

Two bonus findings: **εs3 is inert in 11 of 12 cells** (the Hadamard half is the
only real knob), and **the default is badly mis-set** — dropping εs4 from 1e-6 to
1e-40 takes the n=20 win rate from 20.0% to 54.2% and the restart rate from 442
to 13 per run. So "actively harmful" is a statement about the default; retuned,
the criterion is serviceable but redundant.

**Process note.** The first grid stopped at εs4=1e-12, where the trigger still
fired 39–111×/run — it never reached the disabled end, and A1 could not have been
claimed from it. The pre-registered check caught this. A separate harness bug
(restart counts unlogged because `trace_conditioning` was off) was caught by the
G1 gate; objective values were verified bit-identical afterwards, so the sweep
data stood without re-running.

---

## 3b. Budget sensitivity ✅ RESOLVED 2026-09-10

2,880 runs at 2000n and 4000n, n ∈ {10,20}. **B2**: quadrupling the budget leaves
the arm ordering *identical* (Kendall τ = 1.000), no arm moving more than 0.21
rank. The dimensional story is not a budget artifact.

Also supplied the reference the tolerance sweep lacked: at its best tolerance the
degeneracy trigger beats no-restart on 84–96% of problems, confirming the
narrowing above.

**Process note.** B1 failed as written — but on the gate's specification, not the
data: it was stated in solved-fraction terms while the run measures win rate, and
those disagree in *sign* at n=20. The run reproduces the main experiment exactly
(40.8% vs 40.8%).

---

## 3c. f_L relativity ✅ RESOLVED 2026-09-10

`f_opt` **is** recoverable from this COCO build via the problem's best-parameter
accessor. Extracted for all 480 problems; no run in the study beats it. Every
headline claim re-verified against absolute targets (Δf ≤ 1e-8): both trends hold
(Kelley z=+3.46, p=5.4e-4; degeneracy z=−6.08, p=1.2e-9), and the degeneracy gap
at n=20 is starker absolutely (0.1% vs 3.7%) than relatively. Point comparisons
weaken at n=20 because almost nothing reaches 1e-8 there.

One claim did not survive: "flat is best at n≤10" is metric-dependent, and flat
and size are now reported as indistinguishable except at n=2.

---

## 3d. Statistics ✅ RESOLVED 2026-09-10

Holm correction across all 32 comparisons, exact McNemar on paired per-problem
outcomes, paired win rate as effect size, Cochran–Armitage for the dimensional
trends. Degeneracy claims survive correction; **Kelley's point advantage at n=20
does not** (p=0.094) — it is stated as a trend claim only, and the trend is
significant under both metrics.

**Process note.** The first attempt used unpaired Vargha–Delaney on paired data,
which returned "negligible" for comparisons whose paired win rates run 20%–97%.

---

## 4. Second suite — Moré–Wild 53 (OPTIONAL)

Not implemented; zero presence in the repo. Would harden the "unshifted analytic
functions mislead" claim by adding a third, independent instrument, and its noisy
and piecewise-smooth variants probe restart behaviour where NM is known to
stagnate. Estimated ~2 weeks per the original plan. **Optional** — BBOB plus the
analytic control already supports the claim.

---

## 5. Presentation gaps — MOSTLY RESOLVED 2026-09-10

- ✅ **Figures.** Three, in `paper/figures/`, palette validated with the dataviz
  validator (`size` moved to violet because magenta-against-orange failed the
  normal-vision floor at ΔE 12.9). Rendering and inspecting them caught three
  defects the validator could not: `\%` printing literally, colliding
  annotations, and labels sitting on a line.
- ✅ **Statistics.** See 3d.
- ✅ **Environment capture.** Python 3.14.6, coco-experiment 2.8.2, macOS/arm64,
  seed 101, recorded in the reproducibility paragraph.
- ✅ **Table width.** `tab:spread` was 13 columns and would have overflowed
  `\textwidth` at 11pt; split into two stacked blocks. All tabulars are now ≤10
  columns.
- ✅ **Repository URL** set to https://github.com/Mollinetti/NMs/tree/GBNM.
- ⚠️ **That repo is PRIVATE.** A reproducibility URL referees cannot open is
  worse than none. Make it public, or split this branch into a public repo,
  before submission.
- ⏸️ **Zenodo DOI** deliberately skipped: nothing is published yet.
- ❌ **Affiliation** still a placeholder; note the CAPES agreement requires the
  *corresponding* author to hold a participating Brazilian affiliation at
  acceptance.
- ❌ **No LaTeX toolchain on this machine**, so the document has never been
  compiled. Brace balance, float/tabular pairing and cross-reference labels were
  checked mechanically, but typesetting is unverified.

---

## 6. Known confounds to disclose in the paper (not to fix)

These are properties of the design, not defects, and the paper should state them
rather than engineer them away:

- The mediation analysis is **correlational and within-cell**. It shows conditioning
  tracks outcome; it does not establish or exclude a causal role.
- ~~`f_L` is relative to the compared arms~~ — resolved, see 3c. Retained as a
  *reported* caveat in the paper because the primary tables still use the
  relative measure, with the absolute verification alongside.
- Restart construction has **two** levels, not three; GBNM's probabilistic
  re-seeding is absent (item 1, residual).
