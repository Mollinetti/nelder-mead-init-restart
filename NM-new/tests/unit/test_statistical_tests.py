#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the statistical tests used in the paper's Tables 4 and 5.

Verifies:
- Friedman omnibus test against scipy computed on the same matrix by hand
- Mann-Whitney U pairwise comparison against a reference algorithm
- Holm-Bonferroni step-down correction against a hand-worked example
- The validation guards that stop a malformed comparison producing a p-value
"""

import numpy as np
import pytest
from scipy.stats import friedmanchisquare, mannwhitneyu

from nelder_mead.testing.batch_runner import ExperimentResult
from nelder_mead.testing.result_analyzer import ResultAnalyzer


def make_results(fitnesses, algorithm="alg", problem="prob", violations=None):
    """Build a list of ExperimentResult carrying the given fitness values."""
    if violations is None:
        violations = [0.0] * len(fitnesses)
    return [
        ExperimentResult(
            algorithm_name=algorithm,
            problem_name=problem,
            seed=i,
            best_solution=np.zeros(2),
            best_fitness=float(f),
            best_violation=float(v),
            fes_used=100,
        )
        for i, (f, v) in enumerate(zip(fitnesses, violations))
    ]


class TestDebComparisonValues:
    """Feasibility must dominate the objective when ranking constrained runs.

    This is the correction for a failure mode that would otherwise corrupt every
    table: an algorithm that never reaches the feasible region minimizes over a
    larger set, so it can report an objective *below* the true constrained
    optimum. Ranking on the raw objective would crown it.
    """

    def test_feasible_always_beats_infeasible(self):
        """Even a much worse feasible objective must outrank an infeasible one."""
        results = {
            "feasible_but_poor": make_results([1000.0], violations=[0.0]),
            "infeasible_but_low": make_results([-9999.0], violations=[5.0]),
        }
        values = ResultAnalyzer.deb_comparison_values(results)

        assert values["feasible_but_poor"][0] < values["infeasible_but_low"][0]

    def test_feasible_runs_ordered_by_objective(self):
        """Among feasible runs the objective is the whole story."""
        results = {"a": make_results([3.0, 1.0, 2.0], violations=[0.0, 0.0, 0.0])}
        values = ResultAnalyzer.deb_comparison_values(results)

        assert list(np.argsort(values["a"])) == [1, 2, 0]

    def test_infeasible_runs_ordered_by_violation(self):
        """Among infeasible runs the objective is irrelevant; violation decides."""
        results = {
            "a": make_results([-100.0, -200.0], violations=[1.0, 9.0]),
        }
        values = ResultAnalyzer.deb_comparison_values(results)

        # The run with the lower violation ranks better despite the worse objective
        assert values["a"][0] < values["a"][1]

    def test_reproduces_the_g05_failure_mode(self):
        """The concrete case that motivated this: an infeasible sub-optimal mean.

        On G05 the constrained optimum is 5126.50. An algorithm that stays
        infeasible reported 5115.74, below the optimum, and would have won on raw
        objective. It must lose here.
        """
        results = {
            "cNM": make_results([5115.74], violations=[5.101]),
            "DE": make_results([5127.09], violations=[0.0]),
        }
        values = ResultAnalyzer.deb_comparison_values(results)

        assert values["DE"][0] < values["cNM"][0]

    def test_all_infeasible_falls_back_to_violation_ordering(self):
        """With no feasible run anywhere, violation is the only signal."""
        results = {
            "a": make_results([1.0], violations=[3.0]),
            "b": make_results([2.0], violations=[1.0]),
        }
        values = ResultAnalyzer.deb_comparison_values(results)

        assert values["b"][0] < values["a"][0]

    def test_failed_runs_rank_below_every_completed_run(self):
        """A crashed run must be worse than the worst infeasible completed run."""
        results = {
            "a": make_results([1.0, np.inf], violations=[0.0, 0.0]),
            "b": make_results([2.0], violations=[100.0]),
        }
        values = ResultAnalyzer.deb_comparison_values(results)

        assert values["a"][1] > values["b"][0]

    def test_all_values_are_finite(self):
        """Mann-Whitney needs finite numbers; infinities would be filtered out."""
        results = {
            "a": make_results([1.0, np.inf], violations=[0.0, 2.0]),
            "b": make_results([2.0], violations=[np.inf]),
        }
        values = ResultAnalyzer.deb_comparison_values(results)

        assert all(np.all(np.isfinite(column)) for column in values.values())

    def test_preserves_run_order(self):
        """Values must align with the input runs so seeds stay paired."""
        results = {"a": make_results([5.0, 1.0, 3.0])}
        values = ResultAnalyzer.deb_comparison_values(results)

        assert values["a"].tolist() == [5.0, 1.0, 3.0]

    def test_respects_the_feasibility_tolerance(self):
        """A violation just under the tolerance counts as feasible."""
        results = {"a": make_results([1.0, 2.0], violations=[1e-9, 1e-3])}
        values = ResultAnalyzer.deb_comparison_values(results, feasibility_tol=1e-8)

        assert values["a"][0] == 1.0       # treated as feasible, raw objective kept
        assert values["a"][1] > 1.0        # treated as infeasible, offset applied


class TestHolmCorrection:
    """Holm's step-down procedure, checked against a hand-worked example."""

    def test_known_example(self):
        """Sorted [0.01, 0.03, 0.04] x [3, 2, 1] = [0.03, 0.06, 0.04] -> cummax."""
        adjusted = ResultAnalyzer.holm_correction({"a": 0.01, "b": 0.04, "c": 0.03})

        assert adjusted["a"] == pytest.approx(0.03)
        assert adjusted["c"] == pytest.approx(0.06)
        # 0.04 * 1 = 0.04, but monotonicity lifts it to the preceding 0.06
        assert adjusted["b"] == pytest.approx(0.06)

    def test_monotonic_in_sorted_order(self):
        """Adjusted p-values must never decrease along ascending raw order."""
        raw = {"a": 0.001, "b": 0.008, "c": 0.02, "d": 0.6, "e": 0.9}
        adjusted = ResultAnalyzer.holm_correction(raw)

        by_raw = sorted(raw, key=lambda k: raw[k])
        values = [adjusted[k] for k in by_raw]

        assert values == sorted(values)

    def test_clipped_to_one(self):
        """No adjusted p-value may exceed 1.0."""
        adjusted = ResultAnalyzer.holm_correction({"a": 0.5, "b": 0.6, "c": 0.9})

        assert all(v <= 1.0 for v in adjusted.values())

    def test_single_test_is_unchanged(self):
        """With one hypothesis there is no multiplicity to correct for."""
        adjusted = ResultAnalyzer.holm_correction({"a": 0.023})

        assert adjusted["a"] == pytest.approx(0.023)

    def test_more_powerful_than_bonferroni(self):
        """Holm must never be more conservative than plain Bonferroni."""
        raw = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
        adjusted = ResultAnalyzer.holm_correction(raw)

        for name, value in raw.items():
            assert adjusted[name] <= min(1.0, value * len(raw)) + 1e-12

    def test_empty_input(self):
        """An empty family is a no-op, not an error."""
        assert ResultAnalyzer.holm_correction({}) == {}

    def test_rejects_out_of_range(self):
        """A p-value outside [0, 1] signals a bug upstream and must not pass."""
        with pytest.raises(ValueError, match="outside the range"):
            ResultAnalyzer.holm_correction({"a": 1.5})


class TestFriedmanTest:
    """The omnibus test, checked against scipy on the same matrix."""

    def _three_algorithm_results(self):
        """Three algorithms on four problems, with a consistent winner."""
        data = {
            "cNM": {"p1": [1.0, 1.1, 1.2], "p2": [2.0, 2.1, 2.2],
                    "p3": [3.0, 3.1, 3.2], "p4": [4.0, 4.1, 4.2]},
            "DE": {"p1": [5.0, 5.1, 5.2], "p2": [6.0, 6.1, 6.2],
                   "p3": [7.0, 7.1, 7.2], "p4": [8.0, 8.1, 8.2]},
            "PSO": {"p1": [9.0, 9.1, 9.2], "p2": [10.0, 10.1, 10.2],
                    "p3": [11.0, 11.1, 11.2], "p4": [12.0, 12.1, 12.2]},
        }
        return {
            alg: {prob: make_results(vals, alg, prob) for prob, vals in probs.items()}
            for alg, probs in data.items()
        }

    def test_matches_scipy(self):
        """The statistic and p-value must equal scipy's on the same mean matrix."""
        results = self._three_algorithm_results()
        outcome = ResultAnalyzer.friedman_test(results)

        matrix = np.array(
            [
                [np.mean([r.best_fitness for r in results[alg][prob]])
                 for alg in ["cNM", "DE", "PSO"]]
                for prob in ["p1", "p2", "p3", "p4"]
            ]
        )
        expected_chi2, expected_p = friedmanchisquare(*matrix.T)

        assert outcome["statistic"] == pytest.approx(expected_chi2)
        assert outcome["p_value"] == pytest.approx(expected_p)

    def test_mean_ranks_order_the_algorithms(self):
        """Rank 1 is best; a dominant algorithm must hold the lowest mean rank."""
        outcome = ResultAnalyzer.friedman_test(self._three_algorithm_results())

        assert outcome["mean_ranks"]["cNM"] == pytest.approx(1.0)
        assert outcome["mean_ranks"]["DE"] == pytest.approx(2.0)
        assert outcome["mean_ranks"]["PSO"] == pytest.approx(3.0)

    def test_reports_problem_count(self):
        """num_problems documents how many blocks the test actually used."""
        outcome = ResultAnalyzer.friedman_test(self._three_algorithm_results())

        assert outcome["num_problems"] == 4

    def test_median_statistic(self):
        """Ranking on medians must be accepted and must change nothing here."""
        results = self._three_algorithm_results()
        by_median = ResultAnalyzer.friedman_test(results, statistic="median")

        assert by_median["mean_ranks"]["cNM"] == pytest.approx(1.0)

    def test_rejects_two_algorithms(self):
        """Friedman is undefined for two treatments; the guard must fire."""
        results = self._three_algorithm_results()
        del results["PSO"]

        with pytest.raises(ValueError, match="at least 3 algorithms"):
            ResultAnalyzer.friedman_test(results)

    def test_rejects_incomplete_blocks(self):
        """Algorithms must share a problem set or the ranking is meaningless."""
        results = self._three_algorithm_results()
        del results["DE"]["p4"]

        with pytest.raises(ValueError, match="same problem set"):
            ResultAnalyzer.friedman_test(results)

    def test_rejects_unknown_statistic(self):
        """A typo in `statistic` must fail loudly rather than silently mean-reduce."""
        with pytest.raises(ValueError, match="must be 'mean' or 'median'"):
            ResultAnalyzer.friedman_test(
                self._three_algorithm_results(), statistic="average"
            )


class TestMannWhitneyVsReference:
    """Pairwise comparisons producing the paper's Table 4 layout."""

    def _results_for_one_problem(self):
        """One problem, one clearly better reference and one clearly worse rival."""
        return {
            "cNM": make_results([1.0, 1.1, 1.2, 1.3, 1.4]),
            "DE": make_results([5.0, 5.1, 5.2, 5.3, 5.4]),
            "PSO": make_results([1.05, 1.15, 1.25, 1.35, 1.45]),
        }

    def test_matches_scipy(self):
        """Each cell must equal scipy's mannwhitneyu on the same two samples."""
        results = self._results_for_one_problem()
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(results, reference="cNM")

        reference = [r.best_fitness for r in results["cNM"]]
        for algorithm in ["DE", "PSO"]:
            other = [r.best_fitness for r in results[algorithm]]
            expected_u, expected_p = mannwhitneyu(
                reference, other, alternative="two-sided"
            )
            assert comparisons[algorithm]["statistic"] == pytest.approx(expected_u)
            assert comparisons[algorithm]["p_value"] == pytest.approx(expected_p)

    def test_excludes_the_reference(self):
        """The reference is not compared against itself."""
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(
            self._results_for_one_problem(), reference="cNM"
        )

        assert set(comparisons.keys()) == {"DE", "PSO"}

    def test_detects_a_clear_difference(self):
        """A well-separated pair must come out significant at alpha = 0.05."""
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(
            self._results_for_one_problem(), reference="cNM"
        )

        assert comparisons["DE"]["p_value"] < 0.05

    def test_reports_sample_sizes(self):
        """Sample sizes belong in the output so a reader can judge the power."""
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(
            self._results_for_one_problem(), reference="cNM"
        )

        assert comparisons["DE"]["n_reference"] == 5
        assert comparisons["DE"]["n_other"] == 5

    def test_identical_constant_samples(self):
        """The GTD case: every run of every algorithm hits exactly 0.

        scipy raises on two identical constant samples, but the correct scientific
        answer is 'no evidence of a difference', so this must return p = 1.
        """
        results = {
            "cNM": make_results([0.0] * 5),
            "DE": make_results([0.0] * 5),
        }
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(results, reference="cNM")

        assert comparisons["DE"]["p_value"] == 1.0

    def test_failed_runs_rank_worst_rather_than_being_dropped(self):
        """A crashed run must count against its algorithm, not vanish.

        Dropping non-finite runs would shrink the sample to that algorithm's lucky
        runs, which flatters an algorithm that crashes. Under the feasibility-aware
        comparison they are retained and ranked strictly worst.
        """
        results = {
            "cNM": make_results([1.0, 1.1, 1.2]),
            "DE": make_results([5.0, 5.1, np.inf]),
        }
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(results, reference="cNM")

        assert comparisons["DE"]["n_other"] == 3

    def test_raw_mode_still_drops_infinite_runs(self):
        """With feasibility_aware disabled the old drop-them behaviour is kept."""
        results = {
            "cNM": make_results([1.0, 1.1, 1.2]),
            "DE": make_results([5.0, 5.1, np.inf]),
        }
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(
            results, reference="cNM", feasibility_aware=False
        )

        assert comparisons["DE"]["n_other"] == 2

    def test_one_sided_alternative(self):
        """'less' asks the directional question the paper's discussion poses."""
        results = self._results_for_one_problem()
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(
            results, reference="cNM", alternative="less"
        )

        # cNM is strictly better (lower) than DE on every run
        assert comparisons["DE"]["p_value"] < 0.05

    def test_rejects_missing_reference(self):
        """A misspelled reference must fail rather than silently pick another."""
        with pytest.raises(ValueError, match="not found"):
            ResultAnalyzer.mannwhitney_vs_reference(
                self._results_for_one_problem(), reference="ABC"
            )

    def test_rejects_all_infinite_algorithm_in_raw_mode(self):
        """In raw mode an algorithm that failed every run cannot be rank-tested."""
        results = {
            "cNM": make_results([1.0, 1.1, 1.2]),
            "DE": make_results([np.inf, np.inf, np.inf]),
        }

        with pytest.raises(ValueError, match="no finite results"):
            ResultAnalyzer.mannwhitney_vs_reference(
                results, reference="cNM", feasibility_aware=False
            )

    def test_all_failed_algorithm_ranks_worst(self):
        """An algorithm that failed every run must lose, not be excluded."""
        results = {
            "cNM": make_results([1.0, 1.1, 1.2]),
            "DE": make_results([np.inf, np.inf, np.inf]),
        }
        comparisons = ResultAnalyzer.mannwhitney_vs_reference(
            results, reference="cNM", alternative="less"
        )

        assert comparisons["DE"]["p_value"] < 0.05
