#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the journal benchmark suites.

Consumes the JSON written by run_journal_suite.py and produces:

  journal_scalability.png   performance and robustness against dimension, the
                            figure revision item 2 asks for
  journal_<suite>_convergence.png
                            median convergence with an interquartile band, in the
                            style of the paper's Figures 1-4

Usage:
    python benchmarks/scripts/make_journal_figures.py [suite ...]

With no arguments it draws every suite whose results file exists.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

# A colour-blind-safe qualitative palette, so the figures survive greyscale print
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def load(suite, quick=False):
    """Load one suite's results, or None if it has not been run."""
    path = RESULTS_DIR / f"journal_{suite}{'_quick' if quick else ''}_results.json"
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def style(index):
    """Return a (colour, marker) pair for the index-th algorithm."""
    return PALETTE[index % len(PALETTE)], MARKERS[index % len(MARKERS)]


def _log_dimension_axis(axis, dimensions):
    """Put a log x axis on the swept dimensions, labelling only those values.

    A log axis adds its own decade minor ticks, which on a range like 2..50
    collide with the explicit dimension labels and render as unreadable overlap.
    Suppressing the minor ticks leaves exactly the dimensions that were run.
    """
    axis.set_xscale("log")
    axis.set_xticks(dimensions)
    axis.set_xticklabels([str(d) for d in dimensions])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())


def plot_scalability(payload, output_path):
    """
    Draw performance and robustness against dimension, one column per family.

    The top row is the error to the known optimum on a log scale, which measures
    solution quality. The bottom row is the feasibility rate, which measures
    robustness. Reading them together is the point: an algorithm can look good on
    the first row purely by ignoring the constraints, and the second row exposes
    that.
    """
    families = {}
    for problem_name, meta in payload["problems"].items():
        family = problem_name.split("-")[0]
        families.setdefault(family, []).append((meta["dimension"], problem_name))
    for entries in families.values():
        entries.sort()

    names = sorted(families)
    algorithms = payload["algorithms"]

    figure, axes = plt.subplots(
        2, len(names), figsize=(4.0 * len(names), 7.0), squeeze=False
    )

    for column, family in enumerate(names):
        quality_axis = axes[0][column]
        robustness_axis = axes[1][column]
        entries = families[family]
        dimensions = [dim for dim, _ in entries]

        # A family whose optimum has no closed form is plotted as the raw mean
        # objective instead of as an error, since there is nothing to measure from.
        has_optimum = all(
            payload["problems"][name]["optimum_value"] is not None
            for _, name in entries
        )

        for index, algorithm in enumerate(algorithms):
            colour, marker = style(index)

            quality, feasibility = [], []
            for _, problem_name in entries:
                stats = payload["statistics"][problem_name][algorithm]
                optimum = payload["problems"][problem_name]["optimum_value"]
                mean = stats["mean"]
                if mean is None:
                    quality.append(np.nan)
                elif has_optimum:
                    # Floor the error so an exact hit stays plottable on a log axis
                    quality.append(max(abs(mean - optimum), 1e-12))
                else:
                    quality.append(mean)
                feasibility.append(stats["feasibility_rate"])

            quality_axis.plot(
                dimensions, quality, marker=marker, color=colour, label=algorithm,
                linewidth=1.4, markersize=5,
            )
            robustness_axis.plot(
                dimensions, feasibility, marker=marker, color=colour, label=algorithm,
                linewidth=1.4, markersize=5,
            )

        quality_axis.set_title(family, fontsize=11)
        # Only a strictly positive series can be log-scaled
        finite = [
            v for line in quality_axis.get_lines() for v in line.get_ydata()
            if np.isfinite(v)
        ]
        if finite and min(finite) > 0:
            quality_axis.set_yscale("log")
        _log_dimension_axis(quality_axis, dimensions)
        quality_axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
        quality_axis.set_ylabel(
            "error to optimum\n|mean $-$ $f^*$|" if has_optimum
            else "mean $f(x)$\n(no known $f^*$)",
            fontsize=9,
        )

        robustness_axis.set_ylim(-0.05, 1.05)
        _log_dimension_axis(robustness_axis, dimensions)
        robustness_axis.set_xlabel("dimension $n$")
        robustness_axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
        if column == 0:
            robustness_axis.set_ylabel("feasibility rate")

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(algorithms),
        frameon=False, bbox_to_anchor=(0.5, 1.0),
    )
    figure.suptitle(
        "Solution quality and robustness against problem dimension\n"
        f"{payload['num_runs']} runs, {payload['max_fes']:,} function evaluations",
        y=0.94, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")


def plot_convergence(payload, suite, output_path, max_panels=8):
    """
    Draw median convergence with an interquartile band, per problem.

    The band matters more than the line. The paper's existing figures plot means,
    which on these problems are dominated by the occasional run that never reaches
    the feasible region; the interquartile range shows the typical run instead.
    """
    problem_names = list(payload["problems"])[:max_panels]
    if not problem_names:
        return

    algorithms = payload["algorithms"]
    columns = min(2, len(problem_names))
    rows = int(np.ceil(len(problem_names) / columns))

    figure, axes = plt.subplots(
        rows, columns, figsize=(6.0 * columns, 3.6 * rows), squeeze=False
    )

    for panel, problem_name in enumerate(problem_names):
        axis = axes[panel // columns][panel % columns]

        for index, algorithm in enumerate(algorithms):
            colour, _ = style(index)
            runs = payload["statistics"][problem_name][algorithm]["runs"]
            histories = [r["convergence_history"] for r in runs if r["convergence_history"]]
            if not histories:
                continue

            # Interpolate every run onto a shared evaluation grid before averaging;
            # runs record at their own intervals and restart at different times.
            longest = max(h[-1][0] for h in histories)
            grid = np.linspace(1, longest, 100)
            curves = []
            for history in histories:
                fes = np.array([point[0] for point in history], dtype=float)
                value = np.array([point[1] for point in history], dtype=float)
                finite = np.isfinite(value)
                if finite.sum() < 2:
                    continue
                curves.append(np.interp(grid, fes[finite], value[finite]))
            if not curves:
                continue

            curves = np.array(curves)
            median = np.median(curves, axis=0)
            lower = np.percentile(curves, 25, axis=0)
            upper = np.percentile(curves, 75, axis=0)

            axis.plot(grid, median, color=colour, label=algorithm, linewidth=1.4)
            axis.fill_between(grid, lower, upper, color=colour, alpha=0.15, linewidth=0)

        axis.set_xscale("log")
        axis.set_title(
            f"{problem_name} "
            f"($n={payload['problems'][problem_name]['dimension']}$)",
            fontsize=10,
        )
        axis.set_xlabel("function evaluations")
        axis.set_ylabel("$f(x)$")
        axis.grid(True, which="both", alpha=0.25, linewidth=0.5)

    for spare in range(len(problem_names), rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(algorithms), frameon=False
    )
    figure.suptitle(
        f"Convergence on the {suite} suite: median and interquartile range "
        f"over {payload['num_runs']} runs",
        y=0.99, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")


def main():
    """Draw figures for the requested suites, or for every suite with results."""
    requested = sys.argv[1:] or ["rw", "gseries", "scalability", "blackbox"]

    drew_any = False
    for suite in requested:
        payload = load(suite) or load(suite, quick=True)
        if payload is None:
            print(f"No results for '{suite}'; skipping.")
            continue

        drew_any = True
        suffix = "" if load(suite) else "_quick"

        if suite == "scalability":
            plot_scalability(
                payload, RESULTS_DIR / f"journal_scalability{suffix}.png"
            )

        plot_convergence(
            payload, suite, RESULTS_DIR / f"journal_{suite}{suffix}_convergence.png"
        )

    if not drew_any:
        print("Nothing to draw. Run benchmarks/scripts/run_journal_suite.py first.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
