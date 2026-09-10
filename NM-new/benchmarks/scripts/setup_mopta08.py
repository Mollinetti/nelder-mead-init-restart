#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install and verify the MOPTA08 executable for this machine.

Run this once on a new machine before including MOPTA08 in an experiment. It
detects the platform, downloads the matching executable, checks its SHA-256,
runs it on a reference design, and confirms the answer against a value computed
independently on another platform. Every step is checked, because a benchmark
binary that runs but returns subtly different numbers would invalidate a paper's
results with no other visible symptom.

Usage:
    python benchmarks/scripts/setup_mopta08.py [--no-download] [--dir PATH]
                                               [--timing N]

    --no-download  Verify an executable that is already in place; do not fetch
    --dir PATH     Directory holding the executable (default benchmarks/mopta08)
    --timing N     Time N evaluations to estimate the cost of an experiment
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nelder_mead.benchmarks.mopta08 import (  # noqa: E402
    KNOWN_CHECKSUMS,
    MOPTA08,
    checksum,
    download_executable,
    executable_name,
)

#: MOPTA08 at the centre of its box, computed with the macOS arm64 executable on
#: 2026-08-31. The binary is deterministic and platform-independent to at least
#: this precision, so a mismatch means the wrong executable or a broken download.
REFERENCE_INPUT = 0.5
REFERENCE_OBJECTIVE = 303.2663811285121369
REFERENCE_NUM_VIOLATED = 15
REFERENCE_TOLERANCE = 1e-6


def main():
    """Detect, install and verify the MOPTA08 executable."""
    parser = argparse.ArgumentParser(
        description="Install and verify the MOPTA08 executable."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Verify an executable already in place; do not fetch one",
    )
    parser.add_argument(
        "--dir", type=Path, default=None, help="Directory holding the executable"
    )
    parser.add_argument(
        "--timing",
        type=int,
        default=5,
        help="Time this many evaluations to estimate experiment cost (0 to skip)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MOPTA08 setup")
    print("=" * 70)
    print(f"  system       : {platform.system()} {platform.release()}")
    print(f"  machine      : {platform.machine()}")
    print(f"  python       : {platform.python_version()}")

    # --- 1. platform detection -------------------------------------------------
    try:
        name = executable_name()
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print(f"  executable   : {name}")

    # --- 2. download -----------------------------------------------------------
    try:
        if args.no_download:
            directory = args.dir or (REPO_ROOT / "benchmarks" / "mopta08")
            path = Path(directory) / name
            if not path.exists():
                print(f"\nFAILED: {path} does not exist and --no-download was given.")
                return 1
        else:
            path = download_executable(name, args.dir)
    except RuntimeError as exc:
        print(f"\nFAILED: {exc}")
        return 1

    size_mb = path.stat().st_size / (1 << 20)
    print(f"  path         : {path}")
    print(f"  size         : {size_mb:.1f} MB")

    # --- 3. checksum -----------------------------------------------------------
    print("\nVerifying checksum ...")
    actual = checksum(path)
    expected = KNOWN_CHECKSUMS.get(name)
    if expected is None:
        print(f"  no recorded checksum for {name}; got {actual}")
    elif actual != expected:
        print(f"  FAILED\n    expected {expected}\n    actual   {actual}")
        return 1
    else:
        print(f"  OK  {actual}")

    # --- 4. execute ------------------------------------------------------------
    print("\nRunning the executable on the reference design ...")
    try:
        problem = MOPTA08(binary_path=path)
        x = np.full(problem.dim, REFERENCE_INPUT)
        started = time.perf_counter()
        objective = problem.objective(x)
        constraints = problem.constraint_ineq(x)
        elapsed = time.perf_counter() - started
    except Exception as exc:
        print(f"  FAILED: {exc}")
        if platform.system() == "Windows":
            print(
                "\n  On Windows, check that antivirus or SmartScreen has not "
                "quarantined the executable.\n"
                "  Right-click the file, Properties, and tick 'Unblock' if the "
                "option is shown."
            )
        return 1

    print(f"  dimension    : {problem.dim} (expected 124)")
    print(f"  constraints  : {len(constraints)} (expected 68)")
    print(f"  wall time    : {elapsed * 1000:.0f} ms")

    # --- 5. cross-platform agreement ------------------------------------------
    print("\nChecking the result against the reference value ...")
    print(f"  f(0.5 * 1)   : {objective:.10f}")
    print(f"  reference    : {REFERENCE_OBJECTIVE:.10f}")
    difference = abs(objective - REFERENCE_OBJECTIVE)
    violated = int(np.sum(constraints > 0))

    problems_found = []
    if problem.dim != 124 or len(constraints) != 68:
        problems_found.append("wrong problem shape")
    if difference > REFERENCE_TOLERANCE:
        problems_found.append(f"objective differs by {difference:.3e}")
    if violated != REFERENCE_NUM_VIOLATED:
        problems_found.append(
            f"{violated} constraints violated, expected {REFERENCE_NUM_VIOLATED}"
        )

    if problems_found:
        print("  FAILED: " + "; ".join(problems_found))
        print(
            "\n  This executable does not reproduce the reference result. Do not "
            "use it for experiments: results would not be comparable with those "
            "produced elsewhere."
        )
        return 1

    print(f"  difference   : {difference:.3e}  OK")
    print(f"  violated     : {violated} of 68  OK")

    # --- 6. cost estimate ------------------------------------------------------
    if args.timing > 0:
        print(f"\nTiming {args.timing} evaluations ...")
        rng = np.random.default_rng(0)
        timings = []
        for _ in range(args.timing):
            point = rng.random(problem.dim)
            started = time.perf_counter()
            problem.objective(point)
            timings.append(time.perf_counter() - started)
        median = float(np.median(timings))
        print(f"  median       : {median * 1000:.0f} ms per evaluation")
        print(f"  1 run of 2,000 evaluations : {median * 2000 / 60:.1f} min")
        for algorithms, runs in ((6, 30), (6, 10)):
            total_hours = median * 2000 * algorithms * runs / 3600
            print(
                f"  {algorithms} algorithms x {runs:2d} runs   : "
                f"{total_hours:6.1f} core-hours "
                f"({total_hours / 8:.1f} h wall on 8 cores)"
            )

    print("\n" + "=" * 70)
    print("MOPTA08 is installed and verified.")
    print("Run it with:")
    print("  python benchmarks/scripts/run_journal_suite.py blackbox --quick")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
