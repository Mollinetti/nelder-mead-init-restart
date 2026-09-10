#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the MOPTA08 wrapper.

Tests split into two groups by necessity:

- Platform-dispatch, checksum and error-path tests run everywhere, including on a
  machine that has never downloaded the executable. These cover the Windows code
  paths, which is the point: the Windows selection logic has to be verified on a
  machine that is not Windows, so it is written to be testable by injection.
- Evaluation tests are skipped unless the executable for the host is present.

The reference value 303.2663811285 at x = 0.5 is what ties the platforms
together. The binary is deterministic, so a Windows machine that computes
something else is not running the same benchmark, and its results would not be
comparable with any published number.
"""

import platform
from pathlib import Path

import numpy as np
import pytest

from nelder_mead.benchmarks.mopta08 import (
    KNOWN_CHECKSUMS,
    MOPTA08,
    MOPTA08_DIM,
    MOPTA08_NUM_CONSTRAINTS,
    checksum,
    executable_name,
    is_available,
)
from nelder_mead.benchmarks.problem_suite import OptimizationProblem


REFERENCE_OBJECTIVE = 303.2663811285121369
REFERENCE_NUM_VIOLATED = 15

requires_binary = pytest.mark.skipif(
    not is_available(),
    reason="MOPTA08 executable not installed; run benchmarks/scripts/setup_mopta08.py",
)


class TestPlatformDispatch:
    """Executable selection must be right on platforms we cannot run tests on.

    These are pure functions of (system, machine) precisely so that the Windows
    and Linux paths can be verified from any host.
    """

    @pytest.mark.parametrize(
        "system,machine,expected",
        [
            ("Windows", "AMD64", "mopta08_amd64.exe"),
            ("Windows", "amd64", "mopta08_amd64.exe"),
            ("Windows", "x86_64", "mopta08_amd64.exe"),
            ("Linux", "x86_64", "mopta08_elf64.bin"),
            ("Linux", "amd64", "mopta08_elf64.bin"),
            ("Linux", "i686", "mopta08_elf32.bin"),
            ("Linux", "i386", "mopta08_elf32.bin"),
            ("Linux", "armv7l", "mopta08_armhf.bin"),
            ("Darwin", "arm64", "mopta08_macos_arm64.bin"),
            ("Darwin", "aarch64", "mopta08_macos_arm64.bin"),
        ],
    )
    def test_selects_the_right_executable(self, system, machine, expected):
        """Windows on x86-64 must resolve to the .exe, whatever the host is."""
        assert executable_name(system, machine) == expected

    def test_windows_reports_amd64_as_machine(self):
        """platform.machine() returns 'AMD64' on Windows, not 'x86_64'.

        Matching case-sensitively on 'x86_64' would fail on every Windows box,
        and would do so only once someone actually ran it there.
        """
        assert executable_name("Windows", "AMD64") == "mopta08_amd64.exe"

    def test_windows_on_arm_is_refused_with_guidance(self):
        """No ARM Windows build exists; the error must say what to do instead."""
        with pytest.raises(RuntimeError, match="WSL2"):
            executable_name("Windows", "ARM64")

    def test_unsupported_platform_is_refused(self):
        """An unknown platform must fail loudly rather than guess a filename."""
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            executable_name("Solaris", "sparc")

    def test_unsupported_macos_architecture_is_refused(self):
        """Only Apple Silicon has a published build."""
        with pytest.raises(RuntimeError, match="Only Apple Silicon"):
            executable_name("Darwin", "x86_64")

    def test_host_platform_resolves(self):
        """Whatever machine the suite runs on must map to some executable."""
        assert executable_name().startswith("mopta08_")


class TestChecksums:
    """A silently swapped benchmark binary would invalidate results invisibly."""

    def test_every_platform_has_a_recorded_checksum(self):
        """The three binaries we distribute must all be pinned."""
        assert set(KNOWN_CHECKSUMS) == {
            "mopta08_amd64.exe",
            "mopta08_elf64.bin",
            "mopta08_macos_arm64.bin",
        }

    def test_checksums_are_well_formed(self):
        """Each must be a 64-character hex SHA-256."""
        for name, digest in KNOWN_CHECKSUMS.items():
            assert len(digest) == 64, name
            assert all(c in "0123456789abcdef" for c in digest), name

    def test_checksum_helper_matches_hashlib(self, tmp_path):
        """The chunked reader must agree with a direct hash."""
        import hashlib

        payload = b"mopta08" * 1000
        target = tmp_path / "sample.bin"
        target.write_bytes(payload)

        assert checksum(target) == hashlib.sha256(payload).hexdigest()

    @requires_binary
    def test_installed_binary_matches_its_recorded_checksum(self):
        """The executable actually in use must be the one we pinned."""
        problem = MOPTA08(auto_download=False)
        name = problem.binary_path.name

        assert checksum(problem.binary_path) == KNOWN_CHECKSUMS[name]


class TestMissingBinary:
    """Failure modes must be actionable rather than a bare traceback."""

    def test_explicit_missing_path_raises(self, tmp_path):
        """A path that does not exist must be reported as such."""
        with pytest.raises(FileNotFoundError, match="not found"):
            MOPTA08(binary_path=tmp_path / "nope.bin")

    def test_missing_without_download_explains_how_to_fix(self, tmp_path):
        """The error must name the URL and the destination."""
        with pytest.raises(FileNotFoundError, match="auto_download"):
            MOPTA08(binary_dir=tmp_path, auto_download=False)

    def test_is_available_is_false_for_an_empty_directory(self, tmp_path):
        """Availability must be checkable without downloading anything."""
        assert is_available(binary_dir=tmp_path) is False


@requires_binary
class TestEvaluation:
    """Behaviour of the real executable, where it is installed."""

    @pytest.fixture(scope="class")
    def problem(self):
        """One instance shared across the class; evaluations are expensive."""
        return MOPTA08(auto_download=False)

    def test_problem_shape(self, problem):
        """124 variables, 68 inequality constraints, no equalities."""
        assert problem.dim == MOPTA08_DIM == 124
        assert problem.num_ineq_constraints == MOPTA08_NUM_CONSTRAINTS == 68
        assert problem.num_eq_constraints == 0

    def test_bounds_are_the_unit_box(self, problem):
        """MOPTA08 is defined on [0, 1]^124."""
        lower, upper = problem.bounds

        assert np.array_equal(lower, np.zeros(124))
        assert np.array_equal(upper, np.ones(124))

    def test_reproduces_the_reference_value(self, problem):
        """The cross-platform anchor: f(0.5) must match to 1e-6.

        This is the test that says a Windows machine is running the same
        benchmark as the machine the reference was computed on.
        """
        value = problem.objective(np.full(124, 0.5))

        assert value == pytest.approx(REFERENCE_OBJECTIVE, abs=1e-6)

    def test_reference_constraint_pattern(self, problem):
        """The same design must violate the same number of constraints."""
        constraints = problem.constraint_ineq(np.full(124, 0.5))

        assert len(constraints) == 68
        assert int(np.sum(constraints > 0)) == REFERENCE_NUM_VIOLATED

    def test_deterministic(self, problem):
        """Repeated evaluation of one design must give the same answer."""
        x = np.full(124, 0.25)
        first = problem.objective(x)
        problem._cache_key = None  # defeat the memo to force a real re-run
        second = problem.objective(x)

        assert first == second

    def test_memo_prevents_double_invocation(self, problem):
        """objective() and constraint_ineq() at one point must cost one run.

        At roughly 0.3 s per invocation, launching twice per candidate would
        double the cost of the entire experiment.
        """
        problem.reset_instrumentation()
        x = np.full(124, 0.3)

        problem.objective(x)
        problem.constraint_ineq(x)

        assert problem.eval_count == 1

    def test_distinct_points_each_cost_an_invocation(self, problem):
        """The memo must not serve a stale answer for a different design."""
        problem.reset_instrumentation()

        for value in (0.1, 0.2):
            problem.objective(np.full(124, value))

        assert problem.eval_count == 2

    def test_full_precision_survives_the_text_round_trip(self, problem):
        """Inputs pass through a text file; nearby designs must stay distinct.

        Writing the inputs with too few digits would quantize the search space
        invisibly, and a direct-search method taking small steps would then see a
        flat landscape.
        """
        base = np.full(124, 0.5)
        nudged = base.copy()
        nudged[0] += 1e-12

        assert problem.objective(base) != problem.objective(nudged)

    def test_reports_evaluation_cost(self, problem):
        """The expense is the reason this problem is in the suite."""
        problem.reset_instrumentation()
        problem.objective(np.full(124, 0.4))

        assert problem.eval_count == 1
        assert problem.mean_eval_time > 0.0

    def test_rejects_wrong_dimension(self, problem):
        """A wrong-length design must raise before launching the executable."""
        with pytest.raises(ValueError, match="Expected 124"):
            problem.objective(np.ones(100))

    def test_soft_constraint_value_matches_the_literature_formula(self, problem):
        """f + 10*sum(max(g,0)), for comparison with TuRBO and BAxUS numbers."""
        x = np.full(124, 0.5)
        objective = problem.objective(x)
        constraints = problem.constraint_ineq(x)
        expected = objective + 10.0 * np.sum(np.clip(constraints, 0.0, None))

        assert problem.soft_constraint_value(x) == pytest.approx(expected)

    def test_is_an_optimization_problem(self, problem):
        """BatchRunner requires the OptimizationProblem interface."""
        assert isinstance(problem, OptimizationProblem)

    def test_no_optimum_is_claimed(self, problem):
        """MOPTA08 has no known optimum and none must be invented."""
        assert problem.optimum_value is None


@requires_binary
class TestConcurrencySafety:
    """Parallel runs must not collide over the executable's fixed filenames."""

    def test_instances_get_separate_working_directories(self):
        """The binary reads input.txt and writes output.txt in its cwd.

        Two instances sharing a directory would overwrite each other's files, and
        the experiment runner evaluates many cells at once.
        """
        first = MOPTA08(auto_download=False)
        second = MOPTA08(auto_download=False)

        assert first.work_dir != second.work_dir

    def test_interleaved_instances_stay_correct(self):
        """Alternating between two instances must not mix their results."""
        first = MOPTA08(auto_download=False)
        second = MOPTA08(auto_download=False)

        a = first.objective(np.full(124, 0.5))
        b = second.objective(np.full(124, 0.25))
        a_again = first.objective(np.full(124, 0.5))

        assert a == a_again
        assert a != b

    def test_explicit_work_dir_is_used(self, tmp_path):
        """A caller pinning the directory must get that directory."""
        problem = MOPTA08(auto_download=False, work_dir=tmp_path)

        assert problem.work_dir == tmp_path

        problem.objective(np.full(124, 0.5))

        assert (tmp_path / "input.txt").exists()
        assert (tmp_path / "output.txt").exists()
