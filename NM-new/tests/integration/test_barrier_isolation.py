#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for per-run barrier isolation in BatchRunner.

Barriers are stateful: AugmentedLagrangian carries the Lagrange multipliers
(lambda, mu), the penalty parameter rho, the bound penalty beta and the previous
violation across calls. Passing a single instance via ``barrier=`` therefore leaks
state from one seed into the next, which silently makes runs non-independent and
invalidates any statistical test computed over them.

These tests pin the ``barrier_factory`` path that builds a fresh barrier per run,
and document the leak that the shared-instance path still exhibits.
"""

import numpy as np

from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.benchmarks.constrained import PressureVessel
from nelder_mead.constraints.augmented_lagrangian import AugmentedLagrangian
from nelder_mead.testing.batch_runner import BatchRunner


def make_augmented_lagrangian(problem):
    """Build a fresh AugmentedLagrangian barrier for a problem."""
    return AugmentedLagrangian(
        bounds=problem.bounds,
        num_solutions=problem.dim + 1,
        m_eq=problem.num_eq_constraints,
        p_ineq=problem.num_ineq_constraints,
    )


class TestBarrierFactoryIsolation:
    """A fresh barrier per run is what makes repeated runs independent."""

    def test_repeated_batches_are_identical(self):
        """Two identical batches must produce identical results, run for run."""
        problem = PressureVessel()
        runner = BatchRunner(verbose=False)

        batches = []
        for _ in range(2):
            results = runner.run_experiment(
                algorithm_class=NelderMead,
                problem=problem,
                num_runs=3,
                seeds=[1, 2, 3],
                max_fes=600,
                barrier_factory=make_augmented_lagrangian,
            )
            batches.append([r.best_fitness for r in results])

        assert batches[0] == batches[1]

    def test_run_order_does_not_matter(self):
        """A seed's result must not depend on which seeds ran before it.

        This is the property the shared-instance path breaks: with one barrier
        reused across the batch, seed 3's run starts from multipliers left behind
        by seeds 1 and 2, so running seed 3 alone gives a different answer.
        """
        problem = PressureVessel()
        runner = BatchRunner(verbose=False)

        batch = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=3,
            seeds=[1, 2, 3],
            max_fes=600,
            barrier_factory=make_augmented_lagrangian,
        )

        alone = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=1,
            seeds=[3],
            max_fes=600,
            barrier_factory=make_augmented_lagrangian,
        )

        assert batch[2].best_fitness == alone[0].best_fitness

    def test_each_run_gets_a_distinct_barrier(self):
        """The factory must be called once per run, not once per batch."""
        problem = PressureVessel()
        built = []

        def counting_factory(prob):
            barrier = make_augmented_lagrangian(prob)
            built.append(barrier)
            return barrier

        BatchRunner(verbose=False).run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=3,
            seeds=[1, 2, 3],
            max_fes=300,
            barrier_factory=counting_factory,
        )

        assert len(built) == 3
        assert len({id(b) for b in built}) == 3


class TestSharedBarrierLeaks:
    """Documents why ``barrier=`` must not be used for multi-run experiments."""

    def test_shared_instance_is_reused_across_runs(self):
        """Every run in a batch receives the very same barrier object.

        Nothing resets it between seeds. The leak is currently latent only because
        NelderMead never calls ``barrier.update()``, so the multipliers happen not
        to move during a run; it becomes a real coupling the moment they do. The
        object identity is the defect, and that is what this pins.
        """
        problem = PressureVessel()
        barrier = make_augmented_lagrangian(problem)
        seen = []

        class RecordingNelderMead(NelderMead):
            def run(self):
                seen.append(self.barrier)
                super().run()

        BatchRunner(verbose=False).run_experiment(
            algorithm_class=RecordingNelderMead,
            problem=problem,
            num_runs=2,
            seeds=[1, 2],
            max_fes=300,
            barrier=barrier,
        )

        assert len(seen) == 2
        assert seen[0] is seen[1] is barrier

    def test_factory_overrides_shared_instance(self):
        """When both are given, the factory wins so the safe path is the default."""
        problem = PressureVessel()
        shared = make_augmented_lagrangian(problem)
        built = []

        def counting_factory(prob):
            barrier = make_augmented_lagrangian(prob)
            built.append(barrier)
            return barrier

        BatchRunner(verbose=False).run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=2,
            seeds=[1, 2],
            max_fes=300,
            barrier=shared,
            barrier_factory=counting_factory,
        )

        assert len(built) == 2
        assert all(b is not shared for b in built)
