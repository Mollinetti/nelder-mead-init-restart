#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOPTA08: the automotive mass-optimization benchmark.

MOPTA08 is the standard reference problem for expensive, high-dimensional
constrained optimization: 124 design variables describing a vehicle, one objective
(mass), and 68 performance constraints derived from crash and durability
simulations. It is the case the introduction of the c-NM paper is really arguing
about, and unlike every other benchmark in this package it is not an algebraic
expression at all. It is a compiled binary that this module drives as a
subprocess, so the objective is a true black box in the strict sense: no closed
form, no derivatives, and no way to inspect it other than by evaluating it.

The benchmark is distributed only as platform-specific executables, which is why
it is wrapped rather than reimplemented. Executables exist for 64-bit Windows,
64-bit Linux, 32-bit Linux, 32-bit ARM Linux and Apple Silicon; this module picks
the right one for the host automatically and can download it on first use.

Evaluation protocol, as defined by the binary:

1. Write ``input.txt`` in the working directory: 124 values, one per line.
2. Run the executable with that directory as its working directory.
3. Read ``output.txt``: 69 values, one per line. The first is the objective, the
   remaining 68 are the constraints in the ``g(x) <= 0`` convention.

Each evaluation costs roughly 0.3 s, almost all of it process startup, since the
binary is tens of megabytes and is loaded afresh every call. That is the point:
it makes an evaluation budget a real resource and puts this problem in a different
regime from every algebraic benchmark here.

Note on the aggregation. The widely used BAxUS and TuRBO wrappers fold the
constraints into the objective as ``f(x) + 10 * sum(max(g, 0))`` and return a
single number, because those methods are unconstrained optimizers. This wrapper
deliberately does not: it returns the objective and the 68 raw constraints
separately, so that the barrier and feasibility machinery in this package handles
them the same way it handles every other constrained problem, and so that
feasibility can be reported rather than hidden inside a penalty. Results here are
therefore not directly comparable with numbers reported under that soft-constraint
aggregation.

References:
    Jones, D. R. (2008). Large-scale multi-disciplinary mass optimization in the
    auto industry. MOPTA 2008 Conference, Ontario, Canada.

    Eriksson, D., & Poloczek, M. (2021). Scalable constrained Bayesian
    optimization. AISTATS 2021. (Appendix E.7 describes the soft-constraint form.)

    Papenmeier, L., Nardi, L., & Poloczek, M. (2022). Increasing the scope as you
    learn: adaptive Bayesian optimization in nested subspaces. NeurIPS 2022.
    Executables mirrored at https://mopta.papenmeier.io/
"""

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .problem_suite import OptimizationProblem


#: Problem size, fixed by the binary.
MOPTA08_DIM = 124
MOPTA08_NUM_CONSTRAINTS = 68

#: Where executables are kept by default. Outside the package tree because they
#: are 30-90 MB and must not end up in the distribution.
DEFAULT_BINARY_DIR = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "mopta08"
)

#: Mirror hosting the executables.
DOWNLOAD_BASE_URL = "https://mopta.papenmeier.io"

#: SHA-256 of each executable, verified against the mirror on 2026-08-31. A
#: benchmark binary silently swapped underneath a paper would invalidate its
#: results with no other visible symptom, so downloads are checked.
KNOWN_CHECKSUMS = {
    "mopta08_amd64.exe": (
        "dffab0f371747c55cb386c580e4980ce822db9d113d711359c7fb0856da39ecd"
    ),
    "mopta08_elf64.bin": (
        "2ad78a256980134c2a84debd4ff15a53534c812f48175577f9155cdd22d6ffb5"
    ),
    "mopta08_macos_arm64.bin": (
        "7499786d2514af2256fa18489e657e7cae84d1909dc0710e6ae240f31a50efac"
    ),
}


def executable_name(system: Optional[str] = None, machine: Optional[str] = None) -> str:
    """
    Return the executable filename for a platform.

    Args:
        system: Platform name as returned by ``platform.system()``; defaults to
                the host
        machine: Architecture as returned by ``platform.machine()``; defaults to
                 the host

    Returns:
        The filename of the matching MOPTA08 executable

    Raises:
        RuntimeError: If no executable is published for that platform

    Example:
        >>> executable_name("Windows", "AMD64")
        'mopta08_amd64.exe'
        >>> executable_name("Linux", "x86_64")
        'mopta08_elf64.bin'
        >>> executable_name("Darwin", "arm64")
        'mopta08_macos_arm64.bin'
    """
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()

    if system == "windows":
        if machine in ("amd64", "x86_64"):
            return "mopta08_amd64.exe"
        raise RuntimeError(
            f"No MOPTA08 executable is published for Windows on '{machine}'. "
            "Only 64-bit x86 Windows is supported. On Windows for ARM, run the "
            "Linux build under WSL2 instead."
        )

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "mopta08_elf64.bin"
        if machine in ("i386", "i686"):
            return "mopta08_elf32.bin"
        if machine.startswith("arm") or machine == "armv7l":
            return "mopta08_armhf.bin"
        raise RuntimeError(
            f"No MOPTA08 executable is published for Linux on '{machine}'."
        )

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "mopta08_macos_arm64.bin"
        raise RuntimeError(
            f"No MOPTA08 executable is published for macOS on '{machine}'. "
            "Only Apple Silicon is supported."
        )

    raise RuntimeError(f"Unsupported platform '{system}'.")


def checksum(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_executable(
    name: Optional[str] = None,
    directory: Optional[Path] = None,
    verify: bool = True,
) -> Path:
    """
    Download the MOPTA08 executable for this platform, if it is not already there.

    Args:
        name: Executable filename; defaults to the one matching the host
        directory: Destination directory; defaults to ``benchmarks/mopta08``
        verify: If True, check the SHA-256 against the recorded value and refuse a
                mismatch

    Returns:
        Path to the executable

    Raises:
        RuntimeError: If the download fails or the checksum does not match

    Example:
        >>> path = download_executable()   # doctest: +SKIP
    """
    name = name or executable_name()
    directory = Path(directory) if directory else DEFAULT_BINARY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name

    if target.exists():
        if verify and name in KNOWN_CHECKSUMS:
            actual = checksum(target)
            if actual != KNOWN_CHECKSUMS[name]:
                raise RuntimeError(
                    f"{target} does not match the recorded checksum.\n"
                    f"  expected {KNOWN_CHECKSUMS[name]}\n"
                    f"  actual   {actual}\n"
                    "Delete the file to re-download, or pass verify=False if you "
                    "intend to use a different build."
                )
        return target

    url = f"{DOWNLOAD_BASE_URL}/{name}"
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading MOPTA08 executable {name} from {url} ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, partial)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the MOPTA08 executable from {url}: {exc}\n"
            f"Download it manually and place it at {target}."
        ) from exc

    if verify and name in KNOWN_CHECKSUMS:
        actual = checksum(partial)
        if actual != KNOWN_CHECKSUMS[name]:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded {name} does not match the recorded checksum.\n"
                f"  expected {KNOWN_CHECKSUMS[name]}\n"
                f"  actual   {actual}"
            )

    partial.replace(target)

    # The Windows loader uses the extension; POSIX needs the execute bit.
    if os.name != "nt":
        target.chmod(target.stat().st_mode | 0o755)

    print(f"Saved to {target}", file=sys.stderr)
    return target


class MOPTA08(OptimizationProblem):
    """
    MOPTA08 vehicle mass minimization, evaluated by the distributed binary.

    Mathematical formulation:
        Minimize:  f(x)                                    [vehicle mass]
        Subject to: g_i(x) <= 0,  i = 1..68                [performance limits]
        Bounds:     0 <= x_j <= 1,  j = 1..124

    Both f and g are computed by the MOPTA08 executable. There is no closed form.

    Args:
        binary_path: Path to the executable. If None, looks for the one matching
                     this platform in `binary_dir`.
        binary_dir: Directory to look in; defaults to ``benchmarks/mopta08``
        auto_download: If True, fetch the executable when it is missing
        verify_checksum: If True, check the executable against its recorded SHA-256
        work_dir: Directory for the input/output files. If None, a private
                  temporary directory is created and removed with the object.
                  Each instance must have its own, because the binary reads and
                  writes fixed filenames in its working directory; two instances
                  sharing one directory would race.
        timeout: Seconds to allow a single evaluation before giving up

    Properties:
        - Dimension: 124
        - Number of inequality constraints: 68
        - Number of equality constraints: 0
        - No known optimum. The literature treats roughly 222 as a strong result
          under the soft-constraint aggregation; this wrapper reports the raw
          objective and constraints instead, so that figure is not directly
          comparable.
        - Expensive: roughly 0.3 s per evaluation, almost all process startup

    Attributes:
        eval_count (int): Number of executable invocations
        eval_time (float): Cumulative wall-clock seconds spent in the executable

    Raises:
        FileNotFoundError: If the executable cannot be found and auto_download is
                           False

    Example:
        >>> problem = MOPTA08()                       # doctest: +SKIP
        >>> f = problem.objective(np.full(124, 0.5))  # doctest: +SKIP
        >>> round(f, 3)                               # doctest: +SKIP
        303.266
    """

    def __init__(
        self,
        binary_path: Optional[Path] = None,
        binary_dir: Optional[Path] = None,
        auto_download: bool = True,
        verify_checksum: bool = True,
        work_dir: Optional[Path] = None,
        timeout: float = 120.0,
    ):
        self.binary_path = self._resolve_binary(
            binary_path, binary_dir, auto_download, verify_checksum
        )
        self.timeout = timeout

        # A private working directory per instance. The binary reads input.txt and
        # writes output.txt by fixed name in its cwd, so two instances sharing a
        # directory would overwrite each other's files.
        self._temp_dir = None
        if work_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="mopta08_")
            self.work_dir = Path(self._temp_dir.name)
        else:
            self.work_dir = Path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)

        self.eval_count = 0
        self.eval_time = 0.0

        # Single-entry memo: BaseAlgorithm calls objective() and constraint_ineq()
        # separately for the same point, which would otherwise double the number of
        # subprocess launches, and launching is essentially the whole cost here.
        self._cache_key: Optional[bytes] = None
        self._cache_value: Optional[Tuple[float, np.ndarray]] = None

        super().__init__(
            name="MOPTA08",
            dim=MOPTA08_DIM,
            bounds=(np.zeros(MOPTA08_DIM), np.ones(MOPTA08_DIM)),
            num_eq_constraints=0,
            num_ineq_constraints=MOPTA08_NUM_CONSTRAINTS,
        )

    @staticmethod
    def _resolve_binary(binary_path, binary_dir, auto_download, verify_checksum):
        """Locate the executable, downloading it if allowed."""
        if binary_path is not None:
            path = Path(binary_path)
            if not path.exists():
                raise FileNotFoundError(f"MOPTA08 executable not found at {path}")
            return path

        name = executable_name()
        directory = Path(binary_dir) if binary_dir else DEFAULT_BINARY_DIR
        candidate = directory / name

        if candidate.exists():
            if verify_checksum and name in KNOWN_CHECKSUMS:
                actual = checksum(candidate)
                if actual != KNOWN_CHECKSUMS[name]:
                    raise RuntimeError(
                        f"{candidate} does not match the recorded checksum.\n"
                        f"  expected {KNOWN_CHECKSUMS[name]}\n"
                        f"  actual   {actual}"
                    )
            if os.name != "nt" and not os.access(candidate, os.X_OK):
                candidate.chmod(candidate.stat().st_mode | 0o755)
            return candidate

        if auto_download:
            return download_executable(name, directory, verify=verify_checksum)

        raise FileNotFoundError(
            f"MOPTA08 executable not found at {candidate}.\n"
            f"Download it from {DOWNLOAD_BASE_URL}/{name} and place it there, or "
            "construct MOPTA08(auto_download=True)."
        )

    def _run_binary(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate one design by invoking the executable.

        Args:
            x: Design vector of length 124

        Returns:
            Tuple of (objective, array of 68 constraint values)

        Raises:
            RuntimeError: If the executable fails, times out, or writes output
                          that does not have the expected shape
        """
        key = np.ascontiguousarray(x, dtype=float).tobytes()
        if key == self._cache_key:
            return self._cache_value

        input_file = self.work_dir / "input.txt"
        output_file = self.work_dir / "output.txt"

        # Remove any stale output so a silent failure cannot be read as a result
        output_file.unlink(missing_ok=True)

        # repr() of a Python float is the shortest string that round-trips, so no
        # precision is lost. It must be a Python float, not a numpy scalar: under
        # numpy 2 the latter reprs as "np.float64(0.5)", which the executable's
        # Fortran list-directed read rejects.
        input_file.write_text(
            "".join(f"{float(value)!r}\n" for value in np.asarray(x, dtype=float))
        )

        # On Windows, suppress the console window that would otherwise flash for
        # every one of the many thousands of invocations in an experiment.
        creation_flags = 0
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [str(self.binary_path)],
                cwd=str(self.work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"MOPTA08 executable timed out after {self.timeout} s"
            ) from exc
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                f"MOPTA08 executable failed with exit code {completed.returncode}. "
                f"stderr: {completed.stderr.decode('utf-8', 'replace')[:500]}"
            )

        if not output_file.exists():
            raise RuntimeError(
                f"MOPTA08 executable produced no output.txt in {self.work_dir}."
            )

        values = np.array(
            [float(line) for line in output_file.read_text().split("\n") if line.strip()]
        )

        expected = 1 + MOPTA08_NUM_CONSTRAINTS
        if values.size != expected:
            raise RuntimeError(
                f"MOPTA08 executable wrote {values.size} values, expected {expected} "
                f"(1 objective + {MOPTA08_NUM_CONSTRAINTS} constraints)."
            )

        self.eval_count += 1
        self.eval_time += elapsed

        self._cache_key = key
        self._cache_value = (float(values[0]), values[1:])
        return self._cache_value

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the vehicle mass."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        mass, _ = self._run_binary(x)
        return mass

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the 68 performance constraints (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        _, constraints = self._run_binary(x)
        return constraints

    def soft_constraint_value(self, x: np.ndarray, penalty: float = 10.0) -> float:
        """
        Return the aggregated value used by the Bayesian optimization literature.

        TuRBO and BAxUS report ``f(x) + 10 * sum(max(g(x), 0))`` because they treat
        MOPTA08 as an unconstrained problem. This method exists only so results
        here can be placed on the same scale as those papers; the optimization
        itself uses the objective and constraints separately.

        Args:
            x: Design vector
            penalty: Multiplier on the total violation, 10 in the literature

        Returns:
            The soft-constrained objective value

        Example:
            >>> problem = MOPTA08()                                # doctest: +SKIP
            >>> problem.soft_constraint_value(np.full(124, 0.5))   # doctest: +SKIP
        """
        mass, constraints = self._run_binary(np.asarray(x, dtype=float))
        return mass + penalty * float(np.sum(np.clip(constraints, 0.0, None)))

    def reset_instrumentation(self) -> None:
        """Zero the evaluation counter and timer, e.g. between benchmark runs."""
        self.eval_count = 0
        self.eval_time = 0.0

    @property
    def mean_eval_time(self) -> float:
        """Mean wall-clock seconds per evaluation, 0.0 if none were run."""
        if self.eval_count == 0:
            return 0.0
        return self.eval_time / self.eval_count

    def __del__(self):
        """Clean up the private temporary directory."""
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir is not None:
            try:
                temp_dir.cleanup()
            except Exception:
                pass


def is_available(binary_dir: Optional[Path] = None) -> bool:
    """
    Report whether a MOPTA08 executable is present for this platform.

    Lets callers include MOPTA08 when it is set up and skip it otherwise, without
    triggering a download or raising.

    Args:
        binary_dir: Directory to look in; defaults to ``benchmarks/mopta08``

    Returns:
        True if the executable for this platform exists

    Example:
        >>> isinstance(is_available(), bool)
        True
    """
    try:
        name = executable_name()
    except RuntimeError:
        return False

    directory = Path(binary_dir) if binary_dir else DEFAULT_BINARY_DIR
    return (directory / name).exists()
