"""Runs generated tests in an isolated workspace and reports structured results.

Two isolation modes:

local (default)
    A fresh temp directory plus a subprocess with a hard timeout. This contains
    accidents and stops infinite loops, but generated code still executes on the
    host interpreter with your user's permissions. Fine for code you trust.

docker (--docker)
    The same tests run inside a throwaway container with the network disabled
    and memory/PID caps. Generated code cannot reach the internet, cannot exhaust
    the host, and cannot see the host filesystem apart from one throwaway temp
    directory mounted at /work. On timeout the container is killed by name, so a
    runaway test cannot outlive the run.

Reports are written with plain relative filenames inside the working directory.
We deliberately avoid passing absolute paths to --cov-report, because on Windows
an absolute path begins with a drive letter like C:, and the colon collides with
pytest-cov's `type:path` report syntax, which aborts the whole run. Result
parsing uses pytest's built-in JUnit XML output (no third-party plugin) for
version and platform stability.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

SANDBOX_IMAGE = "testloop-sandbox"


@dataclass
class RunResult:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    collected: bool = True
    coverage: float = 0.0
    uncovered_lines: list[int] = field(default_factory=list)
    output: str = ""
    timed_out: bool = False

    @property
    def all_passed(self) -> bool:
        return self.collected and self.failed == 0 and self.errors == 0 and self.passed > 0


PYTEST_ARGS = [
    "test_target.py",
    "--cov=target", "--cov-report=json",   # writes coverage.json into cwd
    "--junit-xml=results.xml",             # built-in, no plugin needed
    "-p", "no:cacheprovider", "-q",
]


def _parse_junit(path: str) -> tuple[int, int, int, bool]:
    """Return (passed, failed, errors, collected) from a JUnit XML file."""
    root = ET.parse(path).getroot()
    # root is <testsuites> containing <testsuite>, or a bare <testsuite>.
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    tests = failures = errors = skipped = 0
    for s in suites:
        tests += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
    return tests - failures - errors - skipped, failures, errors, tests > 0


def _collect(workdir: str, output: str) -> RunResult:
    """Build a RunResult from the report files pytest left in workdir."""
    result = RunResult(output=output[-6000:])

    junit = os.path.join(workdir, "results.xml")
    if os.path.exists(junit):
        p, f_, err, collected = _parse_junit(junit)
        result.passed, result.failed, result.errors = p, f_, err
        result.collected = collected
    else:
        # pytest aborted before writing a report (config error, collection
        # failure). Surface it as an error instead of a silent 0/0/0.
        result.collected = False
        result.errors = 1

    cov_json = os.path.join(workdir, "coverage.json")
    if os.path.exists(cov_json):
        with open(cov_json) as f:
            cov = json.load(f)
        result.coverage = round(cov.get("totals", {}).get("percent_covered", 0.0), 1)
        files = cov.get("files", {})
        tgt = files.get("target.py") or next(iter(files.values()), {})
        result.uncovered_lines = tgt.get("missing_lines", [])

    return result


def _run_local(workdir: str, timeout: int) -> RunResult:
    cmd = [sys.executable, "-m", "pytest", *PYTEST_ARGS]
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        return RunResult(output=(e.stdout or "") + f"\n[timed out after {timeout}s]",
                         timed_out=True, collected=False)
    return _collect(workdir, output)


def _run_docker(workdir: str, timeout: int) -> RunResult:
    name = f"testloop_{uuid.uuid4().hex[:12]}"
    cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network=none",        # generated code gets no internet
        "--memory=512m",         # cannot exhaust host RAM
        "--pids-limit=128",      # cannot fork-bomb the host
        "-v", f"{workdir}:/work",
        "-w", "/work",
        SANDBOX_IMAGE,
        "python", "-m", "pytest", *PYTEST_ARGS,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        # Killing the docker CLI does not stop the container, so kill it by name.
        subprocess.run(["docker", "kill", name],
                       capture_output=True, text=True)
        return RunResult(output=(e.stdout or "") + f"\n[timed out after {timeout}s]",
                         timed_out=True, collected=False)
    except FileNotFoundError:
        return RunResult(
            output="[docker not found on PATH: install Docker or drop --docker]",
            collected=False, errors=1,
        )

    if "Unable to find image" in output or "No such image" in output:
        return RunResult(
            output=output[-6000:] + f"\n[missing image: build it with "
                                    f"`docker build -t {SANDBOX_IMAGE} .`]",
            collected=False, errors=1,
        )
    return _collect(workdir, output)


def run_tests(source_code: str, test_code: str, timeout: int = 60,
              use_docker: bool = False) -> RunResult:
    workdir = tempfile.mkdtemp(prefix="testloop_")
    try:
        with open(os.path.join(workdir, "target.py"), "w") as f:
            f.write(source_code)
        with open(os.path.join(workdir, "test_target.py"), "w") as f:
            f.write(test_code)
        return (_run_docker if use_docker else _run_local)(workdir, timeout)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
