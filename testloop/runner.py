"""Runs generated tests in an isolated workspace and reports structured results.

v0 isolation model: a fresh temp directory plus a subprocess with a hard
timeout. This contains accidental damage (the generated tests import only the
target module) and stops infinite loops. It does NOT defend against actively
malicious code, since generated code runs on the host interpreter. The
hardening path is a Docker sandbox with no network and a read-only mount;
see README. Being explicit about this boundary is the point, not a gap.

Reports are written with plain relative filenames inside the temp workdir and
read back from there. We deliberately avoid passing absolute paths to
--cov-report, because on Windows an absolute path begins with a drive letter
like C:, and the colon collides with pytest-cov's `type:path` report syntax,
which aborts the whole run. Result parsing uses pytest's built-in JUnit XML
output (no third-party plugin) for maximum version and platform stability.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


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


def _parse_junit(path: str) -> tuple[int, int, int, bool]:
    """Return (passed, failed, errors, collected) from a JUnit XML file."""
    tree = ET.parse(path)
    root = tree.getroot()
    # root is <testsuites> containing <testsuite>, or a bare <testsuite>.
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    tests = failures = errors = skipped = 0
    for s in suites:
        tests += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
    passed = tests - failures - errors - skipped
    return passed, failures, errors, tests > 0


def run_tests(source_code: str, test_code: str, timeout: int = 60) -> RunResult:
    workdir = tempfile.mkdtemp(prefix="testloop_")
    try:
        with open(os.path.join(workdir, "target.py"), "w") as f:
            f.write(source_code)
        with open(os.path.join(workdir, "test_target.py"), "w") as f:
            f.write(test_code)

        # All output paths are plain relative names resolved against cwd=workdir.
        cmd = [
            sys.executable, "-m", "pytest", "test_target.py",
            "--cov=target", "--cov-report=json",   # writes coverage.json in cwd
            "--junit-xml=results.xml",              # built-in, no plugin needed
            "-p", "no:cacheprovider", "-q",
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            return RunResult(
                output=(e.stdout or "") + f"\n[timed out after {timeout}s]",
                timed_out=True, collected=False,
            )

        result = RunResult(output=output[-6000:])

        junit = os.path.join(workdir, "results.xml")
        if os.path.exists(junit):
            p, f_, err, collected = _parse_junit(junit)
            result.passed, result.failed, result.errors = p, f_, err
            result.collected = collected
        else:
            # pytest aborted before producing a report. Surface why (config
            # error, collection failure) instead of silently reporting 0/0/0.
            result.collected = False
            result.errors = 1

        cov_json = os.path.join(workdir, "coverage.json")
        if os.path.exists(cov_json):
            with open(cov_json) as f:
                cov = json.load(f)
            totals = cov.get("totals", {})
            result.coverage = round(totals.get("percent_covered", 0.0), 1)
            files = cov.get("files", {})
            tgt = files.get("target.py") or next(iter(files.values()), {})
            result.uncovered_lines = tgt.get("missing_lines", [])

        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)