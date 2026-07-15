"""Command line entry point.

    python -m testloop path/to/module.py --coverage 85 --max-iters 5
"""

from __future__ import annotations

import argparse
import os
import sys

from .agent import generate_tests
from .llm import LLM


def _print_event(kind: str, i: int, msg: str) -> None:
    tag = {"act": "->", "observe": "..", "done": "**", "bug": "!!"}.get(kind, "  ")
    print(f"  [iter {i}] {tag} {msg}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="testloop",
                                description="Closed-loop pytest generator.")
    p.add_argument("target", help="Python file to generate tests for")
    p.add_argument("--coverage", type=float, default=80.0,
                   help="coverage target percent (default 80)")
    p.add_argument("--max-iters", type=int, default=5,
                   help="max generate/repair iterations (default 5)")
    p.add_argument("--timeout", type=int, default=60,
                   help="per-run test timeout in seconds (default 60)")
    p.add_argument("-o", "--out", default=None,
                   help="where to write tests (default test_<name>.py)")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--docker", action="store_true",
                   help="run generated tests in an isolated container "
                        "(no network, capped memory/pids)")
    p.add_argument("--mock", action="store_true",
                   help="run without the API using scripted responses")
    args = p.parse_args(argv)

    if not os.path.isfile(args.target):
        print(f"error: no such file: {args.target}", file=sys.stderr)
        return 2

    with open(args.target) as f:
        source = f.read()

    out = args.out or f"test_{os.path.basename(args.target)}"

    sandbox = "docker" if args.docker else "local subprocess"
    print(f"testloop: {args.target}  (target {args.coverage}% cov, "
          f"max {args.max_iters} iters, sandbox: {sandbox})")
    llm = LLM(mock=args.mock, model=args.model)
    loop = generate_tests(
        source, llm,
        coverage_target=args.coverage,
        max_iterations=args.max_iters,
        timeout=args.timeout,
        use_docker=args.docker,
        on_event=_print_event,
    )

    with open(out, "w") as f:
        f.write(loop.tests)

    status = {"success": "SUCCESS", "bug_found": "BUG FOUND",
              "incomplete": "INCOMPLETE"}[loop.outcome]
    print(f"\n{status} after {loop.iterations} iteration(s)")
    if loop.outcome == "bug_found":
        print(f"  source bug: {loop.bug_reason}")
        print(f"  the failing test is kept in {out}, marked with # SOURCE BUG")
    print(f"  {loop.result.passed} passed, {loop.result.failed} failed, "
          f"{loop.result.coverage}% coverage")
    print(f"  tokens: {llm.input_tokens} in / {llm.output_tokens} out")
    print(f"  tests written to {out}")
    return {"success": 0, "bug_found": 2, "incomplete": 1}[loop.outcome]


if __name__ == "__main__":
    raise SystemExit(main())
