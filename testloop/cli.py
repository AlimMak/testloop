"""Command line entry point.

    python -m testloop path/to/module.py --coverage 85 --max-iters 5
    python -m testloop ./src --max-files 10 --budget-tokens 50000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import generate_tests
from .llm import LLM


def _print_event(kind: str, i: int, msg: str) -> None:
    tag = {"act": "->", "observe": "..", "done": "**", "bug": "!!"}.get(kind, "  ")
    print(f"  [iter {i}] {tag} {msg}")


# ─── Summary table ────────────────────────────────────────────────────────────

_COL_MODULE = 30
_COL_STATUS = 12
_COL_COV    = 10
_COL_ITERS  =  7
_COL_TOKENS =  9

_STATUS_LABELS = {
    "success":    "SUCCESS",
    "bug_found":  "BUG FOUND",
    "incomplete": "INCOMPLETE",
    "error":      "ERROR",
}


def _fmt_row(module: str, status: str, cov: str, iters: str, tokens: str) -> str:
    if len(module) > _COL_MODULE:
        module = "..." + module[-(_COL_MODULE - 3):]
    return (
        f"{module:<{_COL_MODULE}} "
        f"{status:<{_COL_STATUS}} "
        f"{cov:>{_COL_COV}} "
        f"{iters:>{_COL_ITERS}} "
        f"{tokens:>{_COL_TOKENS}}"
    )


def _print_summary(
    rows: list[dict],
    budget_hit: bool = False,
    skipped: int = 0,
) -> None:
    header = _fmt_row("Module", "Status", "Coverage", "Iters", "Tokens")
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    pass_count = 0
    total_iters = 0
    total_tokens = 0
    cov_values: list[float] = []

    for r in rows:
        status = _STATUS_LABELS.get(r["status"], r["status"].upper())
        cov    = f"{r['coverage']:.1f}%" if r.get("coverage") is not None else "-"
        iters  = str(r["iters"])         if r.get("iters")    is not None else "-"
        tokens = str(r["tokens"])        if r.get("tokens")   is not None else "-"
        print(_fmt_row(r["module"], status, cov, iters, tokens))
        if r["status"] == "success":
            pass_count += 1
        if r.get("iters") is not None:
            total_iters  += r["iters"]
        if r.get("tokens") is not None:
            total_tokens += r["tokens"]
        if r.get("coverage") is not None:
            cov_values.append(r["coverage"])

    print(sep)
    n = len(rows)
    avg_cov = f"{sum(cov_values) / len(cov_values):.1f}% avg" if cov_values else "-"
    print(_fmt_row("TOTAL", f"{pass_count}/{n} pass", avg_cov,
                   str(total_iters), str(total_tokens)))
    if budget_hit:
        print("\n  (budget hit -- run halted early)")
    if skipped:
        print(f"  ({skipped} module(s) skipped by --max-files)")


# ─── Directory mode ───────────────────────────────────────────────────────────

def _run_directory(root: Path, args: argparse.Namespace) -> int:
    from .discovery import collect_package_files, discover_modules, find_import_root

    root = root.resolve()

    # When root is itself a package (has __init__.py), step up to the first
    # ancestor that is NOT a package — that is the directory that must be on
    # sys.path so that `import mypkg.utils` resolves correctly.  Without this,
    # collect_package_files returns flat keys ("utils.py") instead of the
    # required package-relative keys ("mypkg/utils.py"), causing relative
    # imports inside the package to fail at collection time.
    import_root = find_import_root(root)

    # Discover all testable modules from the import root, then filter to those
    # that live under the directory the user actually requested.
    all_modules = [
        (d, p)
        for d, p in discover_modules(import_root)
        if p.is_relative_to(root)
    ]
    if not all_modules:
        print(f"error: no testable modules found under {root}", file=sys.stderr)
        return 2

    max_files = args.max_files or len(all_modules)
    modules = all_modules[:max_files]
    skipped = len(all_modules) - len(modules)

    print(
        f"testloop: {root}  ({len(modules)} module(s), "
        f"target {args.coverage}% cov, max {args.max_iters} iters)"
    )
    if skipped:
        print(f"  (--max-files {max_files}: {skipped} module(s) not scheduled)")

    # Collect all source files relative to the import root so the sandbox
    # workdir gets the full package tree (e.g. mypkg/__init__.py, mypkg/utils.py).
    pkg_files = collect_package_files(import_root)
    llm = LLM(mock=args.mock, model=args.model)

    # --out-dir overrides default placement; absent means "next to each source file".
    out_dir: Path | None = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    prev_in = 0
    prev_out = 0
    budget_hit = False

    for dotted, path in modules:
        print(f"\n  {dotted}")
        try:
            source = path.read_text(encoding="utf-8")
            loop = generate_tests(
                source, llm,
                coverage_target=args.coverage,
                max_iterations=args.max_iters,
                timeout=args.timeout,
                use_docker=args.docker,
                on_event=_print_event,
                module_dotted=dotted,
                package_files=pkg_files,
            )

            if out_dir is not None:
                out_file = out_dir / f"test_{dotted.replace('.', '_')}.py"
            else:
                out_file = path.parent / f"test_{path.stem}.py"
            out_file.write_text(loop.tests, encoding="utf-8")

            mod_in  = loop.input_tokens  - prev_in
            mod_out = loop.output_tokens - prev_out
            prev_in  = loop.input_tokens
            prev_out = loop.output_tokens

            rows.append({
                "module":   dotted,
                "status":   loop.outcome,
                "coverage": loop.result.coverage,
                "iters":    loop.iterations,
                "tokens":   mod_in + mod_out,
                "out_file": str(out_file),
            })

            if loop.outcome == "bug_found":
                print(f"  !! source bug: {loop.bug_reason}")
                print(f"     failing test kept in {out_file}")
        except Exception as exc:
            rows.append({
                "module":   dotted,
                "status":   "error",
                "coverage": None,
                "iters":    None,
                "tokens":   None,
                "error":    str(exc),
            })
            print(f"  [error] {exc}", file=sys.stderr)

        # Budget guard: check after every module.
        total_tokens = llm.input_tokens + llm.output_tokens
        if args.budget_tokens and total_tokens >= args.budget_tokens:
            print(f"\n  [budget] {total_tokens:,} tokens used -- halting")
            budget_hit = True
            break

    _print_summary(rows, budget_hit=budget_hit, skipped=skipped)

    any_bug  = any(r["status"] == "bug_found"  for r in rows)
    all_ok   = all(r["status"] == "success"    for r in rows)
    if any_bug:
        return 2
    if all_ok and not budget_hit:
        return 0
    return 1


# ─── Single-file mode (original behaviour) ────────────────────────────────────

def _run_single_file(target: Path, args: argparse.Namespace) -> int:
    source = target.read_text(encoding="utf-8")

    # Default: write next to the source file, not into cwd.
    out = Path(args.out) if args.out else target.parent / f"test_{target.name}"

    sandbox = "docker" if args.docker else "local subprocess"
    print(f"testloop: {target}  (target {args.coverage}% cov, "
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

    out.write_text(loop.tests, encoding="utf-8")

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


# ─── Entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="testloop",
                                description="Closed-loop pytest generator.")
    p.add_argument("target",
                   help="Python file to generate tests for, or a package directory")
    p.add_argument("--coverage", type=float, default=80.0,
                   help="coverage target percent (default 80)")
    p.add_argument("--max-iters", type=int, default=5,
                   help="max generate/repair iterations per module (default 5)")
    p.add_argument("--timeout", type=int, default=60,
                   help="per-run test timeout in seconds (default 60)")
    p.add_argument("-o", "--out", default=None,
                   help="single-file mode: output file path "
                        "(default: test_<name>.py next to the source file)")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="directory mode: write all generated tests into DIR "
                        "(default: next to each source module)")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--docker", action="store_true",
                   help="run generated tests in an isolated container "
                        "(no network, capped memory/pids)")
    p.add_argument("--mock", action="store_true",
                   help="run without the API using scripted responses")
    p.add_argument("--max-files", type=int, default=0, metavar="N",
                   help="directory mode: test at most N modules (0 = unlimited)")
    p.add_argument("--budget-tokens", type=int, default=0, metavar="N",
                   help="directory mode: halt when cumulative token usage exceeds N")
    args = p.parse_args(argv)

    target = Path(args.target)

    if target.is_dir():
        return _run_directory(target, args)

    if not target.is_file():
        print(f"error: no such file or directory: {args.target}", file=sys.stderr)
        return 2

    return _run_single_file(target, args)


if __name__ == "__main__":
    raise SystemExit(main())
