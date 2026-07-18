"""The closed loop: generate -> run -> observe -> repair, until green + covered.

This is the perceive/decide/act/observe loop applied to test writing. Each turn
the agent acts (writes or repairs tests), observes (runs them, reads failures
and coverage), and decides whether to stop or feed the observation back in.

The agent distinguishes two kinds of stuck: a test it cannot make pass because
the test is wrong (keep repairing), versus a test that fails because the SOURCE
under test is genuinely buggy. In the second case it stops and reports the bug
as a finding rather than contorting the test to make wrong behavior pass. The
model signals this by leading its reply with a TESTLOOP_SOURCE_BUG marker (or by
annotating the offending test with a `# SOURCE BUG` comment).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .llm import LLM, _strip_fences
from .runner import RunResult, run_tests

BUG_MARKER = "TESTLOOP_SOURCE_BUG:"


@dataclass
class LoopResult:
    tests: str
    result: RunResult
    iterations: int
    outcome: str                 # "success" | "bug_found" | "incomplete" | "regressed"
    input_tokens: int
    output_tokens: int
    bug_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome == "success"


def _suite_score(result: RunResult) -> tuple:
    """Return a comparison key for a run result — higher means a better suite.

    Priority order matches the user-facing scoring spec:
    (a) no collection error  (b) coverage  (c) passing tests  (d) fewer failures
    """
    return (
        not result.collection_error,
        result.coverage,
        result.passed,
        -result.failed,
    )


def _extract_bug(text: str) -> tuple[str | None, str]:
    """Return (bug_reason, test_code). bug_reason is None when no bug is declared.

    Recognises either an explicit leading `TESTLOOP_SOURCE_BUG: ...` marker or,
    as a fallback, a `# SOURCE BUG` annotation somewhere in the returned tests.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith(BUG_MARKER):
            reason = stripped[len(BUG_MARKER):].strip()
            return reason, "\n".join(lines[i + 1:]).strip()
        break  # first real line was not the marker
    if "# SOURCE BUG" in text.upper():
        return "source behavior appears incorrect (see # SOURCE BUG comment)", text
    return None, text


def generate_tests(
    source: str,
    llm: LLM,
    coverage_target: float = 80.0,
    max_iterations: int = 5,
    timeout: int = 60,
    use_docker: bool = False,
    on_event=lambda *_: None,
    module_dotted: str = "target",
    package_files: dict[str, str] | None = None,
) -> LoopResult:
    """Run the closed generate/repair loop for a single module.

    Parameters
    ----------
    source:
        The source code of the module under test (used in prompts).
    llm:
        LLM wrapper to use for generation and repair calls.
    coverage_target:
        Minimum coverage percentage required for a "success" outcome.
    max_iterations:
        Maximum number of generate/repair iterations before giving up.
    timeout:
        Per-run pytest timeout in seconds.
    use_docker:
        Run tests in an isolated Docker container.
    on_event:
        Callback ``(kind, iteration, message)`` for progress reporting.
    module_dotted:
        Importable dotted name of the module (e.g. ``"target"`` or
        ``"mypkg.utils"``).  Passed to prompts so the model generates
        correct import statements, and to the runner for coverage flags.
    package_files:
        When testing a module that is part of a package, pass the full
        ``{relative_posix_path: source_text}`` mapping returned by
        :func:`testloop.discovery.collect_package_files`.  ``None`` keeps the
        original single-file behaviour.
    """
    tests = ""
    result = RunResult()
    # Regression tracking: keep the last known-good state so we can revert to it
    # and re-prompt when the model deletes tests instead of fixing them.
    prev_tests: str = ""
    prev_result: RunResult = RunResult()
    prev_test_count: int = 0
    consecutive_regressions: int = 0
    regression_note: str | None = None
    # Best-suite tracking: keep the highest-scoring (tests, result) pair seen so
    # far so that INCOMPLETE / REGRESSED returns report peak performance, not the
    # final (possibly broken) iteration.  None until the first non-regressing,
    # non-collection-error run.
    best_tests: str = ""
    best_result: RunResult | None = None
    # 0%-coverage streak: counts consecutive non-collection-error iterations where
    # tests pass but coverage is 0%.  This indicates the module is uninstrumentable
    # (e.g. a C extension); bail out after 2 such iterations to avoid wasting budget.
    zero_cov_streak: int = 0

    for i in range(1, max_iterations + 1):
        bug_reason = None
        import_contract = prompts._import_contract(module_dotted)

        if i == 1:
            on_event("act", i, "generating initial tests")
            tests = llm.complete(
                prompts.GENERATE_SYSTEM.format(import_contract=import_contract),
                prompts.GENERATE_USER.format(source=source, module_dotted=module_dotted),
            )
        else:
            on_event("act", i, "repairing tests")
            # Determine what output and context to pass to the repair prompt.
            # Priority: regression note > collection error > normal output.
            # On regression we show the previous (non-regressed) run's output so
            # the context is consistent with the reverted test file.
            if regression_note:
                repair_ctx = prev_result
                repair_output = regression_note + "\n" + prev_result.output
            elif result.collection_error:
                repair_ctx = result
                repair_output = (
                    "[COLLECTION ERROR — pytest could not import the test module. "
                    "No tests ran. The import statements are wrong; fix them first.]\n"
                    + result.output
                )
            else:
                repair_ctx = result
                repair_output = result.output

            raw = llm.complete(
                prompts.REPAIR_SYSTEM.format(import_contract=import_contract),
                prompts.REPAIR_USER.format(
                    source=source,
                    module_dotted=module_dotted,
                    tests=tests,
                    output=repair_output,
                    coverage=repair_ctx.coverage,
                    target=coverage_target,
                    uncovered=repair_ctx.uncovered_lines,
                ),
            )
            bug_reason, tests = _extract_bug(raw)
            tests = _strip_fences(tests)  # mirror the generate path

        result = run_tests(
            source, tests,
            timeout=timeout,
            use_docker=use_docker,
            module_dotted=module_dotted,
            package_files=package_files,
        )

        if result.collection_error:
            on_event("observe", i, "collection error (no tests ran — import failed)")
            # Surface the actual ImportError / pytest output so the user can see
            # what went wrong rather than just the summary line above.
            if result.output.strip():
                on_event("output", i, result.output)
            # A collection error is never a deletion regression: the model's
            # imports are broken, not its test count.  Skip the regression guard
            # and leave prev_test_count / prev_tests unchanged so the last
            # known-good state is preserved for potential future reverts.
        else:
            on_event(
                "observe", i,
                f"{result.passed} passed, {result.failed} failed, "
                f"{result.coverage}% coverage",
            )

            # Regression guard: repair produced a suite with fewer tests than the
            # previous iteration.  The model deleted failing tests instead of fixing
            # them.  Revert to the last known-good tests and re-prompt with an
            # explicit rejection note.  Two consecutive regressions → "regressed".
            if i > 1 and result.collected and result.total < prev_test_count:
                removed = prev_test_count - result.total
                consecutive_regressions += 1
                on_event(
                    "regress", i,
                    f"{removed} test(s) deleted — reverting to previous suite",
                )
                regression_note = (
                    f"[REGRESSION — your last submission had {result.total} test(s), "
                    f"down from {prev_test_count}. Tests were removed to avoid failures. "
                    f"That is not allowed. The suite shown below is restored to its state "
                    f"before your last change. Fix the {prev_result.failed} failing "
                    f"test(s) without removing or weakening any assertion.]"
                )
                tests = prev_tests  # revert
                if consecutive_regressions >= 2:
                    _bt = best_tests if best_result is not None else prev_tests
                    _br = best_result if best_result is not None else prev_result
                    return LoopResult(_bt, _br, i, "regressed",
                                      llm.input_tokens, llm.output_tokens)
                continue  # skip success/bug checks — retry with regression note

            # Non-regressing iteration: update state and best-suite tracking.
            consecutive_regressions = 0
            regression_note = None
            prev_tests = tests
            prev_test_count = result.total
            prev_result = result
            if best_result is None or _suite_score(result) > _suite_score(best_result):
                best_tests = tests
                best_result = result

            # 0%-coverage bail-out: tests pass but nothing is instrumented.
            # Happens with C-extension modules where pytest-cov can never report
            # coverage.  Two consecutive zero-coverage passing runs → stop early.
            if result.coverage == 0.0 and result.passed > 0:
                zero_cov_streak += 1
                if zero_cov_streak >= 2:
                    on_event("observe", i,
                             "0% coverage on 2 consecutive runs — "
                             "module likely not instrumentable, stopping early")
                    return LoopResult(best_tests, best_result, i, "incomplete",
                                      llm.input_tokens, llm.output_tokens)
            else:
                zero_cov_streak = 0

        # A source bug requires real test execution: collection must have succeeded
        # and at least one test must have run and failed.  A collection error or
        # timeout (both set collected=False) means no test assertions were
        # evaluated, so we cannot conclude anything about the source.
        if bug_reason and result.collected and not result.all_passed:
            on_event("bug", i, bug_reason)
            return LoopResult(tests, result, i, "bug_found",
                              llm.input_tokens, llm.output_tokens, bug_reason)

        if result.all_passed and result.coverage >= coverage_target:
            on_event("done", i, "target reached")
            return LoopResult(tests, result, i, "success",
                              llm.input_tokens, llm.output_tokens)

    # Return the best suite seen across all iterations, not the last one.
    # If every iteration had a collection error (best_result is None), fall back
    # to the last generated tests so the caller has something to inspect.
    final_tests = best_tests if best_result is not None else tests
    final_result = best_result if best_result is not None else result
    return LoopResult(final_tests, final_result, max_iterations, "incomplete",
                      llm.input_tokens, llm.output_tokens)
