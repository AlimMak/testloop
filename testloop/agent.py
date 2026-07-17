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
from .llm import LLM
from .runner import RunResult, run_tests

BUG_MARKER = "TESTLOOP_SOURCE_BUG:"


@dataclass
class LoopResult:
    tests: str
    result: RunResult
    iterations: int
    outcome: str                 # "success" | "bug_found" | "incomplete"
    input_tokens: int
    output_tokens: int
    bug_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome == "success"


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

    for i in range(1, max_iterations + 1):
        bug_reason = None
        if i == 1:
            on_event("act", i, "generating initial tests")
            tests = llm.complete(
                prompts.GENERATE_SYSTEM.format(module_dotted=module_dotted),
                prompts.GENERATE_USER.format(source=source, module_dotted=module_dotted),
            )
        else:
            on_event("act", i, "repairing tests")
            # When pytest could not collect any tests (e.g. ImportError in the
            # test file), prepend an explicit label so the model knows no tests
            # ran — it must fix the imports before worrying about assertions.
            repair_output = (
                "[COLLECTION ERROR — pytest could not import the test module. "
                "No tests ran. The import statements are wrong; fix them first.]\n"
                + result.output
            ) if result.collection_error else result.output
            raw = llm.complete(
                prompts.REPAIR_SYSTEM.format(module_dotted=module_dotted),
                prompts.REPAIR_USER.format(
                    source=source,
                    module_dotted=module_dotted,
                    tests=tests,
                    output=repair_output,
                    coverage=result.coverage,
                    target=coverage_target,
                    uncovered=result.uncovered_lines,
                ),
            )
            bug_reason, tests = _extract_bug(raw)

        result = run_tests(
            source, tests,
            timeout=timeout,
            use_docker=use_docker,
            module_dotted=module_dotted,
            package_files=package_files,
        )
        if result.collection_error:
            on_event("observe", i, "collection error (no tests ran — import failed)")
        else:
            on_event(
                "observe", i,
                f"{result.passed} passed, {result.failed} failed, "
                f"{result.coverage}% coverage",
            )

        # A declared bug only counts if a test actually fails to confirm it.
        if bug_reason and not result.all_passed:
            on_event("bug", i, bug_reason)
            return LoopResult(tests, result, i, "bug_found",
                              llm.input_tokens, llm.output_tokens, bug_reason)

        if result.all_passed and result.coverage >= coverage_target:
            on_event("done", i, "target reached")
            return LoopResult(tests, result, i, "success",
                              llm.input_tokens, llm.output_tokens)

    return LoopResult(tests, result, max_iterations, "incomplete",
                      llm.input_tokens, llm.output_tokens)
