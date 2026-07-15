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
) -> LoopResult:
    tests = ""
    result = RunResult()

    for i in range(1, max_iterations + 1):
        bug_reason = None
        if i == 1:
            on_event("act", i, "generating initial tests")
            tests = llm.complete(
                prompts.GENERATE_SYSTEM,
                prompts.GENERATE_USER.format(source=source),
            )
        else:
            on_event("act", i, "repairing tests")
            raw = llm.complete(
                prompts.REPAIR_SYSTEM,
                prompts.REPAIR_USER.format(
                    source=source,
                    tests=tests,
                    output=result.output,
                    coverage=result.coverage,
                    target=coverage_target,
                    uncovered=result.uncovered_lines,
                ),
            )
            bug_reason, tests = _extract_bug(raw)

        result = run_tests(source, tests, timeout=timeout, use_docker=use_docker)
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
