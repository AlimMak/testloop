"""Tests for testloop.agent — the generate/observe/repair loop.

All outcomes (success, bug_found, incomplete) are exercised without real API
calls or real test execution: run_tests is patched and LLM uses mock mode.
"""
from unittest.mock import patch

import pytest

import testloop.llm as llm_module
from testloop.agent import LoopResult, _extract_bug, generate_tests
from testloop.llm import LLM
from testloop.runner import RunResult


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_mock_queue():
    """Wipe _MOCK before and after every test to prevent cross-test bleed."""
    llm_module._MOCK.clear()
    yield
    llm_module._MOCK.clear()


def _llm(responses: list[str]) -> LLM:
    """Pre-load _MOCK with scripted responses, then create a mock-mode LLM.

    Responses must be non-empty so __init__ does not seed the demo.
    """
    assert responses, (
        "Provide at least one response so the demo is not seeded. "
        "Use a bare 'import target\\n' if the content does not matter."
    )
    llm_module._MOCK.extend(responses)
    return LLM(mock=True)


# Reusable run-result sentinels.
PASSING = RunResult(passed=3, failed=0, errors=0, collected=True, coverage=90.0)
FAILING = RunResult(passed=0, failed=1, errors=0, collected=True, coverage=30.0)

SOURCE = "def add(a, b): return a + b"
STUB_TESTS = "import target\ndef test_add(): assert target.add(1, 2) == 3\n"


# ─── Outcome: success ─────────────────────────────────────────────────────────

def test_success_on_first_iteration():
    llm = _llm([STUB_TESTS])
    with patch("testloop.agent.run_tests", return_value=PASSING):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "success"
    assert loop.iterations == 1
    assert loop.result.passed == 3
    assert loop.result.coverage == 90.0


def test_success_after_one_repair_round():
    """Iteration 1 fails; iteration 2 passes — outcome is success at iter 2."""
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[FAILING, PASSING]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "success"
    assert loop.iterations == 2


def test_success_respects_coverage_target():
    """run_tests result that passes all tests but misses coverage is not success."""
    low_cov = RunResult(passed=3, failed=0, errors=0, collected=True, coverage=50.0)
    high_cov = RunResult(passed=3, failed=0, errors=0, collected=True, coverage=90.0)
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[low_cov, high_cov]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "success"
    assert loop.iterations == 2


# ─── Outcome: bug_found ───────────────────────────────────────────────────────

def test_bug_found_via_source_bug_marker():
    """TESTLOOP_SOURCE_BUG marker on a repair response triggers bug_found."""
    bug_reply = f"TESTLOOP_SOURCE_BUG: add() is broken for negatives\n{STUB_TESTS}"
    llm = _llm([STUB_TESTS, bug_reply])
    with patch("testloop.agent.run_tests", side_effect=[FAILING, FAILING]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "bug_found"
    assert loop.iterations == 2
    assert loop.bug_reason == "add() is broken for negatives"


def test_bug_reason_extracted_accurately():
    """The reason string after the colon is stored verbatim (stripped)."""
    bug_reply = "TESTLOOP_SOURCE_BUG:   off-by-one in edge case  \n" + STUB_TESTS
    llm = _llm([STUB_TESTS, bug_reply])
    with patch("testloop.agent.run_tests", side_effect=[FAILING, FAILING]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.bug_reason == "off-by-one in edge case"


def test_bug_marker_ignored_when_confirming_test_passes():
    """A SOURCE_BUG marker only counts if the run still fails.

    If the marked test unexpectedly passes, the loop continues normally.
    """
    bug_reply = f"TESTLOOP_SOURCE_BUG: spurious claim\n{STUB_TESTS}"
    llm = _llm([STUB_TESTS, bug_reply])
    # Second run passes despite the marker — loop should become success, not bug_found.
    with patch("testloop.agent.run_tests", side_effect=[FAILING, PASSING]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "success"
    assert loop.bug_reason is None


# ─── Outcome: incomplete ──────────────────────────────────────────────────────

def test_incomplete_when_max_iterations_exhausted():
    llm = _llm([STUB_TESTS])  # _MOCK drains after iter 1; default used for rest
    with patch("testloop.agent.run_tests", return_value=FAILING):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=3)
    assert loop.outcome == "incomplete"
    assert loop.iterations == 3


def test_run_tests_called_exactly_once_per_iteration():
    """run_tests is invoked exactly max_iterations times and no more."""
    llm = _llm([STUB_TESTS])
    call_count = 0

    def counting_run(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return FAILING

    with patch("testloop.agent.run_tests", side_effect=counting_run):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=4)

    assert loop.iterations == 4
    assert call_count == 4


# ─── LoopResult fields ────────────────────────────────────────────────────────

def test_loop_result_succeeded_property_true_on_success():
    llm = _llm([STUB_TESTS])
    with patch("testloop.agent.run_tests", return_value=PASSING):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.succeeded is True


def test_loop_result_succeeded_property_false_on_incomplete():
    llm = _llm([STUB_TESTS])
    with patch("testloop.agent.run_tests", return_value=FAILING):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=1)
    assert loop.succeeded is False


def test_on_event_callback_receives_done_event():
    """The 'done' event is fired when the loop exits successfully."""
    events = []
    llm = _llm([STUB_TESTS])
    with patch("testloop.agent.run_tests", return_value=PASSING):
        generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5,
                       on_event=lambda kind, i, msg: events.append(kind))
    assert "done" in events


def test_on_event_callback_receives_bug_event():
    bug_reply = f"TESTLOOP_SOURCE_BUG: broken\n{STUB_TESTS}"
    events = []
    llm = _llm([STUB_TESTS, bug_reply])
    with patch("testloop.agent.run_tests", side_effect=[FAILING, FAILING]):
        generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5,
                       on_event=lambda kind, i, msg: events.append(kind))
    assert "bug" in events


# ─── _extract_bug — unit tests ────────────────────────────────────────────────

def test_extract_bug_explicit_marker():
    text = "TESTLOOP_SOURCE_BUG: off by one\nimport target\ndef test(): pass\n"
    reason, code = _extract_bug(text)
    assert reason == "off by one"
    assert code == "import target\ndef test(): pass"


def test_extract_bug_marker_strips_reason_whitespace():
    text = "TESTLOOP_SOURCE_BUG:   extra spaces   \nimport target\n"
    reason, code = _extract_bug(text)
    assert reason == "extra spaces"


def test_extract_bug_marker_case_insensitive():
    """The marker check uppercases the line, so lower-case input still matches."""
    text = "testloop_source_bug: lower case\nimport target\n"
    reason, code = _extract_bug(text)
    assert reason == "lower case"


def test_extract_bug_leading_blank_lines_skipped():
    """Empty lines before the marker are ignored."""
    text = "\n\nTESTLOOP_SOURCE_BUG: blank preamble\nimport target\n"
    reason, code = _extract_bug(text)
    assert reason == "blank preamble"


def test_extract_bug_comment_fallback():
    """'# SOURCE BUG' anywhere in the code triggers the comment-fallback path."""
    text = "import target\n# SOURCE BUG: result is wrong\nassert foo() == 1\n"
    reason, code = _extract_bug(text)
    assert reason == "source behavior appears incorrect (see # SOURCE BUG comment)"
    assert code == text  # original text returned unchanged


def test_extract_bug_no_marker_returns_none():
    text = "import target\ndef test(): pass\n"
    reason, code = _extract_bug(text)
    assert reason is None
    assert code == text


def test_extract_bug_non_marker_first_line_does_not_match():
    """A non-empty first line that is not the marker stops the search immediately."""
    text = "import target\nTESTLOOP_SOURCE_BUG: too late\n"
    reason, code = _extract_bug(text)
    # The marker appears on the second real line — it must NOT be recognised.
    assert reason is None
