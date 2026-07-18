"""Tests for testloop.agent — the generate/observe/repair loop.

All outcomes (success, bug_found, incomplete) are exercised without real API
calls or real test execution: run_tests is patched and LLM uses mock mode.
"""
from unittest.mock import patch

import pytest

import testloop.llm as llm_module
from testloop.agent import LoopResult, _extract_bug, _suite_score, generate_tests
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


# ─── Collection error handling ────────────────────────────────────────────────

# A RunResult where pytest failed before any tests were collected.
COLLECTION_ERR = RunResult(passed=0, failed=0, errors=1, collected=False, timed_out=False)


def test_collection_error_observe_message():
    """Collection-error observe event says 'collection error', not '0 passed ...'."""
    events: list[tuple[str, str]] = []
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[COLLECTION_ERR, PASSING]):
        generate_tests(
            SOURCE, llm,
            coverage_target=80.0, max_iterations=5,
            on_event=lambda kind, i, msg: events.append((kind, msg)),
        )
    first_observe = next(msg for kind, msg in events if kind == "observe")
    assert "collection error" in first_observe
    # The normal observe format is "N passed, N failed, N% coverage".
    # A collection error must not produce that format.
    assert "passed," not in first_observe  # "passed," only appears in the count format
    assert "coverage" not in first_observe


def test_collection_error_loop_continues():
    """A collection error alone must not end the loop; the next iteration can succeed."""
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[COLLECTION_ERR, PASSING]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "success"
    assert loop.iterations == 2


def test_collection_error_prefix_in_repair_prompt():
    """[COLLECTION ERROR ...] prefix is injected into the repair user prompt."""
    captured: list[str] = []

    def spy_complete(system: str, user: str) -> str:
        captured.append(user)
        return STUB_TESTS

    llm = _llm([STUB_TESTS])  # prevents _DEMO seeding; actual calls go to spy_complete
    with patch("testloop.agent.run_tests", side_effect=[COLLECTION_ERR, PASSING]):
        with patch.object(llm, "complete", side_effect=spy_complete):
            generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)

    # captured[0] = generate prompt (iteration 1), captured[1] = repair prompt (iteration 2)
    assert len(captured) >= 2
    assert "[COLLECTION ERROR" in captured[1]


def test_repair_system_prompt_contains_dotted_module_name():
    """REPAIR_SYSTEM sent to the LLM must contain the full dotted module name."""
    captured_systems: list[str] = []

    def spy_complete(system: str, user: str) -> str:
        captured_systems.append(system)
        return STUB_TESTS

    llm = _llm([STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[FAILING, PASSING]):
        with patch.object(llm, "complete", side_effect=spy_complete):
            generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5,
                           module_dotted="mypkg.billing")

    # captured_systems[0] = generate system, captured_systems[1] = repair system
    assert len(captured_systems) >= 2
    assert "mypkg.billing" in captured_systems[1]


def test_bug_found_not_triggered_on_collection_error():
    """A SOURCE_BUG marker returned during a collection-error run must not yield bug_found.

    No test assertions ran — the import was broken — so the loop cannot conclude
    that the source is buggy.  It must keep repairing instead.
    """
    bug_reply = f"TESTLOOP_SOURCE_BUG: source is broken\n{STUB_TESTS}"
    llm = _llm([STUB_TESTS, bug_reply])
    # Every run returns a collection error: the marker can never be confirmed.
    with patch("testloop.agent.run_tests", return_value=COLLECTION_ERR):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=3)
    assert loop.outcome != "bug_found"


# ─── Regression guard ────────────────────────────────────────────────────────

# Sentinels for regression tests.
# INIT_FAIL: first run — 28 tests (24 passing, 4 failing), below coverage target.
INIT_FAIL = RunResult(passed=24, failed=4, errors=0, collected=True, coverage=70.0)
# REGRESSED: model deleted the 4 failing tests instead of fixing them.
REGRESSED = RunResult(passed=14, failed=0, errors=0, collected=True, coverage=90.0)
# REPAIRED: model fixed the 4 failing tests; full suite of 28 now passes.
REPAIRED  = RunResult(passed=28, failed=0, errors=0, collected=True, coverage=90.0)
# COLL_ERR_COLLECTED: pytest wrote tests="1" errors="1" — collection error where
# collected=True but no test actually executed.  Must NOT trigger regression guard.
COLL_ERR_COLLECTED = RunResult(passed=0, failed=0, errors=1, collected=True, timed_out=False)


def test_regression_not_treated_as_success():
    """A repaired suite with fewer tests than before must not produce 'success',
    even if all remaining tests pass and coverage meets the target."""
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[INIT_FAIL, REGRESSED, REPAIRED]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    # Regression at iter 2 is rejected; success only arrives at iter 3.
    assert loop.outcome == "success"
    assert loop.iterations == 3


def test_two_consecutive_regressions_return_regressed():
    """Two consecutive regressions halt the loop with outcome 'regressed'."""
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[INIT_FAIL, REGRESSED, REGRESSED]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "regressed"
    assert loop.iterations == 3


def test_regression_note_injected_into_next_repair_prompt():
    """After a regression the next repair user-prompt contains [REGRESSION ...]."""
    captured_users: list[str] = []

    def spy_complete(system: str, user: str) -> str:
        captured_users.append(user)
        return STUB_TESTS

    llm = _llm([STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[INIT_FAIL, REGRESSED, REPAIRED]):
        with patch.object(llm, "complete", side_effect=spy_complete):
            generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)

    # captured_users: [generate(1), repair(2), repair-with-regression-note(3)]
    assert len(captured_users) == 3
    assert "[REGRESSION" in captured_users[2]


def test_regression_resets_on_non_regressing_iteration():
    """A regression followed by a correct repair does not leave the loop stuck."""
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS, STUB_TESTS])
    # iter 1: fail, iter 2: regress, iter 3: regress again → "regressed"
    # But here iter 3 fixes it → should succeed at iter 3.
    with patch("testloop.agent.run_tests",
               side_effect=[INIT_FAIL, REGRESSED, REPAIRED]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "success"


def test_collection_error_with_collected_true_does_not_trigger_regression():
    """A collection error where pytest wrote collected=True must not fire the
    regression guard.

    pytest sometimes records a collection error as a test item (tests='1'
    errors='1'), yielding collected=True but passed=0, failed=0.  The result
    has total=1, which is far below any previous test count, but this is an
    import failure — NOT the model deleting tests.  The loop must continue
    without a regression event.
    """
    events: list[tuple[str, str]] = []
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS])
    # iter 1: 28 tests; iter 2: collection error (collected=True); iter 3: repair OK
    with patch("testloop.agent.run_tests",
               side_effect=[INIT_FAIL, COLL_ERR_COLLECTED, REPAIRED]):
        loop = generate_tests(
            SOURCE, llm,
            coverage_target=80.0, max_iterations=5,
            on_event=lambda kind, i, msg: events.append((kind, msg)),
        )
    # Collection error must not produce a "regress" event.
    regress_events = [msg for kind, msg in events if kind == "regress"]
    assert regress_events == [], f"unexpected regression events: {regress_events}"
    # The loop should recover and succeed at iter 3.
    assert loop.outcome == "success"
    assert loop.iterations == 3


def test_collection_error_preserves_prev_test_count():
    """After a collection error (collected=True), prev_test_count is unchanged,
    so a genuine regression on the following iteration is still detected."""
    events: list[tuple[str, str]] = []
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS, STUB_TESTS])
    # iter 1: 28 tests; iter 2: collection error; iter 3: 14 tests (regression)
    with patch("testloop.agent.run_tests",
               side_effect=[INIT_FAIL, COLL_ERR_COLLECTED, REGRESSED, REPAIRED]):
        loop = generate_tests(
            SOURCE, llm,
            coverage_target=80.0, max_iterations=5,
            on_event=lambda kind, i, msg: events.append((kind, msg)),
        )
    # A regression at iter 3 (14 < 28) must still be detected.
    regress_events = [msg for kind, msg in events if kind == "regress"]
    assert len(regress_events) == 1
    assert loop.outcome == "success"  # regression reverted; iter 4 succeeds


# ─── Best-suite tracking ──────────────────────────────────────────────────────

# Sentinel: a genuinely good run — high coverage, many passing tests.
GOOD_RUN = RunResult(passed=40, failed=3, errors=0, collected=True, coverage=85.3)


def test_incomplete_returns_best_suite_not_last():
    """INCOMPLETE must report the best iteration, not the final broken one.

    Scenario: iter 1 fails; iter 2 has 40/3/85.3%; iter 3 hits a collection
    error.  The loop exhausts max_iterations and exits INCOMPLETE.  The result
    must reflect iter 2, not iter 3.
    """
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests",
               side_effect=[FAILING, GOOD_RUN, COLL_ERR_COLLECTED]):
        loop = generate_tests(SOURCE, llm, coverage_target=90.0, max_iterations=3)
    assert loop.outcome == "incomplete"
    assert loop.result.passed == 40
    assert loop.result.failed == 3
    assert loop.result.coverage == 85.3
    # Tests written to disk must also be from the best iteration.
    # (_strip_fences strips trailing whitespace, so compare without it.)
    assert loop.tests == STUB_TESTS.strip()


def test_best_suite_is_updated_when_later_iteration_improves():
    """If a later non-regressing iteration scores higher it becomes the new best."""
    better = RunResult(passed=50, failed=0, errors=0, collected=True, coverage=92.0)
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests",
               side_effect=[GOOD_RUN, better, COLL_ERR_COLLECTED]):
        loop = generate_tests(SOURCE, llm, coverage_target=95.0, max_iterations=3)
    assert loop.outcome == "incomplete"
    assert loop.result.coverage == 92.0
    assert loop.result.passed == 50


def test_incomplete_all_collection_errors_returns_last_tests():
    """When every iteration has a collection error, fall back to the last tests."""
    events: list[tuple[str, str]] = []
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", return_value=COLLECTION_ERR):
        loop = generate_tests(
            SOURCE, llm, coverage_target=80.0, max_iterations=2,
            on_event=lambda kind, i, msg: events.append((kind, msg)),
        )
    # No crash; outcome is incomplete with last-run values as fallback.
    assert loop.outcome == "incomplete"
    # Every observe event should say "collection error".
    observe_events = [msg for kind, msg in events if kind == "observe"]
    assert all("collection error" in msg for msg in observe_events)


def test_regressed_returns_best_suite():
    """Two consecutive regressions return the best-seen suite, not prev."""
    llm = _llm([STUB_TESTS, STUB_TESTS, STUB_TESTS])
    # iter 1: INIT_FAIL (70%), iter 2: REGRESSED (guard fires), iter 3: REGRESSED again
    with patch("testloop.agent.run_tests",
               side_effect=[INIT_FAIL, REGRESSED, REGRESSED]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "regressed"
    # Best was INIT_FAIL (iter 1); REGRESSED iterations never update best.
    assert loop.result.passed == 24
    assert loop.result.coverage == 70.0


# ─── _suite_score — unit tests ────────────────────────────────────────────────

def test_suite_score_no_collection_error_beats_collection_error():
    good  = RunResult(passed=1, failed=0, errors=0, collected=True, coverage=50.0)
    bad   = RunResult(passed=0, failed=0, errors=1, collected=True, coverage=0.0)
    assert _suite_score(good) > _suite_score(bad)


def test_suite_score_higher_coverage_wins():
    low  = RunResult(passed=10, failed=0, errors=0, collected=True, coverage=60.0)
    high = RunResult(passed=5,  failed=0, errors=0, collected=True, coverage=80.0)
    assert _suite_score(high) > _suite_score(low)


def test_suite_score_more_passing_breaks_coverage_tie():
    a = RunResult(passed=10, failed=2, errors=0, collected=True, coverage=80.0)
    b = RunResult(passed=15, failed=0, errors=0, collected=True, coverage=80.0)
    assert _suite_score(b) > _suite_score(a)


def test_suite_score_fewer_failures_breaks_pass_tie():
    a = RunResult(passed=10, failed=5, errors=0, collected=True, coverage=80.0)
    b = RunResult(passed=10, failed=2, errors=0, collected=True, coverage=80.0)
    assert _suite_score(b) > _suite_score(a)


# ─── Fence stripping ──────────────────────────────────────────────────────────

def test_fenced_repair_response_is_stripped():
    """A repair response wrapped in ```python ... ``` fences must produce
    unfenced tests; otherwise pytest gets a SyntaxError: invalid syntax on
    line 1 of the generated file."""
    fenced = "```python\nimport target\ndef test_foo():\n    assert True\n```"
    llm = _llm([STUB_TESTS, fenced])
    with patch("testloop.agent.run_tests", side_effect=[FAILING, PASSING]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert not loop.tests.startswith("```"), (
        "fence not stripped from repair response — tests would cause SyntaxError"
    )
    assert loop.outcome == "success"


# ─── Collection error output visibility ───────────────────────────────────────

def test_collection_error_emits_output_event_when_output_present():
    """An "output" event is fired when pytest produced non-empty output."""
    err_with_output = RunResult(
        passed=0, failed=0, errors=1, collected=False,
        output="ImportError: No module named 'natsort.__main__'",
    )
    events: list[tuple[str, str]] = []
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[err_with_output, PASSING]):
        generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5,
                       on_event=lambda kind, i, msg: events.append((kind, msg)))
    output_events = [(kind, msg) for kind, msg in events if kind == "output"]
    assert output_events, "expected at least one 'output' event"
    assert "ImportError" in output_events[0][1]


def test_collection_error_no_output_event_when_empty():
    """No "output" event is emitted when pytest output is empty."""
    err_no_output = RunResult(passed=0, failed=0, errors=1, collected=False, output="")
    events: list[tuple[str, str]] = []
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[err_no_output, PASSING]):
        generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5,
                       on_event=lambda kind, i, msg: events.append((kind, msg)))
    output_events = [kind for kind, _ in events if kind == "output"]
    assert output_events == []


# ─── 0%-coverage bail-out ─────────────────────────────────────────────────────

def test_zero_coverage_bail_out_after_two_passing_zero_cov_runs():
    """Two consecutive runs where tests pass but coverage is 0% must exit early
    as INCOMPLETE.  This prevents burning the whole iteration budget on modules
    that can never be instrumented (e.g. C extensions)."""
    zero_cov = RunResult(passed=5, failed=0, errors=0, collected=True, coverage=0.0)
    llm = _llm([STUB_TESTS] * 5)
    with patch("testloop.agent.run_tests", return_value=zero_cov):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "incomplete"
    assert loop.iterations == 2  # bailed out after 2nd zero-cov passing run


def test_zero_coverage_does_not_bail_when_tests_failing():
    """0% coverage with failing tests is the normal early-iteration state.
    The loop must continue so the model can fix the failures."""
    zero_cov_fail = RunResult(passed=0, failed=3, errors=0, collected=True, coverage=0.0)
    llm = _llm([STUB_TESTS, STUB_TESTS])
    with patch("testloop.agent.run_tests", side_effect=[zero_cov_fail, PASSING]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "success"


def test_zero_coverage_streak_resets_on_nonzero_coverage():
    """A non-zero coverage run resets the streak so a later zero-cov pair
    is required to trigger the bail-out again."""
    zero_cov = RunResult(passed=3, failed=0, errors=0, collected=True, coverage=0.0)
    some_cov = RunResult(passed=3, failed=0, errors=0, collected=True, coverage=50.0)
    llm = _llm([STUB_TESTS] * 5)
    # iter 1: 0% (streak=1), iter 2: 50% (streak resets), iter 3: 0% (streak=1),
    # iter 4: 0% (streak=2 → bail out)
    with patch("testloop.agent.run_tests",
               side_effect=[zero_cov, some_cov, zero_cov, zero_cov]):
        loop = generate_tests(SOURCE, llm, coverage_target=80.0, max_iterations=5)
    assert loop.outcome == "incomplete"
    assert loop.iterations == 4


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
