"""Tests for testloop.runner — JUnit XML parsing and coverage-report parsing.

All tests are offline: no subprocesses, no Docker, no network.
"""
import json
import xml.etree.ElementTree as ET

import pytest

from testloop.runner import RunResult, _collect, _parse_junit


# ─── XML fixtures ─────────────────────────────────────────────────────────────

# Bare <testsuite> root (what pytest actually emits by default).
ALL_PASS_XML = """\
<testsuite name="pytest" tests="3" failures="0" errors="0" skipped="0">
  <testcase classname="test_target" name="test_a"/>
  <testcase classname="test_target" name="test_b"/>
  <testcase classname="test_target" name="test_c"/>
</testsuite>"""

# Wrapped in <testsuites> (JUnit canonical form).
SOME_FAILURES_XML = """\
<testsuites>
  <testsuite name="pytest" tests="5" failures="2" errors="0" skipped="0">
    <testcase name="test_a"/>
    <testcase name="test_b"><failure message="AssertionError"/></testcase>
    <testcase name="test_c"/>
    <testcase name="test_d"><failure message="AssertionError"/></testcase>
    <testcase name="test_e"/>
  </testsuite>
</testsuites>"""

# pytest writes errors=1, tests=0 when collection fails entirely.
COLLECTION_ERROR_XML = """\
<testsuite name="pytest" tests="0" errors="1" failures="0" skipped="0"/>"""

EMPTY_SUITE_XML = """\
<testsuite name="pytest" tests="0" failures="0" errors="0" skipped="0"/>"""

# Skipped tests count against the total but not against passed.
WITH_SKIPPED_XML = """\
<testsuite name="pytest" tests="4" failures="1" errors="0" skipped="1">
  <testcase name="test_a"/>
  <testcase name="test_b"><skipped/></testcase>
  <testcase name="test_c"><failure message="x"/></testcase>
  <testcase name="test_d"/>
</testsuite>"""

# Two suites inside <testsuites>; counts should be summed.
MULTI_SUITE_XML = """\
<testsuites>
  <testsuite name="suite_a" tests="2" failures="0" errors="0" skipped="0"/>
  <testsuite name="suite_b" tests="3" failures="1" errors="0" skipped="0"/>
</testsuites>"""


# ─── _parse_junit — table-driven ─────────────────────────────────────────────

@pytest.mark.parametrize("xml_src, expected", [
    pytest.param(ALL_PASS_XML,        (3, 0, 0, True),   id="all_pass"),
    pytest.param(SOME_FAILURES_XML,   (3, 2, 0, True),   id="some_failures"),
    # tests=0, errors=1 → passed = 0 - 0 - 1 - 0 = -1, collected = False
    pytest.param(COLLECTION_ERROR_XML, (-1, 0, 1, False), id="collection_error"),
    pytest.param(EMPTY_SUITE_XML,     (0, 0, 0, False),  id="empty_suite"),
    pytest.param(WITH_SKIPPED_XML,    (2, 1, 0, True),   id="with_skipped"),
    pytest.param(MULTI_SUITE_XML,     (4, 1, 0, True),   id="multi_suite"),
])
def test_parse_junit(tmp_path, xml_src, expected):
    p = tmp_path / "results.xml"
    p.write_text(xml_src, encoding="utf-8")
    assert _parse_junit(str(p)) == expected


def test_parse_junit_malformed_raises(tmp_path):
    p = tmp_path / "results.xml"
    p.write_text("this is not xml <<< >>>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        _parse_junit(str(p))


# ─── _collect — coverage-report parsing ──────────────────────────────────────

def _write_coverage(tmp_path, data: dict) -> None:
    (tmp_path / "coverage.json").write_text(json.dumps(data), encoding="utf-8")


def _write_junit(tmp_path, content: str) -> None:
    (tmp_path / "results.xml").write_text(content, encoding="utf-8")


def test_collect_reads_pass_counts(tmp_path):
    _write_junit(tmp_path, ALL_PASS_XML)
    result = _collect(str(tmp_path), "")
    assert result.passed == 3
    assert result.failed == 0
    assert result.errors == 0
    assert result.collected is True


def test_collect_reads_coverage_percent(tmp_path):
    _write_junit(tmp_path, ALL_PASS_XML)
    _write_coverage(tmp_path, {
        "totals": {"percent_covered": 75.0},
        "files": {"target.py": {"missing_lines": []}},
    })
    result = _collect(str(tmp_path), "")
    assert result.coverage == 75.0


def test_collect_reads_uncovered_lines(tmp_path):
    _write_junit(tmp_path, ALL_PASS_XML)
    _write_coverage(tmp_path, {
        "totals": {"percent_covered": 60.0},
        "files": {"target.py": {"missing_lines": [10, 20, 30]}},
    })
    result = _collect(str(tmp_path), "")
    assert result.uncovered_lines == [10, 20, 30]


def test_collect_falls_back_to_first_file_when_no_target_key(tmp_path):
    """If coverage.json has no 'target.py' key, use the first file listed."""
    _write_junit(tmp_path, ALL_PASS_XML)
    _write_coverage(tmp_path, {
        "totals": {"percent_covered": 80.0},
        "files": {"some_other.py": {"missing_lines": [5, 6]}},
    })
    result = _collect(str(tmp_path), "")
    assert result.uncovered_lines == [5, 6]


def test_collect_missing_junit_sets_error(tmp_path):
    """When pytest aborts before writing a report, _collect signals an error."""
    # No results.xml written.
    result = _collect(str(tmp_path), "collection crashed")
    assert result.collected is False
    assert result.errors == 1


def test_collect_missing_coverage_json_uses_defaults(tmp_path):
    """Missing coverage.json leaves coverage at 0.0 with no uncovered lines."""
    _write_junit(tmp_path, ALL_PASS_XML)
    # No coverage.json written.
    result = _collect(str(tmp_path), "")
    assert result.coverage == 0.0
    assert result.uncovered_lines == []


def test_collect_truncates_long_output(tmp_path):
    _write_junit(tmp_path, ALL_PASS_XML)
    long_output = "x" * 10_000
    result = _collect(str(tmp_path), long_output)
    assert len(result.output) <= 6000


def test_collect_output_is_tail_of_input(tmp_path):
    """The stored output is the last 6 000 chars, not the first."""
    _write_junit(tmp_path, ALL_PASS_XML)
    # 5 000 "A"s then 5 000 "B"s; the last 6 000 chars are 1 000 "A"s + 5 000 "B"s.
    output = "A" * 5_000 + "B" * 5_000
    result = _collect(str(tmp_path), output)
    assert result.output == "A" * 1_000 + "B" * 5_000


# ─── RunResult.all_passed property ───────────────────────────────────────────

@pytest.mark.parametrize("kwargs, expected", [
    pytest.param(
        dict(passed=3, failed=0, errors=0, collected=True),
        True,
        id="all_pass",
    ),
    pytest.param(
        dict(passed=0, failed=0, errors=0, collected=True),
        False,
        id="zero_passed",
    ),
    pytest.param(
        dict(passed=2, failed=1, errors=0, collected=True),
        False,
        id="has_failure",
    ),
    pytest.param(
        dict(passed=2, failed=0, errors=1, collected=True),
        False,
        id="has_error",
    ),
    pytest.param(
        dict(passed=3, failed=0, errors=0, collected=False),
        False,
        id="not_collected",
    ),
])
def test_all_passed_property(kwargs, expected):
    result = RunResult(**kwargs)
    assert result.all_passed is expected


# ─── RunResult.total property ────────────────────────────────────────────────

@pytest.mark.parametrize("passed, failed, errors, expected", [
    pytest.param(0, 0, 0, 0, id="all_zero"),
    pytest.param(5, 0, 0, 5, id="only_passed"),
    pytest.param(0, 3, 0, 3, id="only_failed"),
    pytest.param(0, 0, 2, 2, id="only_errors"),
    pytest.param(4, 3, 1, 8, id="mixed"),
])
def test_total_property(passed, failed, errors, expected):
    result = RunResult(passed=passed, failed=failed, errors=errors)
    assert result.total == expected


# ─── RunResult.collection_error property ─────────────────────────────────────

@pytest.mark.parametrize("kwargs, expected", [
    pytest.param(
        dict(collected=False, timed_out=False),
        True,
        id="not_collected_no_timeout",
    ),
    pytest.param(
        dict(collected=False, timed_out=True),
        False,
        id="timeout_is_not_collection_error",
    ),
    pytest.param(
        dict(collected=True, timed_out=False),
        False,
        id="collected_ok",
    ),
    pytest.param(
        dict(collected=True, timed_out=True),
        False,
        id="collected_and_timed_out",
    ),
    # pytest sometimes writes tests="1" errors="1" for a collection-error item,
    # giving collected=True but passed=0 and failed=0.  This must still be
    # treated as a collection error, not a run with one errored test.
    pytest.param(
        dict(passed=0, failed=0, errors=1, collected=True, timed_out=False),
        True,
        id="collected_true_but_only_errors",
    ),
    # A run with real test results plus an error is NOT a collection error.
    pytest.param(
        dict(passed=3, failed=0, errors=1, collected=True, timed_out=False),
        False,
        id="real_tests_plus_error",
    ),
])
def test_collection_error_property(kwargs, expected):
    result = RunResult(**kwargs)
    assert result.collection_error is expected
