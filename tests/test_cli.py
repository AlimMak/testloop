"""Tests for testloop.cli — output file placement.

All tests patch generate_tests so no subprocess, network, or API calls occur.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from testloop.agent import LoopResult
from testloop.cli import main
from testloop.runner import RunResult


_PASSING = RunResult(passed=2, failed=0, errors=0, collected=True, coverage=90.0)


def _stub_loop(*args, **kwargs) -> LoopResult:
    return LoopResult(
        tests="def test_stub(): pass\n",
        result=_PASSING,
        iterations=1,
        outcome="success",
        input_tokens=0,
        output_tokens=0,
    )


# ─── Single-file mode ─────────────────────────────────────────────────────────

def test_single_file_default_lands_next_to_source(tmp_path):
    """Without -o, the generated file is written next to the source, not cwd."""
    src = tmp_path / "mymodule.py"
    src.write_text("def f(): pass\n", encoding="utf-8")
    with patch("testloop.cli.generate_tests", side_effect=_stub_loop):
        rc = main([str(src), "--mock"])
    assert rc == 0
    assert (tmp_path / "test_mymodule.py").exists()


def test_single_file_default_not_in_tests_dir(tmp_path):
    """Default single-file output must NOT appear inside the repo's tests/ dir."""
    src = tmp_path / "mymodule.py"
    src.write_text("def f(): pass\n", encoding="utf-8")
    with patch("testloop.cli.generate_tests", side_effect=_stub_loop):
        main([str(src), "--mock"])
    assert not (Path("tests") / "test_mymodule.py").exists()


def test_single_file_out_flag_overrides_default(tmp_path):
    """-o <path> writes the file at that path instead of next to the source."""
    src = tmp_path / "mymodule.py"
    src.write_text("def f(): pass\n", encoding="utf-8")
    custom = tmp_path / "custom_tests.py"
    with patch("testloop.cli.generate_tests", side_effect=_stub_loop):
        rc = main([str(src), "--mock", "-o", str(custom)])
    assert rc == 0
    assert custom.exists()
    assert not (tmp_path / "test_mymodule.py").exists()


# ─── Directory mode ───────────────────────────────────────────────────────────

def test_directory_mode_default_lands_next_to_source(tmp_path):
    """Without --out-dir, each generated test file lands next to its source module."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text("def f(): pass\n", encoding="utf-8")
    with patch("testloop.cli.generate_tests", side_effect=_stub_loop):
        rc = main([str(pkg), "--mock"])
    assert rc == 0
    assert (pkg / "test_utils.py").exists()


def test_directory_mode_default_not_in_tests_dir(tmp_path):
    """Default directory-mode output must NOT appear inside the repo's tests/ dir."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text("def f(): pass\n", encoding="utf-8")
    with patch("testloop.cli.generate_tests", side_effect=_stub_loop):
        main([str(pkg), "--mock"])
    assert not (Path("tests") / "test_mypkg_utils.py").exists()


def test_directory_mode_out_dir_override(tmp_path):
    """--out-dir collects all generated tests into the specified directory."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text("def f(): pass\n", encoding="utf-8")
    out_dir = tmp_path / "generated"
    with patch("testloop.cli.generate_tests", side_effect=_stub_loop):
        rc = main([str(pkg), "--mock", "--out-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "test_mypkg_utils.py").exists()
    assert not (pkg / "test_utils.py").exists()


def test_directory_mode_out_dir_created_if_absent(tmp_path):
    """--out-dir is created automatically when it does not yet exist."""
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text("def f(): pass\n", encoding="utf-8")
    out_dir = tmp_path / "does_not_exist" / "nested"
    with patch("testloop.cli.generate_tests", side_effect=_stub_loop):
        main([str(pkg), "--mock", "--out-dir", str(out_dir)])
    assert out_dir.exists()
