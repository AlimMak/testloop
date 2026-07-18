"""Tests for testloop.discovery — module walking and package snapshotting.

All tests are offline: no subprocesses, no network, no temp directories beyond
what pytest's tmp_path fixture provides.
"""

import pytest

from testloop.discovery import (
    _dotted,
    discover_all,
    find_import_root,
    _is_reexport_only,
    _is_skip_dir,
    _is_skip_file,
    collect_package_files,
    discover_modules,
)


# ─── _is_skip_dir ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, expected", [
    ("__pycache__", True),
    (".venv",       True),
    ("venv",        True),
    (".git",        True),
    (".hidden",     True),
    ("src",         False),
    ("mypkg",       False),
    ("tests",       False),
])
def test_is_skip_dir(name, expected):
    assert _is_skip_dir(name) is expected


# ─── _is_skip_file ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name, expected", [
    ("test_utils.py",    True),
    ("test_foo.py",      True),
    ("conftest.py",      True),
    ("setup.py",         True),
    ("__main__.py",      True),   # CLI entry point — not unit-testable
    ("utils.py",         False),
    ("__init__.py",      False),
    ("models.py",        False),
])
def test_is_skip_file(name, expected):
    assert _is_skip_file(name) is expected


# ─── _is_reexport_only ────────────────────────────────────────────────────────

def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_reexport_empty(tmp_path):
    p = _write(tmp_path, "__init__.py", "")
    assert _is_reexport_only(p) is True


def test_reexport_whitespace_only(tmp_path):
    p = _write(tmp_path, "__init__.py", "   \n\n   ")
    assert _is_reexport_only(p) is True


def test_reexport_docstring_only(tmp_path):
    p = _write(tmp_path, "__init__.py", '"""Package docstring."""\n')
    assert _is_reexport_only(p) is True


def test_reexport_import_only(tmp_path):
    p = _write(tmp_path, "__init__.py",
               "from .utils import helper\nfrom .models import Model\n")
    assert _is_reexport_only(p) is True


def test_reexport_dunder_assignments(tmp_path):
    p = _write(tmp_path, "__init__.py",
               '__all__ = ["helper"]\n__version__ = "1.0.0"\n')
    assert _is_reexport_only(p) is True


def test_reexport_try_except_import_guard(tmp_path):
    """try/except ImportError guards (e.g. optional C-extension imports) must be
    treated as re-export-only — they contain no FunctionDef or ClassDef.

    This is the pattern used in natsort's __init__.py:
        try:
            from natsort._natsort import natsorted
        except ImportError:
            from natsort.natsort import natsorted
    """
    p = _write(tmp_path, "__init__.py",
               "try:\n"
               "    from ._fast import helper\n"
               "except ImportError:\n"
               "    from ._slow import helper\n")
    assert _is_reexport_only(p) is True


def test_reexport_globals_update_call(tmp_path):
    """A module-level globals().update({...}) call with no function definitions
    must be treated as re-export-only.

    natsort's __init__.py also contains:
        globals().update({'natsorted': natsorted, ...})
    which is an Expr node with a Call — no FunctionDef present.
    """
    p = _write(tmp_path, "__init__.py",
               "from .core import natsorted\n"
               "globals().update({'natsorted': natsorted})\n")
    assert _is_reexport_only(p) is True


def test_reexport_try_except_with_function_inside_is_not_reexport(tmp_path):
    """A try block that contains a function definition is NOT re-export-only."""
    p = _write(tmp_path, "__init__.py",
               "try:\n"
               "    def _init(): pass\n"
               "except Exception:\n"
               "    pass\n")
    assert _is_reexport_only(p) is False


def test_reexport_with_real_logic(tmp_path):
    p = _write(tmp_path, "__init__.py",
               "from .utils import helper\n\ndef init_plugin():\n    pass\n")
    assert _is_reexport_only(p) is False


def test_reexport_missing_file(tmp_path):
    p = tmp_path / "nonexistent.py"
    assert _is_reexport_only(p) is True


def test_reexport_syntax_error(tmp_path):
    p = _write(tmp_path, "__init__.py", "def (broken syntax:\n")
    # Unparseable — return False so the file is included and the syntax error
    # surfaces naturally when tests run.
    assert _is_reexport_only(p) is False


# ─── _dotted ──────────────────────────────────────────────────────────────────

def test_dotted_top_level(tmp_path):
    f = tmp_path / "utils.py"
    f.touch()
    assert _dotted(tmp_path, f) == "utils"


def test_dotted_nested(tmp_path):
    (tmp_path / "mypkg").mkdir()
    f = tmp_path / "mypkg" / "utils.py"
    f.touch()
    assert _dotted(tmp_path, f) == "mypkg.utils"


def test_dotted_deeply_nested(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    f = tmp_path / "a" / "b" / "c.py"
    f.touch()
    assert _dotted(tmp_path, f) == "a.b.c"


def test_dotted_init(tmp_path):
    (tmp_path / "mypkg").mkdir()
    f = tmp_path / "mypkg" / "__init__.py"
    f.touch()
    assert _dotted(tmp_path, f) == "mypkg.__init__"


# ─── discover_modules ─────────────────────────────────────────────────────────

def _make_pkg(tmp_path, files: dict[str, str]) -> None:
    """Write a package tree under tmp_path from a {rel_path: content} dict."""
    for rel, content in files.items():
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def test_discover_simple_package(tmp_path):
    _make_pkg(tmp_path, {
        "mypkg/__init__.py": "from .utils import helper\n",  # re-export only
        "mypkg/utils.py": "def helper(): pass\n",
        "mypkg/models.py": "class Model: pass\n",
    })
    result = discover_modules(tmp_path)
    dotted_names = [d for d, _ in result]
    assert "mypkg.utils" in dotted_names
    assert "mypkg.models" in dotted_names
    # empty re-export __init__ skipped
    assert "mypkg.__init__" not in dotted_names


def test_discover_skips_test_files(tmp_path):
    _make_pkg(tmp_path, {
        "utils.py": "def f(): pass\n",
        "test_utils.py": "def test_f(): pass\n",
        "conftest.py": "import pytest\n",
    })
    result = discover_modules(tmp_path)
    names = [d for d, _ in result]
    assert "utils" in names
    assert "test_utils" not in names
    assert "conftest" not in names


def test_discover_skips_pycache(tmp_path):
    _make_pkg(tmp_path, {
        "mypkg/__init__.py": "",
        "mypkg/__pycache__/utils.cpython-311.pyc": "",  # not .py but test the dir
        "mypkg/utils.py": "def f(): pass\n",
    })
    # pycache dir isn't .py so rglob won't pick it up, but add a .py there too:
    cache_dir = tmp_path / "mypkg" / "__pycache__"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "generated.py").write_text("x = 1\n", encoding="utf-8")
    result = discover_modules(tmp_path)
    names = [d for d, _ in result]
    assert "mypkg.utils" in names
    # file under __pycache__ must NOT appear
    assert not any("__pycache__" in d for d in names)


def test_discover_skips_hidden_dirs(tmp_path):
    _make_pkg(tmp_path, {
        ".hidden/secret.py": "SECRET = 1\n",
        "public.py": "def f(): pass\n",
    })
    result = discover_modules(tmp_path)
    names = [d for d, _ in result]
    assert "public" in names
    assert "secret" not in names


def test_discover_includes_init_with_logic(tmp_path):
    _make_pkg(tmp_path, {
        "mypkg/__init__.py": "def init_plugin():\n    pass\n",
    })
    result = discover_modules(tmp_path)
    names = [d for d, _ in result]
    assert "mypkg.__init__" in names


def test_discover_respects_max_files_via_slice(tmp_path):
    """discover_modules returns all; callers apply max-files via slicing."""
    _make_pkg(tmp_path, {
        "a.py": "def fa(): pass\n",
        "b.py": "def fb(): pass\n",
        "c.py": "def fc(): pass\n",
    })
    all_modules = discover_modules(tmp_path)
    assert len(all_modules[:2]) == 2


def test_discover_returns_absolute_paths(tmp_path):
    _make_pkg(tmp_path, {"utils.py": "def f(): pass\n"})
    result = discover_modules(tmp_path)
    assert len(result) == 1
    _, path = result[0]
    assert path.is_absolute()


def test_discover_empty_dir(tmp_path):
    assert discover_modules(tmp_path) == []


def test_discover_skips_main_module(tmp_path):
    """__main__.py is a CLI entry point and must not appear in testable modules."""
    _make_pkg(tmp_path, {
        "mypkg/__init__.py": "from .utils import helper\n",
        "mypkg/__main__.py": "from mypkg.cli import main\nif __name__ == '__main__': main()\n",
        "mypkg/utils.py": "def helper(): pass\n",
    })
    result = discover_modules(tmp_path)
    names = [d for d, _ in result]
    assert "mypkg.utils" in names
    assert "mypkg.__main__" not in names


def test_discover_skips_reexport_only_non_init(tmp_path):
    """Any .py file (not just __init__.py) that is re-export-only is skipped."""
    _make_pkg(tmp_path, {
        "pkg/__init__.py": "from .core import func\n",  # re-export only → skip
        "pkg/compat.py":   "from .core import func\n",  # re-export only → skip
        "pkg/core.py":     "def func(): pass\n",        # has logic → keep
    })
    result = discover_modules(tmp_path)
    names = [d for d, _ in result]
    assert "pkg.core" in names
    assert "pkg.__init__" not in names
    assert "pkg.compat" not in names


# ─── discover_all ─────────────────────────────────────────────────────────────

def test_discover_all_splits_testable_and_skipped(tmp_path):
    _make_pkg(tmp_path, {
        "pkg/__init__.py": "from .core import func\n",  # re-export only
        "pkg/core.py":     "def func(): pass\n",
    })
    testable, skipped = discover_all(tmp_path)
    testable_names = [d for d, _ in testable]
    skipped_names  = [d for d, _ in skipped]
    assert "pkg.core" in testable_names
    assert "pkg.__init__" in skipped_names
    assert "pkg.__init__" not in testable_names


def test_discover_all_main_not_in_skipped(tmp_path):
    """__main__.py is silently excluded — it appears in neither list."""
    _make_pkg(tmp_path, {
        "pkg/__main__.py": "import sys\n",
        "pkg/core.py":     "def func(): pass\n",
    })
    testable, skipped = discover_all(tmp_path)
    all_names = [d for d, _ in testable] + [d for d, _ in skipped]
    assert "pkg.__main__" not in all_names
    assert "pkg.core" in [d for d, _ in testable]


def test_discover_all_empty_init_is_skipped(tmp_path):
    _make_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/utils.py": "def f(): pass\n",
    })
    testable, skipped = discover_all(tmp_path)
    skipped_names = [d for d, _ in skipped]
    assert "pkg.__init__" in skipped_names


# ─── collect_package_files ────────────────────────────────────────────────────

def test_collect_all_py_files(tmp_path):
    _make_pkg(tmp_path, {
        "mypkg/__init__.py": "",
        "mypkg/utils.py": "def f(): pass\n",
        "mypkg/test_utils.py": "def test_f(): pass\n",  # included in snapshot
        "mypkg/conftest.py": "# fixtures\n",             # included in snapshot
    })
    files = collect_package_files(tmp_path)
    assert "mypkg/__init__.py" in files
    assert "mypkg/utils.py" in files
    assert "mypkg/test_utils.py" in files
    assert "mypkg/conftest.py" in files


def test_collect_excludes_hidden_dirs(tmp_path):
    _make_pkg(tmp_path, {
        "mypkg/utils.py": "def f(): pass\n",
        ".hidden/secret.py": "SECRET = 1\n",
    })
    files = collect_package_files(tmp_path)
    assert any("utils" in k for k in files)
    assert not any(".hidden" in k for k in files)


def test_collect_excludes_pycache(tmp_path):
    _make_pkg(tmp_path, {"mypkg/utils.py": "def f(): pass\n"})
    cache = tmp_path / "mypkg" / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "generated.py").write_text("x = 1\n", encoding="utf-8")
    files = collect_package_files(tmp_path)
    assert not any("__pycache__" in k for k in files)


def test_collect_posix_paths(tmp_path):
    _make_pkg(tmp_path, {"mypkg/sub/utils.py": "def f(): pass\n"})
    files = collect_package_files(tmp_path)
    assert "mypkg/sub/utils.py" in files  # forward slashes always


def test_collect_preserves_content(tmp_path):
    content = "def answer(): return 42\n"
    _make_pkg(tmp_path, {"utils.py": content})
    files = collect_package_files(tmp_path)
    assert files["utils.py"] == content


# ─── find_import_root ─────────────────────────────────────────────────────────

def test_import_root_non_package_dir(tmp_path):
    """A directory without __init__.py is already the import root."""
    _make_pkg(tmp_path, {"mypkg/__init__.py": "", "mypkg/utils.py": "x=1\n"})
    # tmp_path has no __init__.py — it is its own import root
    assert find_import_root(tmp_path) == tmp_path.resolve()


def test_import_root_steps_up_one(tmp_path):
    """When root itself is a package, step up to its parent."""
    pkg = tmp_path / "mypkg"
    _make_pkg(tmp_path, {"mypkg/__init__.py": "", "mypkg/utils.py": "x=1\n"})
    # mypkg/ has __init__.py; tmp_path does not → import root is tmp_path
    assert find_import_root(pkg) == tmp_path.resolve()


def test_import_root_steps_up_multiple(tmp_path):
    """Walk up through nested packages until finding a non-package ancestor."""
    _make_pkg(tmp_path, {
        "top/__init__.py": "",
        "top/sub/__init__.py": "",
        "top/sub/leaf.py": "x=1\n",
    })
    leaf_pkg = tmp_path / "top" / "sub"
    # top/sub has __init__.py, top has __init__.py, tmp_path does not
    assert find_import_root(leaf_pkg) == tmp_path.resolve()


def test_import_root_already_at_root_uses_parent(tmp_path):
    """A plain (non-package) directory returns itself unchanged."""
    assert find_import_root(tmp_path) == tmp_path.resolve()


def test_import_root_keys_have_package_prefix(tmp_path):
    """When root is a package, collect_package_files uses the import root so
    keys include the package name — not flat bare filenames."""
    _make_pkg(tmp_path, {
        "mypkg/__init__.py": "",
        "mypkg/utils.py": "def f(): pass\n",
    })
    pkg_dir = tmp_path / "mypkg"
    import_root = find_import_root(pkg_dir)
    files = collect_package_files(import_root)
    # Keys must be package-relative paths, not flat names.
    assert "mypkg/utils.py" in files
    assert "utils.py" not in files


def test_import_root_dotted_names_include_package(tmp_path):
    """discover_modules(find_import_root(pkg)) returns dotted names that include
    the package prefix, which is what the sandbox and prompts need."""
    _make_pkg(tmp_path, {
        "mypkg/__init__.py": "",          # empty — will be skipped
        "mypkg/utils.py": "def f(): pass\n",
    })
    pkg_dir = tmp_path / "mypkg"
    import_root = find_import_root(pkg_dir)
    modules = discover_modules(import_root)
    names = [d for d, _ in modules]
    assert "mypkg.utils" in names
    # bare "utils" must NOT appear
    assert "utils" not in names
