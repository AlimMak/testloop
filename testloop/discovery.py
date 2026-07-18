"""Discover testable Python modules in a package tree.

Used by the CLI's directory mode to walk a package root, identify modules
worth testing, and snapshot the full file tree for the sandbox.
"""

from __future__ import annotations

import ast
from fnmatch import fnmatch
from pathlib import Path

# Directory names that are never descended into.
_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".venv", "venv", ".git", ".tox",
    "node_modules", "dist", "build", ".eggs", ".pytest_cache",
})

# File patterns that are silently excluded from module discovery (not reported
# in the summary table).  Test files, config shims, and CLI entry points are
# not unit-testable and produce no useful generated tests.
_SKIP_FILE_PATTERNS: tuple[str, ...] = (
    "test_*.py", "conftest.py", "setup.py", "__main__.py",
)


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _is_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or _is_hidden(name)


def _is_skip_file(name: str) -> bool:
    return any(fnmatch(name, p) for p in _SKIP_FILE_PATTERNS)


def _is_reexport_only(path: Path) -> bool:
    """Return True when the module contains no testable logic.

    A module is considered re-export-only when its AST has no function or
    class definitions at *any* nesting level.  This covers the common cases:

    - empty ``__init__.py``
    - ``__init__.py`` with only imports and ``__all__`` / dunder assignments
    - ``try/except ImportError`` guard blocks (e.g. optional C-extension imports)
    - ``globals().update(...)`` calls and other module-level expressions

    Checking the full walk rather than just top-level statements means
    ``try/except`` bodies are inspected correctly — a package that wraps its
    imports in a ``try`` block (e.g. natsort) is still classified as
    re-export-only unless a function or class definition appears somewhere.

    Unparseable (syntax-error) files return ``False`` so the error surfaces
    naturally when the generated tests attempt to import the module.
    """
    try:
        src = path.read_text(encoding="utf-8").strip()
    except OSError:
        return True
    if not src:
        return True
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False  # unparseable — let the test suite catch the syntax error
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False
    return True


def _dotted(root: Path, path: Path) -> str:
    """Return the dotted module name of *path* relative to *root*."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def discover_all(
    root: Path,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """Return ``(testable, skipped)`` module lists for *root*.

    *testable* — modules with real logic worth generating tests for.
    *skipped*  — modules that exist but are trivially untestable (re-export-only
                 files, empty ``__init__.py``, etc.).

    Files matching :data:`_SKIP_FILE_PATTERNS` (test files, ``__main__.py``, …)
    are excluded entirely and appear in neither list.

    A module is considered *trivial* (placed in *skipped*) when its top-level
    body contains only import statements, ``__all__`` / ``__version__``
    assignments, and bare string literals — no function or class definitions.
    This catches pure re-export ``__init__.py`` files as well as any other
    module that has no logic of its own.
    """
    root = root.resolve()
    testable: list[tuple[str, Path]] = []
    skipped: list[tuple[str, Path]] = []
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root)
        if any(_is_skip_dir(part) for part in rel.parts[:-1]):
            continue
        if _is_skip_file(py_file.name):
            continue
        dotted = _dotted(root, py_file)
        if _is_reexport_only(py_file):
            skipped.append((dotted, py_file))
        else:
            testable.append((dotted, py_file))
    return testable, skipped


def discover_modules(root: Path) -> list[tuple[str, Path]]:
    """Return ``(dotted_module_name, absolute_path)`` for testable modules under *root*.

    Silently excluded:
    - ``test_*.py``, ``conftest.py``, ``setup.py``, ``__main__.py``
    - ``__pycache__``, ``.venv``, and any directory whose name starts with ``.``
    - any ``.py`` file whose top-level body is re-export-only (imports, ``__all__``,
      constants only — no function or class definitions)

    Results are sorted by dotted name for deterministic ordering.

    For directory-mode CLI use, prefer :func:`discover_all` which also returns
    the list of skipped (trivial) modules for table reporting.
    """
    testable, _ = discover_all(root)
    return testable


def find_import_root(path: Path) -> Path:
    """Return the directory that must be on ``sys.path`` for *path* to be importable.

    Walk up from *path* while each directory is a Python package (has an
    ``__init__.py``).  The first ancestor that does *not* have ``__init__.py``
    is the import root — the directory where ``import <pkg>`` resolves.

    Examples::

        scratch/mypkg/   (has __init__.py)   → scratch/
        scratch/         (no __init__.py)     → scratch/
        src/pkg/sub/     (__init__.py at each level, src/ has none) → src/

    This is used by directory mode to ensure ``package_files`` keys are always
    relative to the import root (e.g. ``mypkg/utils.py``), never flat
    (e.g. ``utils.py``), so that relative imports inside the package work.
    """
    candidate = path.resolve()
    while (candidate / "__init__.py").exists():
        parent = candidate.parent
        if parent == candidate:       # filesystem root — stop
            break
        candidate = parent
    return candidate


def collect_package_files(root: Path) -> dict[str, str]:
    """Snapshot all Python source files under *root* for use in the sandbox.

    Returns ``{relative_posix_path: source_text}`` for every ``.py`` file that
    is not inside a hidden or build/cache directory.  The paths are POSIX-style
    (forward slashes) so they can be written on any platform.

    Unlike :func:`discover_modules`, this function includes ``__init__.py``,
    ``conftest.py``, and other support files so that the sandbox gets a
    complete, importable package tree.
    """
    root = root.resolve()
    files: dict[str, str] = {}
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root)
        if any(_is_skip_dir(part) for part in rel.parts[:-1]):
            continue
        try:
            files[rel.as_posix()] = py_file.read_text(encoding="utf-8")
        except OSError:
            pass
    return files
