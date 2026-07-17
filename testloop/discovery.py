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

# File patterns that are skipped during module discovery (but not when
# collecting the full package snapshot for the sandbox).
_SKIP_FILE_PATTERNS: tuple[str, ...] = ("test_*.py", "conftest.py", "setup.py")


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def _is_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or _is_hidden(name)


def _is_skip_file(name: str) -> bool:
    return any(fnmatch(name, p) for p in _SKIP_FILE_PATTERNS)


def _is_reexport_only(path: Path) -> bool:
    """Return True when __init__.py is empty or only contains re-export boilerplate.

    An __init__.py is considered a re-export-only file when every statement is
    one of:
    - an import or from-import (re-exporting names from sub-modules)
    - a bare string literal (docstring / __all__ prose)
    - an assignment whose targets are all ``__dunder__`` names
      (e.g. ``__all__ = [...]``, ``__version__ = "1.0"``)

    Such files contain no testable logic of their own and are skipped by
    :func:`discover_modules`.
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
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # bare string (docstring or __all__ prose)
        if isinstance(stmt, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id.startswith("__") and t.id.endswith("__")
            for t in stmt.targets
        ):
            continue  # __all__ = [...], __version__ = "1.0", etc.
        return False
    return True


def _dotted(root: Path, path: Path) -> str:
    """Return the dotted module name of *path* relative to *root*."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def discover_modules(root: Path) -> list[tuple[str, Path]]:
    """Return ``(dotted_module_name, absolute_path)`` for testable modules under *root*.

    Skipped:
    - ``test_*.py``, ``conftest.py``, ``setup.py``
    - ``__pycache__``, ``.venv``, and any directory whose name starts with ``.``
    - ``__init__.py`` that is empty or only re-exports (no testable logic)

    Results are sorted by dotted name for deterministic ordering.
    """
    root = root.resolve()
    results: list[tuple[str, Path]] = []
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root)
        if any(_is_skip_dir(part) for part in rel.parts[:-1]):
            continue
        if _is_skip_file(py_file.name):
            continue
        if py_file.name == "__init__.py" and _is_reexport_only(py_file):
            continue
        results.append((_dotted(root, py_file), py_file))
    return results


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
