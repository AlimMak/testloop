"""Prompt templates for the generate and repair phases.

Kept in one place so they are easy to iterate on. Prompt quality is most of the
product here, so treat this file as the tuning surface.

All templates that reference the module under test use ``str.format()`` with
named placeholders.  ``module_dotted`` carries the importable dotted name (e.g.
``"target"`` in single-file mode, ``"mypkg.utils"`` in package mode).

The import instruction is defined once in :func:`_import_contract` and injected
into both GENERATE_SYSTEM and REPAIR_SYSTEM via the ``{import_contract}``
placeholder.  This prevents the two prompts from drifting apart and giving the
model contradictory import rules.
"""


def _import_contract(module_dotted: str) -> str:
    """Return the canonical import instruction for both generate and repair prompts.

    Keeping this in one place means the rules are identical in both system
    prompts — a change here applies everywhere automatically.
    """
    return (
        f"- The module under test is `{module_dotted}`.\n"
        f"  Import it by its full dotted name — NEVER just the last component:\n"
        f"      import {module_dotted}                   # correct\n"
        f"      from {module_dotted} import func, Class  # correct\n"
        f"  A bare `import {module_dotted.rsplit('.', 1)[-1]}` causes ImportError;"
        f" `{module_dotted}` is not\n"
        f"  importable as a top-level name because it lives inside a package.\n"
        f"  If an existing test file uses only the last component, fix it first."
    )


GENERATE_SYSTEM = """\
You are a precise pytest test generator.
Output rules (strict):
- Output ONLY Python code. No markdown fences, no commentary.
{import_contract}
- Cover normal cases, boundaries, and error paths (exceptions).
- Tests must be deterministic: no network, no filesystem, no randomness, no real time.
- Prefer many small focused tests over a few large ones.
"""

GENERATE_USER = """\
Source under test ({module_dotted}):

```python
{source}
```

Write a thorough pytest suite for the public functions. Output only the test code."""

REPAIR_SYSTEM = """\
You are a pytest repair and coverage engine.
Output rules (strict):
- Normally, output ONLY the full, updated Python test file. No fences, no commentary.
{import_contract}
- Keep tests that already pass.
- Add new tests to exercise the uncovered lines listed below.
- Tests must stay deterministic.
- Do NOT weaken or rewrite an assertion just to make a failing test pass.

Source bug handling:
- If a test fails because the SOURCE CODE under test is genuinely wrong (its
  behavior is incorrect, not the test), do NOT change the test to accept the
  wrong behavior. Instead make the FIRST line of your reply exactly:
      TESTLOOP_SOURCE_BUG: <one sentence naming the bug and the correct behavior>
  Then, on the following lines, output the full test file, keeping the correct
  (failing) assertion with a `# SOURCE BUG:` comment on the line above it.
"""

REPAIR_USER = """\
Source ({module_dotted}):
```python
{source}
```

Current test file:
```python
{tests}
```

pytest output:
```
{output}
```

Coverage: {coverage}% (target {target}%). Uncovered source lines: {uncovered}

Return the full updated test file. Fix failures and add tests to reach the target."""
