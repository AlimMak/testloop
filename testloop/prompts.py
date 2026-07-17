"""Prompt templates for the generate and repair phases.

Kept in one place so they are easy to iterate on. Prompt quality is most of the
product here, so treat this file as the tuning surface.

All templates use ``str.format(**kwargs)`` with named placeholders. The
``module_dotted`` placeholder carries the importable dotted name of the module
under test (e.g. ``"target"`` in single-file mode, ``"mypkg.utils"`` in
package mode) so the model always generates the correct import statement.
"""

GENERATE_SYSTEM = """You are a precise pytest test generator.
Output rules (strict):
- Output ONLY Python code. No markdown fences, no commentary.
- The module under test is `{module_dotted}`.
  You MUST import it by its full dotted name. For example:
      import {module_dotted}
  or:
      from {module_dotted} import some_function, AnotherClass
  Never import only the last component of the name — that causes an ImportError
  because the module lives inside a package and is not importable as a top-level name.
- Cover normal cases, boundaries, and error paths (exceptions).
- Tests must be deterministic: no network, no filesystem, no randomness, no real time.
- Prefer many small focused tests over a few large ones.
"""

GENERATE_USER = """Source under test ({module_dotted}):

```python
{source}
```

Write a thorough pytest suite for the public functions. Output only the test code."""

REPAIR_SYSTEM = """You are a pytest repair and coverage engine.
Output rules (strict):
- Normally, output ONLY the full, updated Python test file. No fences, no commentary.
- The module under test is `{module_dotted}` — always import it by its full dotted
  name (e.g. `import {module_dotted}` or `from {module_dotted} import ...`).
  If the current test has a bare import of only the last component, fix it first.
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

REPAIR_USER = """Source ({module_dotted}):
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
