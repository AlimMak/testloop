# testloop

A closed-loop test generation agent. Point it at a Python file; it writes a
pytest suite, runs it in an isolated sandbox, reads the failures and coverage
gaps, repairs itself, and loops until the tests pass and hit your coverage
target (or it runs out of iterations).

## Why this is not a wrapper

The interesting part is not "call a model to write tests." It is the feedback
loop: the agent acts, observes real execution results, and uses those
observations to decide what to do next. Structured `RunResult` data (pass/fail
counts, exact uncovered line numbers, timeouts) is fed back into the repair
prompt, so each iteration is grounded in what actually happened rather than in
the model guessing.

## Run it

```bash
pip install pytest pytest-cov anthropic
export ANTHROPIC_API_KEY=sk-...
python -m testloop example_target.py --coverage 90 --max-iters 5
```

Offline demo with no API key (uses scripted responses):

```bash
python -m testloop example_target.py --mock
```

## Architecture

- `runner.py` runs generated tests in a fresh temp dir inside a subprocess with
  a hard timeout, then parses structured results from pytest-json-report and
  coverage.py. Returns pass/fail counts, coverage percent, and the exact
  uncovered line numbers.
- `agent.py` is the loop: generate, run, observe, repair, stop on success.
- `llm.py` wraps the Anthropic SDK, tracks token usage, and has a mock mode.
- `prompts.py` holds the generate and repair prompts (the main tuning surface).
- `cli.py` is the entry point.

## The sandbox boundary (say this in interviews)

Running model-generated code is a real risk. v0 uses process isolation plus a
timeout: a fresh working directory, a subprocess that can be killed, and a wall
clock limit that kills infinite loops. That contains accidents and runaway
loops but does not stop deliberately malicious code, since it still runs on the
host interpreter. The hardening path is a Docker sandbox with no network and a
read-only mount. Knowing exactly where your isolation stops is the point.

## Roadmap / stretch

- Docker sandbox (network off, read-only source mount)
- Coverage-guided repair that targets specific uncovered branches, not just lines
- Distinguish "the test is wrong" from "the source has a bug" and surface both
- Multi-file / whole-repo mode
- Cost estimation from token counts
- A GitHub Action that runs the loop on new PRs