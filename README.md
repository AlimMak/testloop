# testloop

[![CI](https://github.com/AlimMak/testloop/actions/workflows/ci.yml/badge.svg)](https://github.com/AlimMak/testloop/actions)

![demo](assets/demo/.gif)

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

## Install

```bash
pip install -e .          # or: pipx install -e .
```

## Run it

```bash
export ANTHROPIC_API_KEY=sk-...
testloop examples/example_target.py --coverage 90 --max-iters 5
```

Offline demo with no API key (uses scripted responses):

```bash
testloop examples/example_target.py --mock
```

`python -m testloop` also works if you prefer.

## Architecture

- `runner.py` runs generated tests in a fresh temp dir inside a subprocess with
  a hard timeout, then parses structured results from pytest's built-in JUnit XML
  and coverage.py. Returns pass/fail counts, coverage percent, and the exact
  uncovered line numbers.
- `agent.py` is the loop: generate, run, observe, repair, stop on success. It
  also distinguishes "the test is wrong" from "the source has a bug": when the
  model signals a genuine defect with a `TESTLOOP_SOURCE_BUG` marker, the loop
  stops and surfaces it as a finding instead of contorting the tests to pass.
- `llm.py` wraps the Anthropic SDK, tracks token usage, and has a mock mode.
- `prompts.py` holds the generate and repair prompts (the main tuning surface).
- `cli.py` is the entry point.

## Sandboxing

Running model-generated code is the real risk in a tool like this, so isolation
is a first-class feature rather than an afterthought.

**local (default)** — a fresh temp dir plus a subprocess with a hard timeout.
That contains accidents and kills infinite loops, but the code still runs on the
host interpreter with your permissions. Fine for code you trust.

**docker (`--docker`)** — the same tests run in a throwaway container:

- `--network=none` so generated code cannot phone home or pull anything
- `--memory=512m` and `--pids-limit=128` so it cannot exhaust or fork-bomb the host
- the only host path it can see is one throwaway temp dir mounted at `/work`
- the container is named, so a timeout kills it by name; a runaway test cannot
  outlive the run
- `--rm` so nothing is left behind

Build the image once:

```bash
docker build -t testloop-sandbox .
testloop billing.py --docker --coverage 95
```

Where it still stops: the container runs as root and shares the host kernel, so
this is isolation against runaway and misbehaving code, not against a determined
attacker with a kernel exploit. Knowing exactly where your isolation ends is the
point.

## Roadmap / stretch

- Non-root container user and a read-only source mount
- Coverage-guided repair that targets specific uncovered branches, not just lines
- Multi-file / whole-repo mode
- Cost estimation from token counts
- A GitHub Action that runs the loop on new PRs
