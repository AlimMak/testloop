"""Thin Anthropic client with token accounting and an offline mock mode.

Set ANTHROPIC_API_KEY in your environment for real runs. The model string is
configurable; swap it for whatever is current when you run this.
"""

from __future__ import annotations

import re

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 16384  # raised from 4096; full natsort run completes with headroom


class TruncatedResponseError(Exception):
    """Raised when the API stops early because the output token cap was reached.

    The response is incomplete and must NOT be used as-is: feeding truncated
    Python to pytest will produce a SyntaxError on the last line.  The caller
    should surface this as a distinct TRUNCATED outcome and advise the user to
    raise --max-tokens.
    """

    def __init__(self, partial: str, output_tokens: int) -> None:
        super().__init__(
            f"LLM response truncated at {output_tokens} output tokens "
            f"(stop_reason='max_tokens')"
        )
        self.partial = partial
        self.output_tokens = output_tokens


def _strip_fences(text: str) -> str:
    """Models sometimes wrap code in ```python fences despite instructions."""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


class LLM:
    def __init__(
        self,
        mock: bool = False,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.mock = mock
        self.model = model
        self.max_tokens = max_tokens
        self.input_tokens = 0
        self.output_tokens = 0
        self._client = None
        if mock:
            if not _MOCK:  # seed the canned demo unless tests injected their own
                _MOCK.extend(_DEMO)
        else:
            from anthropic import Anthropic  # imported lazily so mock runs need no key
            self._client = Anthropic()

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        """Generate a completion.

        Parameters
        ----------
        max_tokens:
            Override ``self.max_tokens`` for this call only.  Used by the
            retry-on-truncation path in ``agent.py`` to double the cap without
            permanently changing the LLM configuration.
        """
        if self.mock:
            text = _MOCK.pop(0) if _MOCK else "import target\n"
            return _strip_fences(text)
        effective_max = max_tokens if max_tokens is not None else self.max_tokens
        # Use streaming so the Anthropic API accepts large max_tokens values
        # without hitting the 10-minute non-streaming ceiling.  get_final_message()
        # returns the fully-assembled Message, so every field downstream touches
        # (content, stop_reason, usage) is identical to the non-streaming path.
        with self._client.messages.stream(
            model=self.model,
            max_tokens=effective_max,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            final = stream.get_final_message()
        self.input_tokens += final.usage.input_tokens
        self.output_tokens += final.usage.output_tokens
        # The response may contain non-text blocks (e.g. a thinking block) before
        # the text, so join every text block rather than assuming content[0].
        text = "\n".join(
            b.text for b in final.content if getattr(b, "type", None) == "text"
        )
        # Incomplete responses must not silently corrupt the generated file.
        if final.stop_reason == "max_tokens":
            raise TruncatedResponseError(partial=text,
                                         output_tokens=final.usage.output_tokens)
        return _strip_fences(text)


# Scripted responses used only in mock mode, so the agent loop can be exercised
# offline. Populated by tests, or seeded from _DEMO for a keyless CLI demo.
_MOCK: list[str] = []

# A canned two-step demo for example_target.py so `--mock` shows a real loop
# without an API key: attempt 1 leaves a coverage gap, attempt 2 closes it.
_DEMO: list[str] = [
    (
        "import target\n"
        "def test_add():\n"
        "    assert target.add(2, 3) == 5\n"
        "def test_divide_ok():\n"
        "    assert target.divide(10, 2) == 5\n"
    ),
    (
        "import target\n"
        "import pytest\n"
        "def test_add():\n"
        "    assert target.add(2, 3) == 5\n"
        "def test_divide_ok():\n"
        "    assert target.divide(10, 2) == 5\n"
        "def test_divide_zero():\n"
        "    with pytest.raises(ValueError):\n"
        "        target.divide(1, 0)\n"
        "def test_clamp():\n"
        "    assert target.clamp(5, 0, 10) == 5\n"
        "    assert target.clamp(-1, 0, 10) == 0\n"
        "    assert target.clamp(99, 0, 10) == 10\n"
        "def test_clamp_bad_bounds():\n"
        "    with pytest.raises(ValueError):\n"
        "        target.clamp(5, 10, 0)\n"
    ),
]
