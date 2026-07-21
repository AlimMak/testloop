"""Tests for testloop.llm — fence stripping and response-block filtering.

All tests are offline: LLM is constructed via __new__ so the anthropic package
import inside __init__ is never reached, and no real API calls are made.
"""
import types
from unittest.mock import MagicMock, call

import pytest

from testloop.llm import DEFAULT_MODEL, DEFAULT_MAX_TOKENS, LLM, TruncatedResponseError, _strip_fences


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fake_llm() -> LLM:
    """Return an LLM wired to a MagicMock client, bypassing __init__."""
    llm = LLM.__new__(LLM)
    llm.mock = False
    llm.model = DEFAULT_MODEL
    llm.max_tokens = DEFAULT_MAX_TOKENS
    llm.input_tokens = 0
    llm.output_tokens = 0
    llm._client = MagicMock()
    return llm


def _make_block(type_: str, **kwargs) -> types.SimpleNamespace:
    """Minimal stand-in for an Anthropic content block."""
    return types.SimpleNamespace(type=type_, **kwargs)


def _fake_response(blocks, *, input_tokens=10, output_tokens=5,
                   stop_reason="end_turn"):
    """Build a fake final Message object (returned by stream.get_final_message())."""
    return types.SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=types.SimpleNamespace(input_tokens=input_tokens,
                                    output_tokens=output_tokens),
    )


def _fake_stream(response):
    """Return a mock context manager whose get_final_message() returns *response*.

    Usage::

        llm._client.messages.stream.return_value = _fake_stream(_fake_response(...))
    """
    stream = MagicMock()
    stream.__enter__.return_value = stream   # `with ... as stream:` gets itself
    stream.get_final_message.return_value = response
    return stream


# ─── _strip_fences ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    pytest.param(
        "```python\ndef foo(): pass\n```",
        "def foo(): pass",
        id="python_fence",
    ),
    pytest.param(
        "```\ndef foo(): pass\n```",
        "def foo(): pass",
        id="bare_fence",
    ),
    pytest.param(
        "def foo(): pass",
        "def foo(): pass",
        id="no_fence_returned_as_is",
    ),
    pytest.param(
        "  def foo(): pass  \n",
        "def foo(): pass",
        id="no_fence_whitespace_stripped",
    ),
    pytest.param(
        "Here is the code:\n```python\ndef foo(): pass\n```\n",
        "def foo(): pass",
        id="fence_after_preamble",
    ),
    pytest.param(
        "```python\nfirst\n```\n```python\nsecond\n```",
        "first",
        id="multiple_fences_first_wins",
    ),
    pytest.param(
        "```python\nline1\nline2\nline3\n```",
        "line1\nline2\nline3",
        id="multiline_body_preserved",
    ),
    pytest.param(
        "```python\n  indented\n```",
        "indented",
        id="leading_whitespace_stripped_from_body",
    ),
])
def test_strip_fences(text, expected):
    assert _strip_fences(text) == expected


# ─── Streaming API surface ────────────────────────────────────────────────────

def test_complete_uses_stream_not_create():
    """complete() must call messages.stream, not the blocking messages.create."""
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("text", text="import target")])
    )
    llm.complete("sys", "user")
    llm._client.messages.stream.assert_called_once()
    llm._client.messages.create.assert_not_called()


def test_stream_accumulates_text_from_multiple_blocks():
    """Text blocks in the final message are joined; non-text blocks are skipped."""
    blocks = [
        _make_block("thinking", thinking="reasoning..."),
        _make_block("text", text="chunk one"),
        _make_block("text", text="chunk two"),
    ]
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response(blocks, stop_reason="end_turn")
    )
    assert llm.complete("sys", "user") == "chunk one\nchunk two"


def test_stream_stop_reason_from_final_message_controls_truncation():
    """stop_reason is read from the final message returned by get_final_message()."""
    partial = "import target\ndef test_foo():\n    # cut off here"
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response(
            [_make_block("text", text=partial)],
            stop_reason="max_tokens",
            output_tokens=16384,
        )
    )
    with pytest.raises(TruncatedResponseError) as exc_info:
        llm.complete("sys", "user")
    assert exc_info.value.partial == partial
    assert exc_info.value.output_tokens == 16384


# ─── Regression (a): various fence formats reach _strip_fences via complete() ─

def test_complete_strips_python_fence_from_api_response():
    """Fence stripping is applied to the assembled API response text."""
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("text", text="```python\nimport target\n```")])
    )
    assert llm.complete("sys", "user") == "import target"


def test_complete_strips_bare_fence_from_api_response():
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("text", text="```\nimport target\n```")])
    )
    assert llm.complete("sys", "user") == "import target"


def test_complete_returns_plain_text_unchanged():
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("text", text="import target\n")])
    )
    assert llm.complete("sys", "user") == "import target"


# ─── Regression (b): ThinkingBlock before TextBlock must not raise ────────────

def test_thinking_block_before_text_does_not_raise():
    """A leading ThinkingBlock (no .text attribute) must not cause AttributeError."""
    thinking = _make_block("thinking", thinking="Let me reason through this…")
    text_block = _make_block("text", text="def test_foo(): assert True")
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([thinking, text_block])
    )
    result = llm.complete("system", "user")
    assert result == "def test_foo(): assert True"


def test_only_text_blocks_are_joined():
    """Multiple text blocks are joined; all non-text blocks are silently skipped."""
    blocks = [
        _make_block("thinking", thinking="hmm"),
        _make_block("text", text="part one"),
        _make_block("tool_use", id="x", name="y", input={}),
        _make_block("text", text="part two"),
    ]
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(_fake_response(blocks))
    assert llm.complete("sys", "user") == "part one\npart two"


def test_all_non_text_blocks_produce_empty_string():
    """If the response has no text blocks at all, complete() returns an empty string."""
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("thinking", thinking="only thinking, no text")])
    )
    result = llm.complete("sys", "user")
    assert result == ""


# ─── Token accounting ─────────────────────────────────────────────────────────

def test_token_counts_accumulated_across_calls():
    text_block = _make_block("text", text="import target")
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([text_block], input_tokens=100, output_tokens=50)
    )
    llm.complete("s", "u")
    llm.complete("s", "u")
    assert llm.input_tokens == 200
    assert llm.output_tokens == 100


def test_token_counts_start_at_zero():
    llm = _fake_llm()
    assert llm.input_tokens == 0
    assert llm.output_tokens == 0


# ─── Mock-mode path ───────────────────────────────────────────────────────────

def test_mock_mode_pops_from_queue(monkeypatch):
    """complete() in mock mode returns scripted responses in FIFO order."""
    import testloop.llm as llm_module
    monkeypatch.setattr(llm_module, "_MOCK", ["response_one", "response_two"])
    llm = LLM.__new__(LLM)
    llm.mock = True
    llm.model = DEFAULT_MODEL
    llm.input_tokens = 0
    llm.output_tokens = 0
    assert llm.complete("s", "u") == "response_one"
    assert llm.complete("s", "u") == "response_two"


def test_mock_mode_returns_default_when_queue_empty(monkeypatch):
    """When _MOCK is drained, complete() falls back to 'import target\\n'."""
    import testloop.llm as llm_module
    monkeypatch.setattr(llm_module, "_MOCK", [])
    llm = LLM.__new__(LLM)
    llm.mock = True
    llm.model = DEFAULT_MODEL
    llm.input_tokens = 0
    llm.output_tokens = 0
    # _strip_fences strips trailing whitespace, so the fallback arrives without \n
    assert llm.complete("s", "u") == "import target"


# ─── stop_reason / TruncatedResponseError ────────────────────────────────────

def test_end_turn_returns_normally():
    """stop_reason='end_turn' (normal) must return the stripped text as usual."""
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("text", text="import target\n")], stop_reason="end_turn")
    )
    assert llm.complete("sys", "user") == "import target"


def test_max_tokens_raises_truncated_error():
    """stop_reason='max_tokens' must raise TruncatedResponseError, never return text."""
    llm = _fake_llm()
    partial = "import target\ndef test_foo():\n    # response was cut off here"
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response(
            [_make_block("text", text=partial)],
            stop_reason="max_tokens",
            output_tokens=8192,
        )
    )
    with pytest.raises(TruncatedResponseError) as exc_info:
        llm.complete("sys", "user")
    assert exc_info.value.partial == partial
    assert exc_info.value.output_tokens == 8192


def test_truncated_error_tokens_are_already_accounted():
    """Token counts must be updated even when TruncatedResponseError is raised,
    so budget tracking reflects the actual cost of the aborted call."""
    llm = _fake_llm()
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response(
            [_make_block("text", text="partial")],
            stop_reason="max_tokens",
            input_tokens=500,
            output_tokens=8192,
        )
    )
    with pytest.raises(TruncatedResponseError):
        llm.complete("sys", "user")
    assert llm.input_tokens == 500
    assert llm.output_tokens == 8192


def test_default_max_tokens_is_at_least_16384():
    """DEFAULT_MAX_TOKENS must be >= 16 384 — the value needed for a full natsort run."""
    assert DEFAULT_MAX_TOKENS >= 16384


def test_llm_constructor_accepts_max_tokens():
    """LLM stores max_tokens and passes it to the streaming API call."""
    llm = _fake_llm()
    llm.max_tokens = 32768  # non-default value
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("text", text="import target")], stop_reason="end_turn")
    )
    llm.complete("sys", "user")
    call_kwargs = llm._client.messages.stream.call_args
    assert call_kwargs.kwargs["max_tokens"] == 32768


def test_complete_max_tokens_override():
    """Passing max_tokens to complete() overrides self.max_tokens for that call only."""
    llm = _fake_llm()
    llm.max_tokens = 8192  # self.max_tokens is lower
    llm._client.messages.stream.return_value = _fake_stream(
        _fake_response([_make_block("text", text="import target")], stop_reason="end_turn")
    )
    llm.complete("sys", "user", max_tokens=32768)
    call_kwargs = llm._client.messages.stream.call_args
    assert call_kwargs.kwargs["max_tokens"] == 32768
    # self.max_tokens must be unchanged
    assert llm.max_tokens == 8192
