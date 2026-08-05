"""Regression: AIAgent.close() must shut down the context engine.

close() tears down processes, sandboxes, the browser daemon, the httpx client
and the session row — but historically never called the context engine's
teardown. A plugin context engine that holds OS resources (e.g. open SQLite
connections) then leaked them on every agent teardown, reclaimed only by
GC-``__del__`` — which does not run promptly under a long-lived gateway that
caches and evicts many agents — until the process hit EMFILE ("too many open
files").

These tests fail if the ``shutdown()`` call (or its ``_end_session_on_close``
guard) is removed.
"""

from typing import Any, Dict

import pytest

import run_agent
from agent.context_engine import ContextEngine
from run_agent import AIAgent


class _RecordingEngine(ContextEngine):
    """Minimal concrete ContextEngine that records shutdown() calls."""

    def __init__(self) -> None:
        self.shutdown_calls = 0

    @property
    def name(self) -> str:
        return "recording"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def neutralize_os_teardown(monkeypatch):
    """Stub the OS-level teardown close() performs so the test stays hermetic."""
    from tools.process_registry import process_registry

    monkeypatch.setattr(process_registry, "kill_all", lambda **kw: None)
    monkeypatch.setattr(run_agent, "cleanup_vm", lambda *a, **k: None)
    monkeypatch.setattr(run_agent, "cleanup_browser", lambda *a, **k: None)


def _bare_agent(engine, *, end_session_on_close=True):
    """An AIAgent skeleton carrying only what close()'s engine step reads.

    Every close() step is independently guarded, so the unset attributes on a
    __new__ instance no-op safely; we set just the two the engine step reads.
    """
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "ctx-engine-shutdown-test"
    agent.context_compressor = engine
    agent._end_session_on_close = end_session_on_close
    return agent


def test_close_shuts_down_context_engine(neutralize_os_teardown):
    engine = _RecordingEngine()
    AIAgent.close(_bare_agent(engine))
    assert engine.shutdown_calls == 1


def test_close_does_not_shut_down_shared_parent_engine(neutralize_os_teardown):
    # Compression helpers / background-review forks that hand session ownership
    # forward set _end_session_on_close=False; they must not tear down an engine
    # that is still live on the parent agent.
    engine = _RecordingEngine()
    AIAgent.close(_bare_agent(engine, end_session_on_close=False))
    assert engine.shutdown_calls == 0


def test_context_engine_base_shutdown_is_a_noop():
    # The default hook exists on the base and is safe to call on an engine that
    # holds no resources (the built-in compressor).
    engine = _RecordingEngine()
    ContextEngine.shutdown(engine)  # base default must not raise...
    assert engine.shutdown_calls == 0  # ...and must not touch subclass state
