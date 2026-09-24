"""Regression tests for #66887 — the live cached agent must survive a compression rotation.

Bug
---
The agent-cache tuple records ``(agent, sig, message_count, session_id)`` where ``session_id`` is
the id the snapshot was taken for at agent-BUILD time. It is never rewritten while the agent is
reused. When compression rotates the session mid-turn the parent row is ENDED, the live ``AIAgent``
is rebound to the child id, and the gateway's post-run split sync repoints routing at the child —
but the cache tuple still names the (now ended) parent.

On the next turn ``TurnRunner._lookup_cached_agent`` sees exactly the #54878 x #54947 shape
(snapshot id != current id AND snapshot id ended in state.db) and discards the cached agent. That
agent is the live conversation: anything it has not flushed to state.db yet is gone, and the rebuilt
agent starts from the persisted rows only. Observed live on 2026-09-23 (slack C0BDACDEPN0, ``main``
profile): context lost twice inside 25 minutes, each loss preceded by
``Session split detected: <parent> → <child> (compression)`` one turn earlier.

Fix (these tests pin it)
------------------------
Ask the AGENT, not the snapshot, who owns the current session. When the cached agent's own
``session_id`` already equals the routed one, the mismatch is a rotation artifact: re-baseline the
cache tuple (id *and* count — the recorded count tracked the ended parent row) and reuse the live
agent. The genuine #54878 case — the agent itself still bound to the dead session — keeps
discarding, so the self-heal-undo loop that fix guards against cannot come back.
"""

import threading
from types import SimpleNamespace

from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from hermes_state import SessionDB


SESSION_KEY = "agent:main:slack:group:T0EAWEMM3:C0BDACDEPN0"
SIG = "sig-v1"


class _Agent:
    """Enough of AIAgent for the cache guard and the live-history preference."""

    def __init__(self, session_id, messages):
        self.session_id = session_id
        self._session_messages = list(messages)
        self.max_iterations = 1
        self._last_flushed_db_idx = 7


class _SessionStore:
    def __init__(self, db):
        self._db = db

    def _is_session_ended_in_db(self, session_id):
        row = self._db.get_session(session_id) if session_id else None
        return bool(row is not None and row.get("end_reason") is not None)


def _turn_runner(tmp_path, *, cached_sid, agent_sid, routed_sid, persisted_history=()):
    db = SessionDB(db_path=tmp_path / "state.db")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = _SessionStore(db)
    # ``_current_message_count`` reads ``runner._session_db._db.get_session``.
    runner._session_db = SimpleNamespace(_db=db)
    agent = _Agent(agent_sid, [
        {"role": "user", "content": "1. Yes look into it."},
        {"role": "assistant", "content": "on it"},
    ])
    runner._agent_cache[SESSION_KEY] = (agent, SIG, 41, cached_sid)
    ctx = SimpleNamespace(
        session_key=SESSION_KEY, session_id=routed_sid, _interrupt_depth=0,
        history=list(persisted_history), channel_prompt=None, user_config={},
    )
    return TurnRunner(runner, ctx), runner, db, agent


def _lookup(tr, runner):
    """Mirror ``_resolve_turn_agent``'s call sequence for the cache-hit decision."""
    peek_sid, dead = tr._cached_sid_is_dead(runner._agent_cache_lock, runner._agent_cache)
    msg_count = tr._current_message_count()
    return tr._lookup_cached_agent(
        SIG, runner._agent_cache_lock, runner._agent_cache, 25, peek_sid, dead, msg_count)


class TestCompressionRotationKeepsLiveAgent:
    def test_rotated_agent_is_reused_and_snapshot_rebaselined(self, tmp_path):
        """#66887: parent ended by compression, live agent already on the child → reuse."""
        tr, runner, db, agent = _turn_runner(
            tmp_path, cached_sid="20260923_223223_d03383", agent_sid="20260923_224809_25aeae",
            routed_sid="20260923_224809_25aeae")
        db.create_session("20260923_223223_d03383", source="slack")
        db.end_session("20260923_223223_d03383", "compression")
        db.create_session("20260923_224809_25aeae", source="slack")

        found = _lookup(tr, runner)

        assert found.reused is True, (
            "BUG #66887: the live agent was discarded over a stale snapshot id — every turn it had "
            "not flushed to state.db is lost.")
        assert found.agent is agent
        assert found.evicted is None
        # Snapshot re-baselined onto the live row: id AND count, or the cross-process guard evicts
        # this same live agent on the very next turn.
        cached = runner._agent_cache[SESSION_KEY]
        assert cached[0] is agent
        assert cached[3] == "20260923_224809_25aeae"
        assert cached[2] == db.get_session("20260923_224809_25aeae").get("message_count", 0)

    def test_live_turns_survive_a_lagging_persisted_transcript(self, tmp_path):
        """The point of reusing: the agent's unflushed rows still reach the next turn's history."""
        tr, runner, db, agent = _turn_runner(
            tmp_path, cached_sid="parent_sid", agent_sid="child_sid", routed_sid="child_sid",
            persisted_history=[{"role": "user", "content": "1. Yes look into it."}])
        db.create_session("parent_sid", source="slack")
        db.end_session("parent_sid", "compression")
        db.create_session("child_sid", source="slack")

        found = _lookup(tr, runner)
        agent_history, _observed, _media = tr._load_turn_history(found.agent, found.reused)

        assert [m["content"] for m in agent_history] == [
            "1. Yes look into it.", "on it"], (
            "The live agent's unflushed turn did not survive into the next turn's history.")

    def test_genuine_stale_self_heal_still_discards(self, tmp_path):
        """#54878 x #54947 invariant: an agent still bound to the DEAD session is evicted.

        Reusing it lets the post-run split sync write routing back onto the dead id, undoing the
        self-heal and looping on every message.
        """
        tr, runner, db, agent = _turn_runner(
            tmp_path, cached_sid="dead_sid", agent_sid="dead_sid", routed_sid="fresh_sid_after_selfheal")
        db.create_session("dead_sid", source="slack")
        db.end_session("dead_sid", "user_requested")
        db.create_session("fresh_sid_after_selfheal", source="slack")

        found = _lookup(tr, runner)

        assert found.reused is False, "Regression: the #54878 x #54947 self-heal-undo loop is back."
        assert found.agent is None
        assert found.evicted is agent
        assert SESSION_KEY not in runner._agent_cache

    def test_live_sibling_switch_still_reuses(self, tmp_path):
        """#54947 invariant: neither session ended → reuse, unchanged by the rotation branch."""
        tr, runner, db, agent = _turn_runner(
            tmp_path, cached_sid="sA", agent_sid="sA", routed_sid="sB")
        db.create_session("sA", source="slack")
        db.create_session("sB", source="slack")

        found = _lookup(tr, runner)

        assert found.reused is True
        assert found.agent is agent
        # Untouched: a live sibling's snapshot must keep ITS own baseline (#54947).
        assert runner._agent_cache[SESSION_KEY][3] == "sA"
