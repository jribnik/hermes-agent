"""Tests for the Anthropic Claude Code OAuth auth handler
(providers/auth/anthropic_oauth.py) and its registry wiring in
hermes_cli/providers.py.

Covers: is_claude_code_oauth_auth_type, claude_code_credentials_path,
has_claude_code_credentials, _access_token_is_expiring,
resolve_anthropic_oauth_runtime_credentials, get_anthropic_oauth_auth_status,
get_oauth_auth_handler, normalize_auth_type, and the sparse
providers.anthropic user-config override in resolve_user_provider.
"""

import time
from pathlib import Path

import pytest

from providers.auth.anthropic_oauth import (
    ANTHROPIC_OAUTH_REFRESH_SKEW_SECONDS,
    AUTH_TYPE_OAUTH_CLAUDE_CODE,
    DEFAULT_ANTHROPIC_BASE_URL,
    _access_token_is_expiring,
    claude_code_credentials_path,
    get_anthropic_oauth_auth_status,
    has_claude_code_credentials,
    is_claude_code_oauth_auth_type,
    resolve_anthropic_oauth_runtime_credentials,
)
from hermes_cli.auth import AuthError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claude_creds(
    access_token="test-access-token",
    refresh_token="test-refresh-token",
    expires_at=None,
    **extra,
):
    """Create a minimal Claude Code OAuth credential dict."""
    if expires_at is None:
        # 1 hour from now in milliseconds
        expires_at = int((time.time() + 3600) * 1000)
    data = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at,
        "source": "claude_code",
    }
    data.update(extra)
    return data


@pytest.fixture()
def adapter(monkeypatch):
    """Patchable stand-ins for the lazily imported agent.anthropic_credentials
    primitives. Defaults: no credentials, refresh fails, token invalid."""
    import agent.anthropic_credentials as adapter_mod

    state = {
        "creds": None,
        "refreshed_token": None,
        "token_valid": False,
        "refresh_calls": [],
        "read_calls": 0,
    }

    def read_claude_code_credentials():
        state["read_calls"] += 1
        return state["creds"]

    def _refresh_oauth_token(creds):
        state["refresh_calls"].append(creds)
        return state["refreshed_token"]

    def is_claude_code_token_valid(creds):
        return state["token_valid"]

    monkeypatch.setattr(
        adapter_mod, "read_claude_code_credentials", read_claude_code_credentials
    )
    monkeypatch.setattr(adapter_mod, "_refresh_oauth_token", _refresh_oauth_token)
    monkeypatch.setattr(
        adapter_mod, "is_claude_code_token_valid", is_claude_code_token_valid
    )
    return state


# ---------------------------------------------------------------------------
# is_claude_code_oauth_auth_type
# ---------------------------------------------------------------------------

def test_auth_type_canonical_value():
    assert is_claude_code_oauth_auth_type(AUTH_TYPE_OAUTH_CLAUDE_CODE)


@pytest.mark.parametrize("value", [
    "oauth_claude_code",
    "claude_code",
    "claude_code_oauth",
    "oauth-claude-code",
    "claude-code",
    "  OAuth_Claude_Code  ",
    "CLAUDE_CODE",
])
def test_auth_type_accepted_spellings(value):
    assert is_claude_code_oauth_auth_type(value)


@pytest.mark.parametrize("value", [
    None,
    "",
    "api_key",
    "oauth_external",
    "oauth_device_code",
    "claude",
    "codex",
    0,
])
def test_auth_type_rejected_values(value):
    assert not is_claude_code_oauth_auth_type(value)


# ---------------------------------------------------------------------------
# claude_code_credentials_path
# ---------------------------------------------------------------------------

def test_credentials_path_returns_expected_location():
    path = claude_code_credentials_path()
    assert path == Path.home() / ".claude" / ".credentials.json"


# ---------------------------------------------------------------------------
# has_claude_code_credentials
# ---------------------------------------------------------------------------

def test_has_credentials_true_when_record_exists(adapter):
    adapter["creds"] = _make_claude_creds()
    assert has_claude_code_credentials() is True


def test_has_credentials_false_when_missing(adapter):
    adapter["creds"] = None
    assert has_claude_code_credentials() is False


def test_has_credentials_never_raises(monkeypatch):
    import agent.anthropic_credentials as adapter_mod

    def boom():
        raise RuntimeError("keychain locked")

    monkeypatch.setattr(adapter_mod, "read_claude_code_credentials", boom)
    assert has_claude_code_credentials() is False


# ---------------------------------------------------------------------------
# _access_token_is_expiring
# ---------------------------------------------------------------------------

def test_expiring_absent_expiry_never_expires():
    assert _access_token_is_expiring(None) is False
    assert _access_token_is_expiring(0) is False
    assert _access_token_is_expiring("") is False


def test_expiring_unparseable_treated_as_expiring():
    assert _access_token_is_expiring("not-a-number") is True
    assert _access_token_is_expiring(object()) is True


def test_expiring_future_token_not_expiring():
    future_ms = int((time.time() + 3600) * 1000)
    assert _access_token_is_expiring(future_ms) is False


def test_expiring_past_token_is_expiring():
    past_ms = int((time.time() - 60) * 1000)
    assert _access_token_is_expiring(past_ms) is True


def test_expiring_within_skew_window():
    inside_skew_ms = int(
        (time.time() + ANTHROPIC_OAUTH_REFRESH_SKEW_SECONDS - 30) * 1000
    )
    assert _access_token_is_expiring(inside_skew_ms) is True


def test_expiring_custom_skew():
    soon_ms = int((time.time() + 60) * 1000)
    assert _access_token_is_expiring(soon_ms, skew_seconds=0) is False
    assert _access_token_is_expiring(soon_ms, skew_seconds=120) is True


def test_expiring_negative_skew_clamped_to_zero():
    soon_ms = int((time.time() + 60) * 1000)
    assert _access_token_is_expiring(soon_ms, skew_seconds=-9999) is False


# ---------------------------------------------------------------------------
# resolve_anthropic_oauth_runtime_credentials
# ---------------------------------------------------------------------------

def test_resolve_missing_credentials_raises(adapter):
    adapter["creds"] = None
    with pytest.raises(AuthError) as exc:
        resolve_anthropic_oauth_runtime_credentials()
    assert exc.value.code == "claude_code_auth_missing"
    assert exc.value.provider == "anthropic"
    assert exc.value.relogin_required is True


def test_resolve_valid_token_no_refresh(adapter):
    expires_at = int((time.time() + 3600) * 1000)
    adapter["creds"] = _make_claude_creds(
        access_token="fresh-token", expires_at=expires_at
    )
    result = resolve_anthropic_oauth_runtime_credentials()
    assert result["provider"] == "anthropic"
    assert result["base_url"] == DEFAULT_ANTHROPIC_BASE_URL
    assert result["api_key"] == "fresh-token"
    assert result["source"] == "claude_code"
    assert result["expires_at_ms"] == expires_at
    assert result["auth_file"] == str(claude_code_credentials_path())
    assert adapter["refresh_calls"] == []


def test_resolve_expiring_token_refreshes(adapter):
    adapter["creds"] = _make_claude_creds(
        access_token="stale-token",
        expires_at=int((time.time() - 60) * 1000),
    )
    adapter["refreshed_token"] = "rotated-token"
    result = resolve_anthropic_oauth_runtime_credentials()
    assert result["api_key"] == "rotated-token"
    assert len(adapter["refresh_calls"]) == 1


def test_resolve_refresh_rereads_rotated_credentials(adapter):
    stale = _make_claude_creds(
        access_token="stale-token",
        expires_at=int((time.time() - 60) * 1000),
    )
    rotated_expiry = int((time.time() + 7200) * 1000)
    adapter["creds"] = stale
    adapter["refreshed_token"] = "rotated-token"

    import agent.anthropic_credentials as adapter_mod

    def refresh_and_rotate(creds):
        adapter["refresh_calls"].append(creds)
        adapter["creds"] = _make_claude_creds(
            access_token="rotated-token", expires_at=rotated_expiry
        )
        return "rotated-token"

    adapter_mod._refresh_oauth_token = refresh_and_rotate
    result = resolve_anthropic_oauth_runtime_credentials()
    assert result["api_key"] == "rotated-token"
    # Reported expiry must match the token returned, not the stale record.
    assert result["expires_at_ms"] == rotated_expiry


def test_resolve_force_refresh_with_valid_token(adapter):
    adapter["creds"] = _make_claude_creds(access_token="old-token")
    adapter["refreshed_token"] = "forced-new-token"
    result = resolve_anthropic_oauth_runtime_credentials(force_refresh=True)
    assert result["api_key"] == "forced-new-token"
    assert len(adapter["refresh_calls"]) == 1


def test_resolve_refresh_failed_but_token_still_valid_keeps_it(adapter):
    adapter["creds"] = _make_claude_creds(access_token="still-good")
    adapter["refreshed_token"] = None
    adapter["token_valid"] = True
    result = resolve_anthropic_oauth_runtime_credentials(force_refresh=True)
    assert result["api_key"] == "still-good"


def test_resolve_expired_beyond_refresh_raises(adapter):
    adapter["creds"] = _make_claude_creds(
        access_token="dead-token",
        expires_at=int((time.time() - 3600) * 1000),
    )
    adapter["refreshed_token"] = None
    adapter["token_valid"] = False
    with pytest.raises(AuthError) as exc:
        resolve_anthropic_oauth_runtime_credentials()
    assert exc.value.code == "claude_code_token_expired"
    assert exc.value.relogin_required is True


def test_resolve_missing_access_token_raises(adapter):
    adapter["creds"] = _make_claude_creds(access_token="")
    with pytest.raises(AuthError) as exc:
        resolve_anthropic_oauth_runtime_credentials()
    assert exc.value.code == "claude_code_access_token_missing"
    assert exc.value.relogin_required is True


def test_resolve_refresh_skipped_when_disabled(adapter):
    adapter["creds"] = _make_claude_creds(
        access_token="expiring-token",
        expires_at=int((time.time() + 30) * 1000),
    )
    result = resolve_anthropic_oauth_runtime_credentials(refresh_if_expiring=False)
    assert result["api_key"] == "expiring-token"
    assert adapter["refresh_calls"] == []


# ---------------------------------------------------------------------------
# get_anthropic_oauth_auth_status
# ---------------------------------------------------------------------------

def test_auth_status_logged_in(adapter):
    expires_at = int((time.time() + 3600) * 1000)
    adapter["creds"] = _make_claude_creds(
        access_token="status-token", expires_at=expires_at
    )
    status = get_anthropic_oauth_auth_status()
    assert status["logged_in"] is True
    assert status["api_key"] == "status-token"
    assert status["expires_at_ms"] == expires_at
    assert status["auth_file"] == str(claude_code_credentials_path())


def test_auth_status_logged_out_never_raises(adapter):
    adapter["creds"] = None
    status = get_anthropic_oauth_auth_status()
    assert status["logged_in"] is False
    assert "not found" in status["error"]
    assert status["auth_file"] == str(claude_code_credentials_path())


def test_auth_status_stale_unrefreshable_token_not_logged_in(adapter):
    adapter["creds"] = _make_claude_creds(
        access_token="dead-token",
        expires_at=int((time.time() - 3600) * 1000),
    )
    adapter["refreshed_token"] = None
    adapter["token_valid"] = False
    status = get_anthropic_oauth_auth_status()
    assert status["logged_in"] is False


# ---------------------------------------------------------------------------
# hermes_cli.providers — OAuth handler registry
# ---------------------------------------------------------------------------

def test_get_oauth_auth_handler_resolves_claude_code():
    from hermes_cli.providers import get_oauth_auth_handler

    handler = get_oauth_auth_handler("oauth_claude_code")
    assert handler is resolve_anthropic_oauth_runtime_credentials


@pytest.mark.parametrize("spelling", [
    "claude_code",
    "claude-code",
    "OAuth-Claude-Code",
    "claude_code_oauth",
])
def test_get_oauth_auth_handler_alias_spellings(spelling):
    from hermes_cli.providers import get_oauth_auth_handler

    assert get_oauth_auth_handler(spelling) is (
        resolve_anthropic_oauth_runtime_credentials
    )


@pytest.mark.parametrize("value", ["", None, "api_key", "oauth_external", "bogus"])
def test_get_oauth_auth_handler_unknown_returns_none(value):
    from hermes_cli.providers import get_oauth_auth_handler

    assert get_oauth_auth_handler(value) is None


# ---------------------------------------------------------------------------
# hermes_cli.providers.normalize_auth_type
# ---------------------------------------------------------------------------

def test_normalize_auth_type_claude_code_collapses_to_oauth_external():
    from hermes_cli.providers import normalize_auth_type

    assert normalize_auth_type("oauth_claude_code") == "oauth_external"
    assert normalize_auth_type("claude-code") == "oauth_external"


def test_normalize_auth_type_empty_uses_default():
    from hermes_cli.providers import normalize_auth_type

    assert normalize_auth_type("") == "api_key"
    assert normalize_auth_type(None) == "api_key"
    assert normalize_auth_type(None, default="oauth_external") == "oauth_external"


def test_normalize_auth_type_passthrough():
    from hermes_cli.providers import normalize_auth_type

    assert normalize_auth_type("api_key") == "api_key"
    assert normalize_auth_type("OAuth-Device-Code") == "oauth_device_code"


# ---------------------------------------------------------------------------
# hermes_cli.providers.resolve_user_provider — sparse anthropic override
# ---------------------------------------------------------------------------

def test_sparse_anthropic_auth_type_override_inherits_builtin():
    """``providers.anthropic.auth_type: oauth_claude_code`` alone must keep
    the built-in anthropic transport/base_url/env vars instead of degrading
    to the openai_chat + empty-url custom-provider defaults."""
    from hermes_cli.providers import get_provider, resolve_user_provider

    builtin = get_provider("anthropic")
    assert builtin is not None

    resolved = resolve_user_provider(
        "anthropic",
        {"anthropic": {"auth_type": "oauth_claude_code"}},
    )
    assert resolved is not None
    assert resolved.auth_type == "oauth_external"
    assert resolved.transport == builtin.transport
    assert resolved.base_url == builtin.base_url
    assert resolved.api_key_env_vars == builtin.api_key_env_vars
    assert resolved.source == "user-config"


def test_user_provider_explicit_fields_still_win():
    from hermes_cli.providers import resolve_user_provider

    resolved = resolve_user_provider(
        "anthropic",
        {
            "anthropic": {
                "auth_type": "oauth_claude_code",
                "base_url": "https://proxy.example.com",
                "key_env": "MY_ANTHROPIC_KEY",
            }
        },
    )
    assert resolved is not None
    assert resolved.base_url == "https://proxy.example.com"
    assert resolved.api_key_env_vars == ("MY_ANTHROPIC_KEY",)


def test_alias_entry_does_not_inherit_builtin():
    """An alias name ("openai" → openrouter) is not a canonical provider —
    it must keep the plain custom-provider defaults, not hijack openrouter's
    definition."""
    from hermes_cli.providers import resolve_user_provider

    resolved = resolve_user_provider(
        "openai",
        {"openai": {"api": "https://example.com/v1"}},
    )
    assert resolved is not None
    assert resolved.transport == "openai_chat"
    assert resolved.base_url == "https://example.com/v1"
    assert resolved.auth_type == "api_key"
    assert resolved.is_aggregator is False
