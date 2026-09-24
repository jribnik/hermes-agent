"""Anthropic OAuth auth handler — Claude Code / Claude Pro credentials.

Lets Hermes call Anthropic models (Fable, Opus, Sonnet) on a Claude Pro /
Max subscription without an API key, by reusing the OAuth token that the
Claude CLI manages at ``~/.claude/.credentials.json`` (and, on macOS
>= Claude Code 2.1.114, the "Claude Code-credentials" Keychain entry).

This module is the ``oauth_external`` runtime-credential handler for the
``anthropic`` provider, selected via::

    hermes config set providers.anthropic.auth_type oauth_claude_code

Token reading, refresh (single-use rotating refresh tokens, racing Claude
Code's own refresher), and credential persistence are owned by
``agent.anthropic_credentials`` — this handler wraps those primitives in the
``resolve_*_runtime_credentials`` interface that ``runtime_provider``
expects for OAuth providers (see ``resolve_qwen_runtime_credentials``).

Module-level imports must stay stdlib-only: the ``providers`` package is
imported early (plugin discovery), so ``agent`` / ``hermes_cli`` are
imported lazily inside functions to avoid layer cycles.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# Refresh 2 min before expiry — matches ACCESS_TOKEN_REFRESH_SKEW_SECONDS
# for the other short-lived OAuth providers (Qwen, Codex).
ANTHROPIC_OAUTH_REFRESH_SKEW_SECONDS = 120

# Canonical config value for ``providers.anthropic.auth_type``. Accepted
# spellings are normalized by is_claude_code_oauth_auth_type().
AUTH_TYPE_OAUTH_CLAUDE_CODE = "oauth_claude_code"

_CLAUDE_CODE_AUTH_TYPE_ALIASES = frozenset({
    "oauth_claude_code",
    "claude_code",
    "claude_code_oauth",
})


def is_claude_code_oauth_auth_type(value: Any) -> bool:
    """True when a config ``auth_type`` value selects Claude Code OAuth.

    Accepts dash/underscore spellings (``oauth-claude-code`` etc.) so
    hand-edited configs don't silently fall back to API-key auth.
    """
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized in _CLAUDE_CODE_AUTH_TYPE_ALIASES


def claude_code_credentials_path() -> Path:
    """Path of the Claude CLI's OAuth credential file."""
    return Path.home() / ".claude" / ".credentials.json"


def has_claude_code_credentials() -> bool:
    """True when a Claude Code OAuth credential record exists (file or
    Keychain). Never raises — probe for setup wizards and pickers."""
    try:
        from agent.anthropic_credentials import read_claude_code_credentials

        return read_claude_code_credentials() is not None
    except Exception as exc:
        logger.debug("Claude Code credential probe failed: %s", exc)
        return False


def _access_token_is_expiring(
    expires_at_ms: Any,
    skew_seconds: int = ANTHROPIC_OAUTH_REFRESH_SKEW_SECONDS,
) -> bool:
    """True when the token expires within *skew_seconds*.

    ``expiresAt`` of 0/absent means "no expiry" (managed keys) — never
    expiring. Unparseable values are treated as expiring so the refresh
    path gets a chance to replace them.
    """
    if not expires_at_ms:
        return False
    try:
        expiry_ms = int(expires_at_ms)
    except (TypeError, ValueError):
        return True
    return (time.time() + max(0, int(skew_seconds))) * 1000 >= expiry_ms


def resolve_anthropic_oauth_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = ANTHROPIC_OAUTH_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """Resolve a Claude Code OAuth token for Anthropic inference.

    Reads the credential record Claude Code maintains, refreshing through
    the Anthropic OAuth token endpoints when the access token is expired
    or expiring. Raises :class:`hermes_cli.auth.AuthError` when no
    credentials exist or they are expired beyond refresh — callers fall
    back or surface the relogin hint.
    """
    from agent.anthropic_credentials import (
        _refresh_oauth_token,
        is_claude_code_token_valid,
        read_claude_code_credentials,
    )
    from hermes_cli.auth import AuthError

    creds = read_claude_code_credentials()
    if not creds:
        raise AuthError(
            "Claude Code credentials not found. Run 'claude /login' (or "
            "'claude setup-token') first, or set ANTHROPIC_API_KEY to use "
            "API-key auth instead.",
            provider="anthropic",
            code="claude_code_auth_missing",
            relogin_required=True,
        )

    access_token = str(creds.get("accessToken") or "").strip()
    should_refresh = bool(force_refresh)
    if not should_refresh and refresh_if_expiring:
        should_refresh = _access_token_is_expiring(
            creds.get("expiresAt"), refresh_skew_seconds
        )

    if should_refresh:
        refreshed = _refresh_oauth_token(creds)
        if refreshed:
            access_token = refreshed
            # Refresh rotated the stored pair — re-read so the reported
            # expiry matches the token we return.
            creds = read_claude_code_credentials() or creds
        elif not is_claude_code_token_valid(creds):
            raise AuthError(
                "Claude Code OAuth token is expired and could not be "
                "refreshed. Re-authenticate with 'claude /login'.",
                provider="anthropic",
                code="claude_code_token_expired",
                relogin_required=True,
            )
        # else: refresh failed but the current token is still valid —
        # keep serving it and let a later call retry the refresh.

    if not access_token:
        raise AuthError(
            "Claude Code credentials are missing an access token. "
            "Re-authenticate with 'claude /login'.",
            provider="anthropic",
            code="claude_code_access_token_missing",
            relogin_required=True,
        )

    return {
        "provider": "anthropic",
        "base_url": DEFAULT_ANTHROPIC_BASE_URL,
        "api_key": access_token,
        "source": creds.get("source") or "claude_code",
        "expires_at_ms": creds.get("expiresAt") or None,
        "auth_file": str(claude_code_credentials_path()),
    }


def get_anthropic_oauth_auth_status() -> Dict[str, Any]:
    """Status summary for doctor/status commands. Never raises.

    Validates the runtime credentials including refresh-on-expiry, so a
    stale token doesn't show up as "logged in" (same contract as
    ``get_qwen_auth_status``).
    """
    auth_path = claude_code_credentials_path()
    try:
        creds = resolve_anthropic_oauth_runtime_credentials(refresh_if_expiring=True)
        return {
            "logged_in": True,
            "auth_file": str(auth_path),
            "source": creds.get("source"),
            "api_key": creds.get("api_key"),
            "expires_at_ms": creds.get("expires_at_ms"),
        }
    except Exception as exc:
        return {
            "logged_in": False,
            "auth_file": str(auth_path),
            "error": str(exc),
        }
