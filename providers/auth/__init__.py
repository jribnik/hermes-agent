"""Auth handlers for provider profiles with non-API-key ``auth_type``.

Each module implements the runtime credential interface used by
``hermes_cli.runtime_provider`` for ``oauth_external`` providers:

  - ``resolve_<provider>_runtime_credentials(...) -> dict``
      Returns ``{"provider", "base_url", "api_key", "source",
      "expires_at_ms", "auth_file"}``. Raises ``hermes_cli.auth.AuthError``
      when credentials are missing or expired beyond refresh.
  - ``get_<provider>_auth_status() -> dict``
      Never raises — reports ``{"logged_in": bool, ...}`` for doctor/status.

Modules here must stay import-light (stdlib only at module level): the
``providers`` package is imported early and this subpackage is picked up by
its legacy ``pkgutil`` discovery scan. Import ``hermes_cli`` / ``agent``
lazily inside functions.
"""
