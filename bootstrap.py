"""
Pre-flight script. Run by the Procfile before streamlit starts.

Writes .streamlit/secrets.toml from environment variables so that st.login()
can pick up OAuth config. Keeps secrets out of git: at build time the file
doesn't exist, at runtime it's generated from Railway env vars.

Required env vars (only when REQUIRE_AUTH=true).

Auth0 path (MrQ standard — preferred):
    AUTH0_DOMAIN             e.g. mrq-vibes.uk.auth0.com
    AUTH0_CLIENT_ID          from the SSO panel on the app detail page
    AUTH0_CLIENT_SECRET      from the SSO panel on the app detail page
    AUTH_REDIRECT_URI        e.g. https://<your-app>.up.railway.app/oauth2callback
    AUTH_COOKIE_SECRET       any random 32+ char string

Google path (fallback, for non-MrQ deploys):
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    AUTH_REDIRECT_URI
    AUTH_COOKIE_SECRET

Optional:
    ALLOWED_EMAIL_DOMAINS    comma-separated, default mrq.com
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    require_auth = os.environ.get(
        "REQUIRE_AUTH",
        "true" if os.environ.get("RAILWAY_ENVIRONMENT") else "false",
    ).lower() in ("1", "true", "yes")

    if not require_auth:
        print("[bootstrap] REQUIRE_AUTH=false — skipping auth config write")
        return

    # Preferred: Auth0 (MrQ standard SSO).
    auth0_domain = os.environ.get("AUTH0_DOMAIN", "").strip()
    auth0_client_id = os.environ.get("AUTH0_CLIENT_ID", "").strip()
    auth0_client_secret = os.environ.get("AUTH0_CLIENT_SECRET", "").strip()
    # Fallback: raw Google OIDC (for local or non-MrQ deployments).
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()

    redirect_uri = os.environ.get("AUTH_REDIRECT_URI", "").strip()
    cookie_secret = os.environ.get("AUTH_COOKIE_SECRET", "").strip()

    if auth0_domain and auth0_client_id and auth0_client_secret and redirect_uri and cookie_secret:
        provider = "auth0"
        client_id = auth0_client_id
        client_secret = auth0_client_secret
        server_metadata_url = f"https://{auth0_domain}/.well-known/openid-configuration"
    elif google_client_id and google_client_secret and redirect_uri and cookie_secret:
        provider = "google"
        client_id = google_client_id
        client_secret = google_client_secret
        server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
    else:
        # Neither provider is fully configured. Don't crash — the app itself
        # will display a friendly error if REQUIRE_AUTH=true.
        print(
            "[bootstrap] WARNING: no auth provider configured. "
            "For Auth0 set AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, "
            "AUTH_REDIRECT_URI, AUTH_COOKIE_SECRET. "
            "App will show an auth error if REQUIRE_AUTH=true."
        )
        return

    secrets_dir = Path(".streamlit")
    secrets_dir.mkdir(exist_ok=True)
    secrets_path = secrets_dir / "secrets.toml"

    # Use json.dumps to safely quote any awkward characters in the values.
    lines = [
        "[auth]",
        f"redirect_uri = {json.dumps(redirect_uri)}",
        f"cookie_secret = {json.dumps(cookie_secret)}",
        f"client_id = {json.dumps(client_id)}",
        f"client_secret = {json.dumps(client_secret)}",
        f"server_metadata_url = {json.dumps(server_metadata_url)}",
        'client_kwargs = {"scope" = "openid email profile"}',
    ]
    if provider == "google":
        # Google account picker hint — restricts to Workspace if cookies present.
        # The app server-side checks the email domain regardless.
        lines.append(
            f'# Google-only hint (no effect on Auth0)\n'
            f'# client_kwargs.hd = "{os.environ.get("OAUTH_HOSTED_DOMAIN", "mrq.com")}"'
        )
    lines.append("")
    secrets_path.write_text("\n".join(lines))
    # Lock down permissions in case the host filesystem is shared.
    try:
        secrets_path.chmod(0o600)
    except Exception:
        pass
    print(f"[bootstrap] Wrote {secrets_path} (provider: {provider})")


if __name__ == "__main__":
    main()
