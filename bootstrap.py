"""
Pre-flight script. Run by the Procfile before streamlit starts.

Writes .streamlit/secrets.toml from environment variables so that st.login()
can pick up OAuth config. Keeps secrets out of git: at build time the file
doesn't exist, at runtime it's generated from Railway env vars.

Required env vars (only when REQUIRE_AUTH=true):
    GOOGLE_CLIENT_ID         OAuth 2.0 Client ID
    GOOGLE_CLIENT_SECRET     OAuth 2.0 Client secret
    AUTH_REDIRECT_URI        e.g. https://<your-app>.up.railway.app/oauth2callback
    AUTH_COOKIE_SECRET       any random 32+ char string

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

    required = {
        "GOOGLE_CLIENT_ID": os.environ.get("GOOGLE_CLIENT_ID"),
        "GOOGLE_CLIENT_SECRET": os.environ.get("GOOGLE_CLIENT_SECRET"),
        "AUTH_REDIRECT_URI": os.environ.get("AUTH_REDIRECT_URI"),
        "AUTH_COOKIE_SECRET": os.environ.get("AUTH_COOKIE_SECRET"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        # Don't crash — the app itself will display a friendly error.
        print(f"[bootstrap] WARNING: missing env vars: {', '.join(missing)}. "
              "App will show an auth error until these are set.")
        return

    secrets_dir = Path(".streamlit")
    secrets_dir.mkdir(exist_ok=True)
    secrets_path = secrets_dir / "secrets.toml"

    # Use json.dumps to safely quote any awkward characters in the values.
    lines = [
        "[auth]",
        f"redirect_uri = {json.dumps(required['AUTH_REDIRECT_URI'])}",
        f"cookie_secret = {json.dumps(required['AUTH_COOKIE_SECRET'])}",
        f"client_id = {json.dumps(required['GOOGLE_CLIENT_ID'])}",
        f"client_secret = {json.dumps(required['GOOGLE_CLIENT_SECRET'])}",
        'server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"',
        # Restricts the Google account picker to MrQ Workspace if user already has cookies.
        # The app still server-side checks the email domain regardless.
        f'client_kwargs = {{"hd": "{os.environ.get("OAUTH_HOSTED_DOMAIN", "mrq.com")}"}}',
        "",
    ]
    secrets_path.write_text("\n".join(lines))
    # Lock down permissions in case the host filesystem is shared.
    try:
        secrets_path.chmod(0o600)
    except Exception:
        pass
    print(f"[bootstrap] Wrote {secrets_path} from env vars")


if __name__ == "__main__":
    main()
