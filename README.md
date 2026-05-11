# MrQ Attribution Liveboard v2

Internal performance marketing analytics tool. Streamlit app over BigQuery,
designed for Performance Marketing Managers and the Head of Performance
Marketing.

Six pages: Executive Overview, Channel View, Campaign / Ad Explorer,
Data Quality / Attribution Health, Recommendations, Metric Dictionary.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then either upload a CSV or point the sidebar at BigQuery. For BQ locally,
set the path to your service account key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/mrq/bq-service-account.json"
```

## Deploy to Railway

1. Push this repo to GitHub (private).
2. In Railway, **New Project → Deploy from GitHub repo**, pick this one.
3. Set the env vars from `.env.example` in Railway's **Variables** tab.
4. Once Railway assigns a URL, set `AUTH_REDIRECT_URI` to
   `https://<your-railway-domain>/oauth2callback` and add that same URL
   under **Authorized redirect URIs** in the Google Cloud OAuth client.
5. Redeploy. Visit the URL — you should be prompted to sign in with Google
   and rejected unless your email ends with `@mrq.com`.

## Files

- `app.py` — the Streamlit application.
- `bootstrap.py` — pre-flight script that writes `.streamlit/secrets.toml`
  from env vars before streamlit starts.
- `Procfile` — Railway's run command.
- `runtime.txt` — Python version pin.
- `requirements.txt` — Python dependencies.
- `.env.example` — env-var template (no secrets).
- `.gitignore` — keeps keys, data files, secrets, venv out of git.

## Data treatment

The dataset mixes grains via `spend_attribution_approach`: `ad_level`,
`ad_group_level`, `campaign_level`, `channel_level`, `affiliate_level`. These
are alternative breakdowns, not nested. `spend_row_type`
(matched / spend_only / residual) only populates at the granular grains;
higher grains are already aggregated. By default, efficiency metrics (CPA,
LTV:CAC, CPA APD2+) restrict to matched rows on granular grains; toggleable
in the sidebar.

Recommendations are rules-based with transparent reasons and a confidence
label based on spend × FTD volume. They are a triage list, not a directive.

Period comparison defaults to "same window in previous month".
