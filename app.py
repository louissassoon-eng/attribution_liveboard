"""
MrQ Attribution Liveboard v2 — internal performance marketing analytics.

Single-file Streamlit app. Loads an attribution + spend CSV, exposes six pages:
Executive Overview, Channel View, Campaign / Ad Explorer, Data Quality,
Recommendations, Metric Dictionary. Designed for Performance Marketing
Managers and the Head of Performance Marketing.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Data treatment notes are in the Metric Dictionary tab and summarised in the
README block at the bottom of this file.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

APP_TITLE = "MrQ Attribution Liveboard v2"
CURRENCY = "£"

# Allowed email domains for Google auth. Comma-separated env var overrides this.
ALLOWED_EMAIL_DOMAINS = [
    d.strip().lower()
    for d in os.environ.get("ALLOWED_EMAIL_DOMAINS", "mrq.com").split(",")
    if d.strip()
]

# Require Google login. Defaults to true if running on Railway (auto-detected
# via RAILWAY_ENVIRONMENT), false locally. Override with REQUIRE_AUTH=true|false.
REQUIRE_AUTH = os.environ.get(
    "REQUIRE_AUTH",
    "true" if os.environ.get("RAILWAY_ENVIRONMENT") else "false",
).lower() in ("1", "true", "yes")

# Lock the data source. When true, the app auto-loads from BigQuery on visit
# using BQ_DEFAULT_* settings and users cannot change them. Designed for
# production deploys. Defaults to true when running on Railway.
LOCK_DATA_SOURCE = os.environ.get(
    "LOCK_DATA_SOURCE",
    "true" if os.environ.get("RAILWAY_ENVIRONMENT") else "false",
).lower() in ("1", "true", "yes")
BQ_DEFAULT_TABLE = os.environ.get("BQ_DEFAULT_TABLE", "").strip()
BQ_DEFAULT_DATE_COLUMN = os.environ.get("BQ_DEFAULT_DATE_COLUMN", "date").strip()
try:
    BQ_DEFAULT_LOOKBACK_DAYS = int(os.environ.get("BQ_DEFAULT_LOOKBACK_DAYS", "365"))
except ValueError:
    BQ_DEFAULT_LOOKBACK_DAYS = 365

# Channels whose spend is brand/awareness rather than performance-attributable.
# These show up in tables but recommendations don't apply CPA/LTV:CAC scrutiny
# to them and the "0 FTDs despite spend" trigger is suppressed.
# Override with BRAND_CHANNELS="ATL,Sponsorship,Out of home" etc.
BRAND_CHANNELS = set(
    c.strip() for c in os.environ.get("BRAND_CHANNELS", "ATL").split(",") if c.strip()
)

# Data-day cutoff: BQ refresh happens before this time. Cache invalidates daily
# at this UK-local time so a morning visit gets the fresh-loaded data.
DATA_DAY_CUTOFF_HOUR = int(os.environ.get("DATA_DAY_CUTOFF_HOUR", "8"))
DATA_DAY_CUTOFF_MINUTE = int(os.environ.get("DATA_DAY_CUTOFF_MINUTE", "30"))
UK_TZ = ZoneInfo("Europe/London")


def _data_day_key() -> str:
    """Cache key that flips once per day at DATA_DAY_CUTOFF UK time.

    Before the cutoff, we still consider the previous day current (BQ hasn't
    refreshed yet). After the cutoff, today becomes the new data day.
    """
    now = datetime.now(UK_TZ)
    cutoff = now.replace(
        hour=DATA_DAY_CUTOFF_HOUR,
        minute=DATA_DAY_CUTOFF_MINUTE,
        second=0,
        microsecond=0,
    )
    if now < cutoff:
        return (now.date() - timedelta(days=1)).isoformat()
    return now.date().isoformat()

# Default alert thresholds — UK iGaming sensible defaults.
# All overridable from the sidebar at runtime.
DEFAULT_THRESHOLDS = {
    "cpa_ftd_deterioration_pct": 20.0,        # CPA up by >20% triggers alert
    "ltv_cac_deterioration_pct": -15.0,       # LTV:CAC down by >15% triggers alert
    "cpa_apd2_deterioration_pct": 25.0,       # CPA APD2+ up by >25% triggers alert
    "unmatched_spend_share_pct": 15.0,        # >15% of spend unmatched triggers alert
    "min_spend_for_alert": 500.0,             # ignore alerts below £500 spend in window
    "tagging_coverage_warn_pct": 50.0,        # tagging coverage below 50% warns
    "tagging_coverage_critical_pct": 25.0,    # below 25% critical
    "scale_ltv_cac_floor": 1.5,               # Scale candidates need LTV:CAC >= 1.5
    "scale_cpa_improvement_pct": -10.0,       # ...or CPA improvement >= 10% drop
    "min_ftd_for_recommendation": 5,          # need ≥5 FTDs in window before recommending
}

# Saved views storage location
SAVED_VIEWS_PATH = Path(os.environ.get(
    "MRQ_SAVED_VIEWS_PATH",
    Path(__file__).parent / "saved_views.json",
))

# Approach levels that carry matched/spend_only/residual semantics
GRANULAR_APPROACHES = {"ad_group_level", "ad_level"}

# Numeric columns we want to coerce on load
NUMERIC_COLS = [
    "spend", "num_sessions", "num_sessions_new", "num_sessions_returning",
    "registrations", "legitimate_registrations", "leg_reg_to_ftd",
    "leg_reg_imm_ftd", "ftd_players", "imm_ftd_players", "conv_ftd_players",
    "first_deposit_amount", "sum_pltv", "sum_apd_first_week", "apd_2_players",
    "savvy_staker_players", "platform_conversions",
    "tagging_session_campaign_id_filled", "tagging_total_sessions",
]

# Default uploads sample path (used when running locally with the bundled CSV)
DEFAULT_CSV_HINT = "attribution_spend_metrics.csv"


# ---------------------------------------------------------------------------
# 2. SAFE MATH HELPERS
# ---------------------------------------------------------------------------

def safe_div(num, den):
    """Element-wise or scalar safe divide. Returns NaN where den is 0/NaN."""
    if isinstance(num, (pd.Series, np.ndarray)) or isinstance(den, (pd.Series, np.ndarray)):
        num_a = np.asarray(num, dtype="float64")
        den_a = np.asarray(den, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where((den_a == 0) | np.isnan(den_a), np.nan, num_a / den_a)
        if isinstance(num, pd.Series):
            return pd.Series(out, index=num.index)
        if isinstance(den, pd.Series):
            return pd.Series(out, index=den.index)
        return out
    if den is None or pd.isna(den) or den == 0:
        return float("nan")
    return num / den


def pct_change(curr, prev):
    """Safe percentage change (curr - prev) / prev * 100."""
    if prev is None or pd.isna(prev) or prev == 0:
        return float("nan")
    return (curr - prev) / abs(prev) * 100.0


def fmt_money(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{CURRENCY}{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{CURRENCY}{v/1_000:.1f}k"
    return f"{CURRENCY}{v:,.0f}"


def fmt_int(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{int(round(v)):,}"


def fmt_ratio(v, decimals: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{decimals}f}"


def fmt_pct(v, decimals: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+.{decimals}f}%"


# ---------------------------------------------------------------------------
# 3. DATA LOADING & SHAPING
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, signature: str) -> pd.DataFrame:
    """Read CSV, coerce types, add helper date fields. Cached on file signature."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    return _shape(df)


@st.cache_data(show_spinner=False)
def load_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _shape(df)


@st.cache_data(ttl=86400, show_spinner="Querying BigQuery…")
def load_bq(query: str, key_path: str | None, project_override: str | None,
            data_day: str = "") -> pd.DataFrame:
    """
    Run a SQL query against BigQuery and return a shaped DataFrame.

    Cached per (query, key_path, project, data_day) tuple. data_day flips once
    per day at the configured UK cutoff time so the morning's first visit gets
    fresh data; subsequent visits within the same data-day are served from
    cache. Use the 'Refresh data' button in the sidebar to bust early.

    Auth resolution:
      1. If key_path is given, use that JSON key file.
      2. Else if GOOGLE_APPLICATION_CREDENTIALS env var is set, use that.
      3. Else fall back to Application Default Credentials (gcloud auth).
    """
    from google.cloud import bigquery
    from google.oauth2 import service_account

    # Preference order: env var JSON > key file > Application Default Credentials.
    env_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if env_json:
        info = json.loads(env_json)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(
            credentials=creds,
            project=project_override or creds.project_id,
        )
    elif key_path and Path(key_path).expanduser().exists():
        creds = service_account.Credentials.from_service_account_file(
            str(Path(key_path).expanduser()),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(
            credentials=creds,
            project=project_override or creds.project_id,
        )
    else:
        client = bigquery.Client(project=project_override or None)

    job = client.query(query)
    try:
        df = job.to_dataframe(create_bqstorage_client=True)
    except Exception:
        # Fall back without BQ Storage API if it's not enabled
        df = job.to_dataframe(create_bqstorage_client=False)
    return _shape(df)


def _shape(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce numerics
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Date handling
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Helper fields
    df["day"] = df["date"].dt.normalize() if "date" in df.columns else pd.NaT
    df["week_start"] = df["day"] - pd.to_timedelta(df["day"].dt.weekday, unit="D")
    df["month_start"] = df["day"].values.astype("datetime64[M]")
    df["month_label"] = pd.to_datetime(df["month_start"]).dt.strftime("%Y-%m")
    # Friendly fillers for missing dimension labels
    for c in ["source", "channel", "name", "ad_group_name", "ad_name",
              "attribution_platform", "spend_attribution_approach", "spend_row_type"]:
        if c in df.columns:
            df[c] = df[c].fillna("(missing)")
    # Numeric fills (treat NaN spend/conversion as 0)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ---------------------------------------------------------------------------
# 4. AGGREGATION & DERIVED METRICS
# ---------------------------------------------------------------------------

AGG_SUMS = [
    "spend", "num_sessions", "num_sessions_new", "num_sessions_returning",
    "registrations", "legitimate_registrations", "leg_reg_to_ftd",
    "leg_reg_imm_ftd", "ftd_players", "imm_ftd_players", "conv_ftd_players",
    "first_deposit_amount", "sum_pltv", "sum_apd_first_week",
    "apd_2_players", "savvy_staker_players", "platform_conversions",
    "tagging_session_campaign_id_filled", "tagging_total_sessions",
]


def aggregate(
    df: pd.DataFrame,
    group_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Sum measure cols. If group_cols None, return a single-row total."""
    cols = [c for c in AGG_SUMS if c in df.columns]
    if not group_cols:
        out = df[cols].sum(min_count=1).to_frame().T
    else:
        out = df.groupby(list(group_cols), dropna=False)[cols].sum(min_count=1).reset_index()
    return _add_derived(out)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpa_ftd"] = safe_div(df.get("spend"), df.get("ftd_players"))
    df["ltv_cac"] = safe_div(df.get("sum_pltv"), df.get("spend"))
    df["pltv"] = df.get("sum_pltv")
    df["apd1"] = df.get("sum_apd_first_week")
    df["cpa_apd2"] = safe_div(df.get("spend"), df.get("apd_2_players"))
    df["cost_per_session"] = safe_div(df.get("spend"), df.get("num_sessions"))
    df["cost_per_registration"] = safe_div(df.get("spend"), df.get("registrations"))
    df["registration_rate"] = safe_div(df.get("registrations"), df.get("num_sessions"))
    df["ftd_rate"] = safe_div(df.get("ftd_players"), df.get("num_sessions"))
    df["leg_reg_rate"] = safe_div(df.get("legitimate_registrations"), df.get("registrations"))
    df["tagging_coverage"] = safe_div(
        df.get("tagging_session_campaign_id_filled"),
        df.get("tagging_total_sessions"),
    )
    return df


def filter_efficiency(
    df: pd.DataFrame,
    include_unmatched: bool,
) -> pd.DataFrame:
    """Restrict ad_group_level / ad_level rows to matched only when toggle is off.

    Higher-level approaches (channel/campaign/affiliate) are pass-through because
    they don't carry spend_row_type semantics.
    """
    if include_unmatched or "spend_row_type" not in df.columns:
        return df
    granular_mask = df["spend_attribution_approach"].isin(GRANULAR_APPROACHES)
    matched_mask = df["spend_row_type"] == "matched"
    return df[(~granular_mask) | (matched_mask & granular_mask)]


# ---------------------------------------------------------------------------
# 5. PERIOD HANDLING
# ---------------------------------------------------------------------------

@dataclass
class Period:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def days(self) -> int:
        return (self.end.normalize() - self.start.normalize()).days + 1

    def slice(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["date"] >= self.start) & (df["date"] <= self.end)]


def previous_period(p: Period, mode: str) -> Period:
    """
    mode="prev_month" -> same window shifted by 1 calendar month
    mode="prev_year"  -> same window shifted by 1 calendar year
    mode="prev_equal" -> immediately preceding equal-length window
    """
    if mode == "prev_year":
        return Period(p.start - relativedelta(years=1), p.end - relativedelta(years=1))
    if mode == "prev_equal":
        length = p.days
        new_end = p.start - timedelta(days=1)
        new_start = new_end - timedelta(days=length - 1)
        return Period(new_start, new_end)
    # default: prev_month
    new_start = p.start - relativedelta(months=1)
    new_end = p.end - relativedelta(months=1)
    return Period(new_start, new_end)


def mtd_period(today: pd.Timestamp | None = None) -> Period:
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    return Period(today.replace(day=1), today)


# ---------------------------------------------------------------------------
# 6. SAVED VIEWS (LOCAL JSON)
# ---------------------------------------------------------------------------

def load_saved_views() -> dict:
    if SAVED_VIEWS_PATH.exists():
        try:
            return json.loads(SAVED_VIEWS_PATH.read_text())
        except Exception:
            return {}
    return {}


def write_saved_views(views: dict) -> None:
    SAVED_VIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAVED_VIEWS_PATH.write_text(json.dumps(views, default=str, indent=2))


# ---------------------------------------------------------------------------
# 7. CHARTS
# ---------------------------------------------------------------------------

def line_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None,
               title: str = "", y_label: str | None = None) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        yaxis_title=y_label or y,
    )
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None,
              title: str = "", orientation: str = "v") -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title, orientation=orientation)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    return fig


def scatter_chart(df: pd.DataFrame, x: str, y: str, size: str | None = None,
                  color: str | None = None, hover_name: str | None = None,
                  title: str = "") -> go.Figure:
    fig = px.scatter(df, x=x, y=y, size=size, color=color,
                     hover_name=hover_name, title=title, size_max=40)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    return fig


def sparkline(values: Iterable[float]) -> go.Figure:
    vals = list(values)
    fig = go.Figure(data=go.Scatter(y=vals, mode="lines", line=dict(width=2)))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=40,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. KPI CARDS
# ---------------------------------------------------------------------------

KPI_DEFS = [
    # (label, metric_key, formatter, lower_is_better)
    ("Spend",         "spend",         fmt_money, False),
    ("FTDs",          "ftd_players",   fmt_int,   False),
    ("LTV:CAC",       "ltv_cac",       lambda v: fmt_ratio(v, 2), False),
    ("CPA (FTD)",     "cpa_ftd",       fmt_money, True),
    ("pLTV",          "pltv",          fmt_money, False),
    ("FW2+ players",  "apd_2_players", fmt_int,   False),
    ("CPA APD2+",     "cpa_apd2",      fmt_money, True),
]


def kpi_grid(curr_row: pd.Series, prev_row: pd.Series | None,
             trend_df: pd.DataFrame | None = None,
             show_sparkline: bool = False) -> None:
    cols = st.columns(len(KPI_DEFS))
    for col, (label, key, formatter, lower_better) in zip(cols, KPI_DEFS):
        with col:
            curr = curr_row.get(key)
            prev = prev_row.get(key) if prev_row is not None else None
            change = pct_change(curr, prev)
            arrow = ""
            if not pd.isna(change):
                worsened = (change > 0 and lower_better) or (change < 0 and not lower_better)
                arrow = "▲" if change > 0 else "▼"
                color = "#c0392b" if worsened else "#27ae60"
                delta_text = f"<span style='color:{color}'>{arrow} {fmt_pct(change)}</span>"
            else:
                delta_text = "<span style='color:#888'>n/a</span>"
            st.markdown(
                f"<div style='padding:6px 8px;border:1px solid #eee;border-radius:6px'>"
                f"<div style='font-size:12px;color:#666'>{label}</div>"
                f"<div style='font-size:22px;font-weight:600'>{formatter(curr)}</div>"
                f"<div style='font-size:12px'>{delta_text} vs prev</div></div>",
                unsafe_allow_html=True,
            )
            if show_sparkline and trend_df is not None and key in trend_df.columns:
                spark = trend_df[key].fillna(0).tolist()
                if spark:
                    st.plotly_chart(sparkline(spark), use_container_width=True,
                                    config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# 9. RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------

def build_recommendations(
    df_curr: pd.DataFrame,
    df_prev: pd.DataFrame,
    group_cols: list[str],
    thresholds: dict,
    dq_signals: dict | None = None,
) -> pd.DataFrame:
    """Rules-based recommendation table.

    Joins current & previous on group_cols, computes deltas, and applies rules.
    Returns one row per entity with recommendation, reasons, supporting deltas,
    and a confidence label based on spend volume and data quality.
    """
    curr = aggregate(df_curr, group_cols)
    prev = aggregate(df_prev, group_cols)

    keep = group_cols + ["spend", "ftd_players", "sum_pltv",
                         "apd_2_players", "cpa_ftd", "ltv_cac", "cpa_apd2"]
    curr = curr[[c for c in keep if c in curr.columns]]
    prev = prev[[c for c in keep if c in prev.columns]]

    merged = curr.merge(prev, on=group_cols, how="left", suffixes=("", "_prev"))

    rows: list[dict] = []
    for _, r in merged.iterrows():
        spend = r.get("spend") or 0
        ftds = r.get("ftd_players") or 0
        cpa_change = pct_change(r.get("cpa_ftd"), r.get("cpa_ftd_prev"))
        ltv_cac_change = pct_change(r.get("ltv_cac"), r.get("ltv_cac_prev"))
        apd2_cpa_change = pct_change(r.get("cpa_apd2"), r.get("cpa_apd2_prev"))

        reasons: list[str] = []
        rec = "Hold"

        # Brand / above-the-line channels: not click-through attributable.
        # Skip performance scrutiny entirely.
        entity_channel = r.get("channel") if "channel" in group_cols else None
        if entity_channel and entity_channel in BRAND_CHANNELS:
            rows.append({
                **{c: r[c] for c in group_cols},
                "spend": spend,
                "ftds": ftds,
                "cpa_ftd": r.get("cpa_ftd"),
                "cpa_ftd_Δ%": cpa_change,
                "ltv_cac": r.get("ltv_cac"),
                "ltv_cac_Δ%": ltv_cac_change,
                "cpa_apd2": r.get("cpa_apd2"),
                "cpa_apd2_Δ%": apd2_cpa_change,
                "recommendation": "Brand",
                "reasons": f"Brand / above-the-line — performance attribution doesn't apply ({fmt_money(spend)} spend)",
                "confidence": "N/A",
            })
            continue

        # Min spend gate
        if spend < thresholds["min_spend_for_alert"]:
            rec = "Hold"
            reasons.append(f"below {CURRENCY}{thresholds['min_spend_for_alert']:.0f} spend gate")
        else:
            # Special case: meaningful spend with zero attributed FTDs
            if ftds == 0:
                rec = "Investigate"
                reasons.append(f"{fmt_money(spend)} spend, 0 attributed FTDs — check tagging / attribution path")

            # Investigate triggers
            if not pd.isna(cpa_change) and cpa_change >= thresholds["cpa_ftd_deterioration_pct"]:
                reasons.append(f"CPA deterioration {fmt_pct(cpa_change)}")
            if not pd.isna(ltv_cac_change) and ltv_cac_change <= thresholds["ltv_cac_deterioration_pct"]:
                reasons.append(f"LTV:CAC deterioration {fmt_pct(ltv_cac_change)}")
            if not pd.isna(apd2_cpa_change) and apd2_cpa_change >= thresholds["cpa_apd2_deterioration_pct"]:
                reasons.append(f"APD2+ CPA deterioration {fmt_pct(apd2_cpa_change)}")

            if reasons and rec != "Investigate":
                rec = "Investigate"

            if rec != "Investigate":
                # Scale candidates: solid LTV:CAC AND (improvement OR strong base)
                ltv_cac_now = r.get("ltv_cac")
                if (ftds >= thresholds["min_ftd_for_recommendation"]
                        and not pd.isna(ltv_cac_now)
                        and ltv_cac_now >= thresholds["scale_ltv_cac_floor"]):
                    if (not pd.isna(cpa_change)
                            and cpa_change <= thresholds["scale_cpa_improvement_pct"]):
                        rec = "Scale"
                        reasons.append(f"CPA improved {fmt_pct(cpa_change)}, LTV:CAC {fmt_ratio(ltv_cac_now)}")
                    elif not pd.isna(ltv_cac_change) and ltv_cac_change >= 5:
                        rec = "Scale"
                        reasons.append(f"LTV:CAC improving ({fmt_pct(ltv_cac_change)}), at {fmt_ratio(ltv_cac_now)}")
                    else:
                        rec = "Hold"
                        reasons.append(f"stable at LTV:CAC {fmt_ratio(ltv_cac_now)}")
                else:
                    # Below LTV:CAC floor — surface positive trend but explain why not Scale
                    rec = "Hold"
                    if (not pd.isna(ltv_cac_now) and ltv_cac_now < thresholds["scale_ltv_cac_floor"]
                            and not pd.isna(ltv_cac_change) and ltv_cac_change >= 5):
                        reasons.append(
                            f"LTV:CAC improving ({fmt_pct(ltv_cac_change)}) but still below "
                            f"{thresholds['scale_ltv_cac_floor']} floor at {fmt_ratio(ltv_cac_now)}"
                        )
                    elif not pd.isna(ltv_cac_now) and ltv_cac_now < thresholds["scale_ltv_cac_floor"]:
                        reasons.append(f"LTV:CAC {fmt_ratio(ltv_cac_now)} below {thresholds['scale_ltv_cac_floor']} Scale floor")
                    else:
                        reasons.append("stable, sub-threshold volume or LTV:CAC")

        # Confidence
        if spend >= 10_000 and ftds >= 50:
            confidence = "High"
        elif spend >= 2_000 and ftds >= 10:
            confidence = "Medium"
        else:
            confidence = "Low"

        rows.append({
            **{c: r[c] for c in group_cols},
            "spend": spend,
            "ftds": ftds,
            "cpa_ftd": r.get("cpa_ftd"),
            "cpa_ftd_Δ%": cpa_change,
            "ltv_cac": r.get("ltv_cac"),
            "ltv_cac_Δ%": ltv_cac_change,
            "cpa_apd2": r.get("cpa_apd2"),
            "cpa_apd2_Δ%": apd2_cpa_change,
            "recommendation": rec,
            "reasons": "; ".join(reasons) or "—",
            "confidence": confidence,
        })

    if not rows:
        return pd.DataFrame(columns=group_cols + [
            "spend", "ftds", "cpa_ftd", "cpa_ftd_Δ%", "ltv_cac", "ltv_cac_Δ%",
            "cpa_apd2", "cpa_apd2_Δ%", "recommendation", "reasons", "confidence",
        ])
    return pd.DataFrame(rows).sort_values("spend", ascending=False, kind="mergesort")


# ---------------------------------------------------------------------------
# 10. DATA QUALITY CHECKS
# ---------------------------------------------------------------------------

def data_quality(df_window: pd.DataFrame, thresholds: dict) -> dict:
    """Returns a dict of dq metrics + a summary status (Healthy/Warning/Investigate)."""
    out: dict = {}
    total_spend = df_window["spend"].sum()
    if "spend_row_type" in df_window.columns:
        granular = df_window[df_window["spend_attribution_approach"].isin(GRANULAR_APPROACHES)]
        granular_spend = granular["spend"].sum()
        unmatched_spend = granular[granular["spend_row_type"] != "matched"]["spend"].sum()
        residual_spend = granular[granular["spend_row_type"] == "residual"]["spend"].sum()
        out["unmatched_spend_share_pct"] = (
            (unmatched_spend / granular_spend * 100) if granular_spend else float("nan")
        )
        out["residual_spend_share_pct"] = (
            (residual_spend / granular_spend * 100) if granular_spend else float("nan")
        )
        out["granular_spend"] = granular_spend
        out["unmatched_spend"] = unmatched_spend
        out["residual_spend"] = residual_spend
    else:
        out["unmatched_spend_share_pct"] = float("nan")
        out["residual_spend_share_pct"] = float("nan")
    out["total_spend"] = total_spend

    # Tagging coverage
    sess_total = df_window["tagging_total_sessions"].sum() if "tagging_total_sessions" in df_window.columns else 0
    sess_filled = df_window["tagging_session_campaign_id_filled"].sum() if "tagging_session_campaign_id_filled" in df_window.columns else 0
    out["tagging_coverage_pct"] = (sess_filled / sess_total * 100) if sess_total else float("nan")
    out["tagging_total_sessions"] = sess_total
    out["tagging_filled_sessions"] = sess_filled

    # Blank rates for key dimensions
    out["blank_rates"] = {}
    for col in ["source", "channel", "name", "ad_group_name", "ad_name"]:
        if col in df_window.columns:
            blank = (df_window[col] == "(missing)") | df_window[col].isna()
            out["blank_rates"][col] = blank.mean() * 100

    # Anomaly: spend > 0 with sessions = 0
    if "num_sessions" in df_window.columns:
        anomaly = df_window[(df_window["spend"] > 0) & (df_window["num_sessions"] == 0)]
        out["spend_no_sessions_rows"] = len(anomaly)
        out["spend_no_sessions_amount"] = anomaly["spend"].sum()

    # Anomaly: ftd_players > registrations
    if {"ftd_players", "registrations"}.issubset(df_window.columns):
        out["ftd_gt_reg_rows"] = int((df_window["ftd_players"] > df_window["registrations"]).sum())

    # Status
    status = "Healthy"
    notes: list[str] = []
    if out["unmatched_spend_share_pct"] is not None and not pd.isna(out["unmatched_spend_share_pct"]):
        if out["unmatched_spend_share_pct"] >= thresholds["unmatched_spend_share_pct"]:
            status = "Warning"
            notes.append(f"Unmatched spend share {out['unmatched_spend_share_pct']:.1f}% above threshold")
    if not pd.isna(out["tagging_coverage_pct"]):
        if out["tagging_coverage_pct"] < thresholds["tagging_coverage_critical_pct"]:
            status = "Investigate"
            notes.append(f"Tagging coverage {out['tagging_coverage_pct']:.1f}% critically low")
        elif out["tagging_coverage_pct"] < thresholds["tagging_coverage_warn_pct"] and status == "Healthy":
            status = "Warning"
            notes.append(f"Tagging coverage {out['tagging_coverage_pct']:.1f}% below warn level")
    if out.get("spend_no_sessions_rows", 0) > 0:
        if status == "Healthy":
            status = "Warning"
        notes.append(f"{out['spend_no_sessions_rows']:,} rows with spend but zero sessions ({fmt_money(out['spend_no_sessions_amount'])})")
    out["status"] = status
    out["notes"] = notes
    return out


# ---------------------------------------------------------------------------
# 11. DOWNLOAD HELPERS
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fig_to_image_bytes(fig: go.Figure, fmt: str = "png") -> bytes | None:
    """Try to export Plotly figure to image bytes via kaleido.

    Returns None if kaleido is not installed; the user can still right-click
    on the chart in the browser to download it.
    """
    try:
        return fig.to_image(format=fmt, scale=2)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 12. APP
# ---------------------------------------------------------------------------

def gate_auth() -> None:
    """Require Google login restricted to ALLOWED_EMAIL_DOMAINS.

    Uses Streamlit's native auth (st.login / st.user / st.logout), which reads
    OAuth config from st.secrets["auth"]. Bootstrap script writes those values
    from env vars at container startup.

    Behaviour:
      - REQUIRE_AUTH=false: pass-through (local dev mode).
      - REQUIRE_AUTH=true but [auth] not configured: show a friendly error.
      - Otherwise: show a login button until the user signs in. Reject any
        email whose domain isn't in ALLOWED_EMAIL_DOMAINS.
    """
    if not REQUIRE_AUTH:
        return

    if "auth" not in st.secrets:
        st.error(
            "Authentication is required (REQUIRE_AUTH=true) but the [auth] "
            "section is missing from secrets. Set GOOGLE_CLIENT_ID, "
            "GOOGLE_CLIENT_SECRET, AUTH_REDIRECT_URI and AUTH_COOKIE_SECRET "
            "in the environment."
        )
        st.stop()

    if not getattr(st, "user", None) or not st.user.is_logged_in:
        st.title(APP_TITLE)
        st.write(
            f"Sign in with your MrQ account "
            f"({', '.join('@' + d for d in ALLOWED_EMAIL_DOMAINS)}) to continue."
        )
        if st.button("Sign in", type="primary"):
            st.login()
        st.stop()

    email = (getattr(st.user, "email", "") or "").lower()
    if not any(email.endswith("@" + d) for d in ALLOWED_EMAIL_DOMAINS):
        st.error(
            f"Access is restricted to {', '.join('@' + d for d in ALLOWED_EMAIL_DOMAINS)} "
            f"accounts. You're signed in as **{email or 'unknown'}**."
        )
        if st.button("Sign out"):
            st.logout()
        st.stop()

    # Signed in & allowed — show identity + sign-out in sidebar
    with st.sidebar:
        st.caption(f"Signed in as {email}")
        if st.button("Sign out", key="_signout"):
            st.logout()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    gate_auth()
    st.title(APP_TITLE)
    st.caption(
        "Internal performance marketing tool for the MrQ team. "
        "Treats different data grains separately. "
        "Recommendations are heuristic — they support, not replace, judgement."
    )

    df = _load_data_widget()
    if df is None or df.empty:
        st.info("Upload a CSV with the attribution + spend schema to begin.")
        return

    thresholds = _thresholds_widget()
    period, prev_period_obj, prev_mode = _period_widget(df)
    filters, include_unmatched, group_grain_map = _filters_widget(df)

    # Filtered current and previous slices
    df_curr_raw = period.slice(df)
    df_prev_raw = prev_period_obj.slice(df)
    df_curr = _apply_filters(df_curr_raw, filters)
    df_prev = _apply_filters(df_prev_raw, filters)
    df_curr_eff = filter_efficiency(df_curr, include_unmatched)
    df_prev_eff = filter_efficiency(df_prev, include_unmatched)

    # Saved views & top banner
    _saved_views_widget(filters, include_unmatched, period, prev_mode, thresholds)

    _interpretation_banner(df_curr, include_unmatched, period, prev_period_obj, prev_mode)

    # Page nav (tabs across the top). st.tabs renders all bodies on every
    # rerun, but the heavy work (BQ fetch + _shape) is cached, so this is
    # cheap. Per-page aggregations are recomputed cheaply from the shaped df.
    tab_overview, tab_channels, tab_explorer, tab_dq, tab_recs, tab_dict = st.tabs([
        "Executive Overview",
        "Channel View",
        "Campaign / Ad Explorer",
        "Data Quality",
        "Recommendations",
        "Dictionary",
    ])
    with tab_overview:
        page_overview(df, df_curr_eff, df_prev_eff, period, prev_period_obj, thresholds)
    with tab_channels:
        page_channels(df_curr_eff, df_prev_eff)
    with tab_explorer:
        page_explorer(df_curr, df_prev)  # explorer respects own row-type filter
    with tab_dq:
        page_data_quality(df_curr, period, thresholds)
    with tab_recs:
        page_recommendations(df_curr_eff, df_prev_eff, thresholds)
    with tab_dict:
        page_dictionary()


# ---------------------------------------------------------------------------
# 13. SIDEBAR WIDGETS
# ---------------------------------------------------------------------------

def _load_data_widget() -> pd.DataFrame | None:
    # Production / locked-down mode: auto-load from BQ using env-var settings.
    # No source picker, no query mode toggle. Users can still hit Refresh.
    if LOCK_DATA_SOURCE and BQ_DEFAULT_TABLE:
        return _load_bq_locked()

    st.sidebar.header("Data source")
    src = st.sidebar.radio(
        "Load from",
        ["Upload CSV", "Local file path", "BigQuery"],
        horizontal=False,
    )
    if src == "Upload CSV":
        upload = st.sidebar.file_uploader("CSV file", type=["csv"])
        if upload is None:
            # Try sibling CSV next to app for convenience
            sibling = Path(__file__).parent / DEFAULT_CSV_HINT
            if sibling.exists():
                st.sidebar.caption(f"Auto-loaded sibling file: {sibling.name}")
                return load_path(str(sibling))
            return None
        return load_csv(upload.getvalue(), upload.name + str(upload.size))
    if src == "Local file path":
        path = st.sidebar.text_input("Path to CSV", value=str(Path(__file__).parent / DEFAULT_CSV_HINT))
        if path and Path(path).exists():
            return load_path(path)
        st.sidebar.warning("File not found at that path.")
        return None
    # BigQuery branch
    return _load_bq_widget()


def _load_bq_locked() -> pd.DataFrame | None:
    """Production data loader: BigQuery, fully driven by env vars.

    Designed for the deployed app — users don't configure the connection.
    They see the data loaded automatically with a refresh button and a
    compact 'last loaded' caption in the sidebar.
    """
    where = (
        f"WHERE {BQ_DEFAULT_DATE_COLUMN} >= DATE_SUB(CURRENT_DATE(), "
        f"INTERVAL {BQ_DEFAULT_LOOKBACK_DAYS} DAY)"
        if BQ_DEFAULT_LOOKBACK_DAYS > 0
        else ""
    )
    query = f"SELECT * FROM `{BQ_DEFAULT_TABLE}` {where}".strip()

    data_day = _data_day_key()
    with st.sidebar:
        st.header("Data source")
        st.caption(f"BigQuery: `{BQ_DEFAULT_TABLE}`")
        st.caption(
            f"Window: last {BQ_DEFAULT_LOOKBACK_DAYS} days "
            f"(by `{BQ_DEFAULT_DATE_COLUMN}`)"
            if BQ_DEFAULT_LOOKBACK_DAYS > 0
            else "Window: full table"
        )
        st.caption(
            f"Data day: {data_day} "
            f"(auto-refresh at {DATA_DAY_CUTOFF_HOUR:02d}:{DATA_DAY_CUTOFF_MINUTE:02d} UK)"
        )
        if st.button("Refresh data now", help="Clears the cache and refetches from BigQuery immediately"):
            load_bq.clear()
            st.rerun()

    try:
        df = load_bq(query, None, None, data_day)
        return df
    except Exception as e:
        st.error(f"BigQuery query failed: {e}")
        st.caption(
            "If this is an access error, ask data to grant `roles/bigquery.dataViewer` "
            "on the dataset to the deployed service account."
        )
        return None


def _load_bq_widget() -> pd.DataFrame | None:
    """BigQuery data source. Reads service account key from env var by default."""
    with st.sidebar.expander("BigQuery connection", expanded=True):
        env_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        key_path = st.text_input(
            "Service account key path",
            value=env_key,
            help="Path to the service account JSON key file. "
                 "Set GOOGLE_APPLICATION_CREDENTIALS once in your shell to skip this. "
                 "DO NOT paste the key contents here, only the path.",
            placeholder="~/.config/mrq/bq-service-account.json",
        )
        project = st.text_input(
            "Project ID (optional, overrides key default)",
            value=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            placeholder="mrq-data-prod",
        )

        # Two query modes: simple (table reference) or advanced (full SQL)
        mode = st.radio(
            "Query mode",
            ["Table reference", "Custom SQL"],
            horizontal=True,
            key="bq_mode",
        )
        last = st.session_state.get("_bq_last", {})
        if mode == "Table reference":
            table_ref = st.text_input(
                "Table",
                value=last.get("table_ref", "project.dataset.table"),
                help="Format: project.dataset.table",
                key="bq_table_ref",
            )
            date_col = st.text_input(
                "Date column",
                value=last.get("date_col", "date"),
                key="bq_date_col",
            )
            since_days = st.number_input(
                "Pull last N days (0 = all)",
                value=int(last.get("since_days", 365)),
                min_value=0,
                step=30,
                key="bq_since_days",
            )
            where_clause = (
                f"WHERE {date_col} >= DATE_SUB(CURRENT_DATE(), INTERVAL {since_days} DAY)"
                if since_days else ""
            )
            query = f"SELECT * FROM `{table_ref}` {where_clause}".strip()
            st.code(query, language="sql")
        else:
            query = st.text_area(
                "SQL",
                value=last.get("query", "SELECT * FROM `project.dataset.table` LIMIT 1000"),
                height=160,
                key="bq_query",
            )

        col1, col2 = st.columns(2)
        with col1:
            run = st.button("Run query", type="primary", key="bq_run")
        with col2:
            refresh = st.button("Refresh (clear cache)", key="bq_refresh")

        if refresh:
            load_bq.clear()
            st.success("Cache cleared. Click 'Run query' to refetch.")

        if not run and "_bq_df" not in st.session_state:
            st.caption("Set the connection, then click Run query.")
            return None

        if run:
            try:
                df = load_bq(query, key_path or None, project or None, _data_day_key())
                st.session_state["_bq_df"] = df
                st.session_state["_bq_last"] = {
                    "query": query,
                    **({
                        "table_ref": st.session_state.get("bq_table_ref"),
                        "date_col": st.session_state.get("bq_date_col"),
                        "since_days": st.session_state.get("bq_since_days"),
                    } if mode == "Table reference" else {}),
                }
                st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns from BigQuery.")
            except Exception as e:
                st.error(f"BigQuery query failed: {e}")
                return None
        return st.session_state.get("_bq_df")


def _thresholds_widget() -> dict:
    with st.sidebar.expander("Alert thresholds", expanded=False):
        out = dict(DEFAULT_THRESHOLDS)
        out["cpa_ftd_deterioration_pct"] = st.number_input(
            "CPA FTD deterioration % (alert if ≥)", value=float(out["cpa_ftd_deterioration_pct"]), step=1.0)
        out["ltv_cac_deterioration_pct"] = st.number_input(
            "LTV:CAC deterioration % (alert if ≤, neg means worse)", value=float(out["ltv_cac_deterioration_pct"]), step=1.0)
        out["cpa_apd2_deterioration_pct"] = st.number_input(
            "CPA APD2+ deterioration % (alert if ≥)", value=float(out["cpa_apd2_deterioration_pct"]), step=1.0)
        out["unmatched_spend_share_pct"] = st.number_input(
            "Unmatched spend share % (alert if ≥)", value=float(out["unmatched_spend_share_pct"]), step=1.0)
        out["min_spend_for_alert"] = st.number_input(
            f"Min spend gate ({CURRENCY})", value=float(out["min_spend_for_alert"]), step=50.0)
        out["tagging_coverage_warn_pct"] = st.number_input(
            "Tagging coverage warn %", value=float(out["tagging_coverage_warn_pct"]), step=1.0)
        out["tagging_coverage_critical_pct"] = st.number_input(
            "Tagging coverage critical %", value=float(out["tagging_coverage_critical_pct"]), step=1.0)
        out["scale_ltv_cac_floor"] = st.number_input(
            "Scale LTV:CAC floor", value=float(out["scale_ltv_cac_floor"]), step=0.1)
        out["scale_cpa_improvement_pct"] = st.number_input(
            "Scale CPA improvement % (CPA must drop by at least this; negative)",
            value=float(out["scale_cpa_improvement_pct"]), step=1.0)
        out["min_ftd_for_recommendation"] = st.number_input(
            "Min FTDs for Scale recommendation", value=int(out["min_ftd_for_recommendation"]), step=1)
        return out


def _period_widget(df: pd.DataFrame) -> tuple[Period, Period, str]:
    st.sidebar.header("Period")
    today = pd.Timestamp.today().normalize()
    data_max = df["date"].max()
    if pd.isna(data_max):
        data_max = today
    data_min = df["date"].min()
    if pd.isna(data_min):
        data_min = data_max - pd.Timedelta(days=365)

    preset = st.sidebar.selectbox(
        "Preset",
        ["Month to date", "Previous full month", "Last 7 days", "Last 14 days",
         "Last 28 days", "Last 90 days", "Custom"],
        index=0,
    )

    end_default = min(data_max, today)
    if preset == "Month to date":
        start = end_default.replace(day=1)
        end = end_default
    elif preset == "Previous full month":
        first_this = end_default.replace(day=1)
        last_prev = first_this - pd.Timedelta(days=1)
        start = last_prev.replace(day=1)
        end = last_prev
    elif preset == "Last 7 days":
        start, end = end_default - pd.Timedelta(days=6), end_default
    elif preset == "Last 14 days":
        start, end = end_default - pd.Timedelta(days=13), end_default
    elif preset == "Last 28 days":
        start, end = end_default - pd.Timedelta(days=27), end_default
    elif preset == "Last 90 days":
        start, end = end_default - pd.Timedelta(days=89), end_default
    else:
        start, end = end_default.replace(day=1), end_default

    sel = st.sidebar.date_input(
        "Date range",
        value=(start.date(), end.date()),
        min_value=data_min.date(),
        max_value=data_max.date(),
    )
    if isinstance(sel, tuple) and len(sel) == 2:
        start, end = pd.Timestamp(sel[0]), pd.Timestamp(sel[1])

    period = Period(start, end)

    prev_mode = st.sidebar.selectbox(
        "Compare vs",
        ["Same window in previous month", "Same window in previous year",
         "Immediately preceding equal-length window", "No comparison"],
        index=0,
    )
    mode_map = {
        "Same window in previous month": "prev_month",
        "Same window in previous year": "prev_year",
        "Immediately preceding equal-length window": "prev_equal",
        "No comparison": "none",
    }
    mode = mode_map[prev_mode]
    if mode == "none":
        # Empty previous period — use a 0-day window outside data
        prev_obj = Period(period.start - pd.Timedelta(days=1), period.start - pd.Timedelta(days=1))
    else:
        prev_obj = previous_period(period, mode)
    return period, prev_obj, mode


def _filters_widget(df: pd.DataFrame) -> tuple[dict, bool, dict]:
    st.sidebar.header("Filters")
    filters: dict = {}
    for col in ["source", "channel", "attribution_platform", "spend_attribution_approach"]:
        if col in df.columns:
            opts = sorted(df[col].dropna().unique().tolist())
            sel = st.sidebar.multiselect(col, opts, default=[])
            if sel:
                filters[col] = sel
    if "spend_row_type" in df.columns:
        rt_opts = sorted([x for x in df["spend_row_type"].dropna().unique() if x != "(missing)"])
        rt_sel = st.sidebar.multiselect("spend_row_type (granular grains only)", rt_opts, default=[])
        if rt_sel:
            filters["spend_row_type"] = rt_sel
    include_unmatched = st.sidebar.checkbox(
        "Include unmatched/residual in efficiency metrics",
        value=False,
        help="Off (default) restricts ad/ad_group level rows to matched only when computing CPA, LTV:CAC, etc. "
             "Channel/campaign/affiliate-level rows are unaffected because they don't carry spend_row_type.",
    )
    return filters, include_unmatched, {}


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for col, sel in filters.items():
        if col in out.columns and sel:
            out = out[out[col].isin(sel)]
    return out


def _saved_views_widget(filters, include_unmatched, period, prev_mode, thresholds):
    with st.sidebar.expander("Saved views", expanded=False):
        views = load_saved_views()
        if views:
            sel = st.selectbox("Load view", ["—"] + list(views.keys()))
            if sel and sel != "—":
                st.code(json.dumps(views[sel], indent=2, default=str), language="json")
                st.caption("Apply by setting the matching filters in the sidebar.")
        name = st.text_input("Save current view as", "")
        if st.button("Save view") and name.strip():
            views[name.strip()] = {
                "filters": filters,
                "include_unmatched": include_unmatched,
                "period": {"start": str(period.start.date()), "end": str(period.end.date())},
                "prev_mode": prev_mode,
                "thresholds": thresholds,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_saved_views(views)
            st.success(f"Saved '{name}'.")
        if views:
            del_name = st.selectbox("Delete view", ["—"] + list(views.keys()), key="del_view")
            if st.button("Delete") and del_name and del_name != "—":
                del views[del_name]
                write_saved_views(views)
                st.warning(f"Deleted '{del_name}'.")


def _interpretation_banner(df_curr, include_unmatched, period, prev_period_obj, prev_mode):
    if df_curr.empty:
        st.warning("No rows in the selected period after filters. Loosen filters or extend the date range.")
        return
    grain_mix = df_curr["spend_attribution_approach"].value_counts(dropna=False)
    grain_str = ", ".join(f"{k}: {v:,}" for k, v in grain_mix.items())
    eff_text = "matched only on granular grains" if not include_unmatched else "all row types in efficiency"
    msg = (
        f"**Period:** {period.start.date()} → {period.end.date()} "
        f"({period.days} days). "
        f"**Prev:** {prev_period_obj.start.date()} → {prev_period_obj.end.date()} "
        f"({prev_mode}). "
        f"**Rows by grain:** {grain_str}. **Efficiency rule:** {eff_text}."
    )
    st.info(msg)


# ---------------------------------------------------------------------------
# 14. PAGE: EXECUTIVE OVERVIEW
# ---------------------------------------------------------------------------

def page_overview(df_full, df_curr, df_prev, period, prev_period_obj, thresholds):
    st.header("Executive Overview")
    show_spark = st.toggle("Show sparklines on KPI cards", value=False)

    curr_total = aggregate(df_curr).iloc[0]
    prev_total = aggregate(df_prev).iloc[0] if not df_prev.empty else None

    daily = aggregate(df_curr, ["day"]).sort_values("day")
    kpi_grid(curr_total, prev_total, trend_df=daily, show_sparkline=show_spark)

    st.subheader("Trend over time (current period)")
    if daily.empty:
        st.warning("No daily rows in selected window.")
    else:
        cols = st.columns(3)
        with cols[0]:
            st.plotly_chart(line_chart(daily, "day", "spend", title="Spend"), use_container_width=True)
            st.plotly_chart(line_chart(daily, "day", "cpa_ftd", title="CPA (FTD)"), use_container_width=True)
        with cols[1]:
            st.plotly_chart(line_chart(daily, "day", "ftd_players", title="FTDs"), use_container_width=True)
            st.plotly_chart(line_chart(daily, "day", "pltv", title="pLTV"), use_container_width=True)
        with cols[2]:
            st.plotly_chart(line_chart(daily, "day", "ltv_cac", title="LTV:CAC"), use_container_width=True)
            st.plotly_chart(line_chart(daily, "day", "cpa_apd2", title="CPA APD2+"), use_container_width=True)

    st.subheader("Channel summary")
    ch_curr = aggregate(df_curr, ["channel"])
    ch_prev = aggregate(df_prev, ["channel"]) if not df_prev.empty else None
    ch_table = _summary_table(ch_curr, ch_prev, "channel")
    st.dataframe(ch_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download channel summary (CSV)",
        df_to_csv_bytes(ch_table),
        "channel_summary.csv",
        "text/csv",
    )

    st.subheader("Top movers (by spend Δ vs previous period)")
    movers = _top_movers(df_curr, df_prev, "channel")
    st.dataframe(movers, use_container_width=True, hide_index=True)


def _summary_table(curr: pd.DataFrame, prev: pd.DataFrame | None, dim: str) -> pd.DataFrame:
    keep = [dim, "spend", "ftd_players", "sum_pltv", "cpa_ftd", "ltv_cac", "cpa_apd2", "apd_2_players"]
    curr = curr[[c for c in keep if c in curr.columns]].copy()
    if prev is not None and not prev.empty:
        prev_renamed = prev[[c for c in keep if c in prev.columns]].copy()
        prev_renamed.columns = [dim] + [f"{c}_prev" for c in prev_renamed.columns if c != dim]
        merged = curr.merge(prev_renamed, on=dim, how="left")
    else:
        merged = curr
        for c in [c for c in keep if c != dim]:
            merged[f"{c}_prev"] = np.nan

    for c in ["spend", "ftd_players", "sum_pltv", "cpa_ftd", "ltv_cac", "cpa_apd2"]:
        if c in merged.columns and f"{c}_prev" in merged.columns:
            merged[f"{c}_Δ%"] = [pct_change(a, b) for a, b in zip(merged[c], merged[f"{c}_prev"])]
    merged = merged.sort_values("spend", ascending=False, kind="mergesort")
    show_cols = [dim, "spend", "spend_Δ%", "ftd_players", "ftd_players_Δ%",
                 "cpa_ftd", "cpa_ftd_Δ%", "ltv_cac", "ltv_cac_Δ%",
                 "cpa_apd2", "cpa_apd2_Δ%", "sum_pltv", "sum_pltv_Δ%"]
    show_cols = [c for c in show_cols if c in merged.columns]
    return merged[show_cols].round(2)


def _top_movers(df_curr: pd.DataFrame, df_prev: pd.DataFrame, dim: str, n: int = 10) -> pd.DataFrame:
    curr = aggregate(df_curr, [dim])
    prev = aggregate(df_prev, [dim])
    m = curr.merge(prev, on=dim, how="outer", suffixes=("", "_prev")).fillna(0)
    m["spend_Δ"] = m["spend"] - m["spend_prev"]
    m["ftds_Δ"] = m.get("ftd_players", 0) - m.get("ftd_players_prev", 0)
    m["spend_Δ%"] = [pct_change(a, b) for a, b in zip(m["spend"], m["spend_prev"])]
    m = m.reindex(m["spend_Δ"].abs().sort_values(ascending=False).index).head(n)
    return m[[dim, "spend_prev", "spend", "spend_Δ", "spend_Δ%", "ftds_Δ"]].round(2)


# ---------------------------------------------------------------------------
# 15. PAGE: CHANNEL VIEW
# ---------------------------------------------------------------------------

def page_channels(df_curr, df_prev):
    st.header("Channel View")
    if df_curr.empty:
        st.warning("No data in the selected period.")
        return
    dim = st.radio("Group by", ["channel", "source", "attribution_platform"], horizontal=True)
    metric = st.selectbox(
        "Metric to plot",
        ["spend", "ftd_players", "cpa_ftd", "ltv_cac", "pltv", "cpa_apd2",
         "registration_rate", "ftd_rate"],
        index=0,
    )
    curr = aggregate(df_curr, [dim])
    prev = aggregate(df_prev, [dim])
    table = _summary_table(curr, prev, dim)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download (CSV)",
        df_to_csv_bytes(table),
        f"{dim}_summary.csv",
        "text/csv",
    )

    st.subheader(f"Bar — {metric} by {dim}")
    bar_df = curr.sort_values(metric, ascending=False).head(15)
    st.plotly_chart(bar_chart(bar_df, dim, metric, title=f"{metric} by {dim}"), use_container_width=True)

    st.subheader("Time series")
    ts = aggregate(df_curr, [dim, "day"]).sort_values("day")
    if not ts.empty:
        st.plotly_chart(
            line_chart(ts, "day", metric, color=dim, title=f"{metric} over time"),
            use_container_width=True,
        )

    st.subheader("Contribution analysis")
    contrib_cols = ["spend", "ftd_players", "sum_pltv"]
    cdf = curr[[dim] + [c for c in contrib_cols if c in curr.columns]].copy()
    for c in contrib_cols:
        if c in cdf.columns:
            tot = cdf[c].sum()
            cdf[f"{c}_share_%"] = (cdf[c] / tot * 100) if tot else np.nan
    cdf = cdf.sort_values("spend", ascending=False)
    st.dataframe(cdf.round(2), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 16. PAGE: CAMPAIGN / AD EXPLORER
# ---------------------------------------------------------------------------

def page_explorer(df_curr, df_prev):
    st.header("Campaign / Ad Explorer")
    st.caption(
        "Detailed exploration. Pick a grain — defaults to the approach that matches "
        "the level you're grouping by, so you don't mix grains accidentally."
    )

    if df_curr.empty:
        st.warning("No data in the selected period.")
        return

    grain = st.selectbox(
        "Grain (filters approach automatically)",
        ["ad_level", "ad_group_level", "campaign_level", "any (mix grains, careful)"],
        index=0,
    )
    if grain != "any (mix grains, careful)":
        df_curr = df_curr[df_curr["spend_attribution_approach"] == grain]
        df_prev = df_prev[df_prev["spend_attribution_approach"] == grain]
    else:
        st.warning("Mixing grains can double-count spend across approach levels. Read carefully.")

    group_by = st.multiselect(
        "Group by",
        ["day", "week_start", "month_start", "source", "channel", "name",
         "ad_group_name", "ad_name", "spend_row_type"],
        default=["channel", "name"] if grain in ("ad_level", "ad_group_level") else ["channel"],
    )

    min_spend = st.number_input(
        f"Min spend guardrail ({CURRENCY}) — exclude rows below this from rankings",
        value=50.0,
        step=10.0,
    )

    if not group_by:
        st.info("Pick at least one group-by field.")
        return

    agg = aggregate(df_curr, group_by)
    agg = agg[agg["spend"] >= min_spend].copy()
    sort_col = st.selectbox("Sort by", ["spend", "ftd_players", "cpa_ftd", "ltv_cac", "cpa_apd2", "sum_pltv"], index=0)
    asc = sort_col in {"cpa_ftd", "cpa_apd2"}
    agg = agg.sort_values(sort_col, ascending=asc, kind="mergesort").head(500)

    st.dataframe(agg.round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered (CSV)",
        df_to_csv_bytes(agg),
        "explorer.csv",
        "text/csv",
    )

    if {"spend", "ftd_players"}.issubset(agg.columns):
        st.subheader("Efficiency scatter — Spend vs FTDs (size = pLTV)")
        scat_df = agg.copy()
        scat_df["pLTV_size"] = scat_df.get("sum_pltv", 0).clip(lower=0)
        scat_df["pLTV_size"] = scat_df["pLTV_size"].fillna(0).replace(0, 1)
        hover_name = group_by[-1] if group_by[-1] in scat_df.columns else group_by[0]
        st.plotly_chart(
            scatter_chart(
                scat_df, x="spend", y="ftd_players",
                size="pLTV_size", color="cpa_ftd",
                hover_name=hover_name,
                title="Spend vs FTDs (colour = CPA, size = pLTV)",
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# 17. PAGE: DATA QUALITY
# ---------------------------------------------------------------------------

def page_data_quality(df_curr, period, thresholds):
    st.header("Data Quality / Attribution Health")
    if df_curr.empty:
        st.warning("No data in the selected period.")
        return
    dq = data_quality(df_curr, thresholds)

    status = dq["status"]
    color_map = {"Healthy": "#27ae60", "Warning": "#e67e22", "Investigate": "#c0392b"}
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:6px;background:{color_map[status]};color:white;font-weight:600;font-size:18px'>"
        f"Status: {status}</div>",
        unsafe_allow_html=True,
    )
    if dq["notes"]:
        for n in dq["notes"]:
            st.write(f"• {n}")

    st.subheader("Spend by row type (granular grains only)")
    granular = df_curr[df_curr["spend_attribution_approach"].isin(GRANULAR_APPROACHES)]
    if granular.empty:
        st.info("No granular (ad/ad_group) rows in this window — matched/spend_only/residual not applicable.")
    else:
        rt = granular.groupby("spend_row_type", dropna=False)["spend"].agg(["sum", "count"]).reset_index()
        rt.columns = ["spend_row_type", "spend", "rows"]
        total = rt["spend"].sum()
        rt["share_%"] = rt["spend"] / total * 100 if total else np.nan
        st.dataframe(rt.round(2), use_container_width=True, hide_index=True)
        st.plotly_chart(
            bar_chart(rt, "spend_row_type", "spend", title="Spend by row type"),
            use_container_width=True,
        )

    st.subheader("Rows by attribution approach")
    ap = df_curr.groupby("spend_attribution_approach")["spend"].agg(["sum", "count"]).reset_index()
    ap.columns = ["spend_attribution_approach", "spend", "rows"]
    st.dataframe(ap.round(2), use_container_width=True, hide_index=True)

    st.subheader("Tagging coverage")
    cov = dq["tagging_coverage_pct"]
    st.metric(
        "Coverage (campaign id filled / total sessions)",
        f"{cov:.1f}%" if not pd.isna(cov) else "n/a",
        help=f"{int(dq['tagging_filled_sessions']):,} / {int(dq['tagging_total_sessions']):,} sessions",
    )

    st.subheader("Blank rates on key dimensions")
    if dq["blank_rates"]:
        b = pd.DataFrame(
            [(k, v) for k, v in dq["blank_rates"].items()],
            columns=["dimension", "blank_%"],
        ).round(2)
        st.dataframe(b, use_container_width=True, hide_index=True)

    st.subheader("Anomalies")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Rows with spend > 0 but sessions = 0", f"{dq.get('spend_no_sessions_rows', 0):,}",
                  help=fmt_money(dq.get('spend_no_sessions_amount', 0)))
    with cols[1]:
        st.metric("Rows with FTDs > registrations", f"{dq.get('ftd_gt_reg_rows', 0):,}")

    st.subheader("Day-over-day spend movement (current window)")
    daily_spend = df_curr.groupby("day")["spend"].sum().reset_index().sort_values("day")
    daily_spend["dod_change_pct"] = daily_spend["spend"].pct_change() * 100
    st.plotly_chart(
        bar_chart(daily_spend, "day", "dod_change_pct", title="Spend day-over-day % change"),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# 18. PAGE: RECOMMENDATIONS
# ---------------------------------------------------------------------------

def page_recommendations(df_curr, df_prev, thresholds):
    st.header("Recommendations")
    st.caption(
        "Rules-based. Each row shows the recommendation and the reasons. "
        "Confidence reflects spend volume and FTD count, not statistical significance. "
        "Treat as a triage list, not a directive."
    )
    if df_curr.empty:
        st.warning("No data in the selected period.")
        return

    grain = st.radio("Recommend at", ["channel", "channel + source", "ad_group_name", "ad_name", "name (campaign)"], horizontal=True)
    grain_map = {
        "channel": ["channel"],
        "channel + source": ["channel", "source"],
        "ad_group_name": ["ad_group_name"],
        "ad_name": ["ad_name"],
        "name (campaign)": ["name"],
    }
    cols = grain_map[grain]
    if grain in ("ad_group_name",):
        df_curr_g = df_curr[df_curr["spend_attribution_approach"] == "ad_group_level"]
        df_prev_g = df_prev[df_prev["spend_attribution_approach"] == "ad_group_level"]
    elif grain == "ad_name":
        df_curr_g = df_curr[df_curr["spend_attribution_approach"] == "ad_level"]
        df_prev_g = df_prev[df_prev["spend_attribution_approach"] == "ad_level"]
    elif grain == "name (campaign)":
        df_curr_g = df_curr[df_curr["spend_attribution_approach"].isin(["campaign_level", "ad_group_level", "ad_level"])]
        df_prev_g = df_prev[df_prev["spend_attribution_approach"].isin(["campaign_level", "ad_group_level", "ad_level"])]
    else:
        df_curr_g = df_curr
        df_prev_g = df_prev

    recs = build_recommendations(df_curr_g, df_prev_g, cols, thresholds)
    counts = recs["recommendation"].value_counts().to_dict()
    cols_kpi = st.columns(4)
    cols_kpi[0].metric("Scale", counts.get("Scale", 0))
    cols_kpi[1].metric("Hold", counts.get("Hold", 0))
    cols_kpi[2].metric("Investigate", counts.get("Investigate", 0))
    cols_kpi[3].metric("Brand", counts.get("Brand", 0),
                       help="Brand/above-the-line channels excluded from performance scrutiny")

    flt = st.multiselect(
        "Filter recommendation",
        ["Scale", "Hold", "Investigate", "Brand"],
        default=["Scale", "Investigate"],
    )
    show = recs[recs["recommendation"].isin(flt)] if flt else recs
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "Download recommendations (CSV)",
        df_to_csv_bytes(recs),
        "recommendations.csv",
        "text/csv",
    )


# ---------------------------------------------------------------------------
# 19. PAGE: METRIC DICTIONARY
# ---------------------------------------------------------------------------

def page_dictionary():
    st.header("Metric Dictionary / Rules")
    st.markdown(
        """
### Core metrics

- **Spend** — `sum(spend)` across all rows in the selection. Includes matched, spend_only, and residual on granular grains.
- **FTDs** — First-time depositors. `sum(ftd_players)`.
- **LTV:CAC** — `sum(sum_pltv) / sum(spend)`. By default, on granular grains (ad/ad_group), this uses **matched rows only** so the numerator and denominator are aligned. Toggle "include unmatched" to relax.
- **CPA (FTD)** — `sum(spend) / sum(ftd_players)`.
- **pLTV** — predicted LTV. `sum(sum_pltv)`.
- **APD1 / first week APD** — sum of `sum_apd_first_week`. This is a value field, not a count.
- **APD2+ players** — count of players with ≥2 active player days.
- **CPA APD2+** — `sum(spend) / sum(apd_2_players)`.
- **Cost per session / per registration** — divide-by-zero protected.
- **Registration rate** — `registrations / num_sessions`.
- **FTD rate** — `ftd_players / num_sessions`.
- **Tagging coverage** — `tagging_session_campaign_id_filled / tagging_total_sessions`. Indicates how often we successfully tagged a session with a campaign id.

### Row types (`spend_row_type`)

These only apply to **granular grains** — `ad_group_level` and `ad_level` rows. They are NaN on channel/campaign/affiliate-level rows because those rows are already aggregated.

- **matched** — spend and attributed performance are linked on the row.
- **spend_only** — spend exists but no attributed performance is linked. Either the click happened but tracking missed the conversion, or the conversion was attributed elsewhere.
- **residual** — leftover spend that could not be cleanly allocated at that grain.

Total spend always includes all three. Efficiency metrics (CPA, LTV:CAC) default to matched-only on granular grains. The toggle "Include unmatched/residual in efficiency metrics" overrides this.

### Grains (`spend_attribution_approach`)

- **affiliate_level** — affiliate breakdowns.
- **channel_level** — channel rollups (covers organic, RAF, CRM, ATL, etc., and residual paid spend).
- **campaign_level** — paid media at campaign granularity.
- **ad_group_level** — paid media at ad group granularity.
- **ad_level** — paid media at ad granularity.

These are alternative breakdowns, not nested. For a channel rollup you can sum across all approach levels (e.g., Meta App total = Meta App ad_level + Meta App channel_level rows). Within a single approach, rows are mutually exclusive.

### Misleading comparisons to avoid

- **Don't** compare an ad-level CPA to a channel-level CPA without context — the channel total includes residual/unallocated spend that isn't on any ad row.
- **Don't** rank ads by CPA when their spend is below a guardrail; tiny denominators produce extreme values. Use the min-spend guardrail in the explorer.
- **Don't** read APD2+ figures as conversion to date — `apd_2_players` requires sufficient time for second-day activity. Recent days will under-report. Same for `sum_apd_first_week`.
- **Don't** treat "Unattributed" channel as a normal channel; it's a residual bucket.
- **ATL** spend has no direct attribution. CPAs here will be infinite or undefined and recommendations will be weak.

### How recommendations work

Heuristic only — they support, not replace, judgement.

- **Scale** — spend ≥ gate, FTDs ≥ minimum, LTV:CAC ≥ floor, AND either CPA improved enough OR LTV:CAC improving.
- **Investigate** — at least one of: CPA deteriorated past threshold, LTV:CAC deteriorated past threshold, APD2+ CPA deteriorated past threshold.
- **Hold** — everything else, or below the spend gate.
- **Brand** — channels in `BRAND_CHANNELS` env var (default: `ATL`). These are above-the-line / brand / awareness channels where click-through attribution doesn't apply, so they're excluded from CPA/LTV:CAC scrutiny. Spend still shows in summaries; just no Scale/Hold/Investigate verdict.

Confidence labels reflect spend × FTD volume only. They are **not** statistical significance.

### Period comparison

Default mode is "same window in previous month": e.g. 1–28 April 2026 → 1–28 March 2026. Other modes available: previous year (same month, prior year) and immediately preceding equal-length window (GA4-style).
        """
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
