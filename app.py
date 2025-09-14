#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocktwits Success-Rate — Online Dashboard (Streamlit) — Robust HTTP Handling

Changes in this build
- Retries with backoff for 429/5xx and respect `Retry-After`.
- Friendly error messages in the UI instead of a crash on HTTPError.
- Default headers incl. User-Agent/Accept.
- Optional Stocktwits access token read from Streamlit **Secrets** (`ST_ACCESS_TOKEN`) or sidebar.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dateutil import parser as dtp
from datetime import datetime, timedelta, timezone

# ==============================
# Config & Heuristics
# ==============================
BASE_URL = os.environ.get("ST_BASE_URL", "https://api.stocktwits.com/api/2")
DEFAULT_USER = "ExecutiveEdge"

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "StocktwitsDashboard/1.0 (+Streamlit)",
}

BULL_KEYWORDS = re.compile(r"\b(long|buy|adding|accum(ulate|ing)|calls?|call\s*spreads?|bull(ish)?)\b", re.I)
BEAR_KEYWORDS = re.compile(r"\b(short|sell|trim(ming)?|puts?|put\s*spreads?|bear(ish)?|fade|dump(ing)?)\b", re.I)
PT_DOLLAR = re.compile(r"(?:(?:pt|price\s*target|target)\s*[:=]?\s*\$?)(\d{1,6}(?:\.\d{1,2})?)", re.I)
PT_ARROW = re.compile(r"(?:->|to)\s*\$?(\d{1,6}(?:\.\d{1,2})?)", re.I)
MULT_X = re.compile(r"\b(\d+(?:\.\d+)?)\s*x\b|\bx\s*(\d+(?:\.\d+)?)\b|\b(double|triple)\b", re.I)

US_EASTERN = pytz.timezone("US/Eastern")

@dataclass
class Call:
    message_id: int
    created_at: datetime  # UTC
    username: str
    symbol: str
    direction: int  # +1 bullish, -1 bearish
    source: str     # "sentiment" or "heuristic"
    body: str
    target_price: Optional[float] = None
    multiplier: Optional[float] = None

# ==============================
# HTTP session with retries
# ==============================
def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# ==============================
# Stocktwits Client
# ==============================
class StocktwitsClient:
    def __init__(self, access_token: Optional[str] = None, base_url: str = BASE_URL, session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.session = session or build_session()
        self.token = access_token

    def _get(self, path: str, **params) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if self.token:
            params.setdefault("access_token", self.token)
        try:
            r = self.session.get(url, params=params, timeout=30, headers=DEFAULT_HEADERS)
            # Handle common API errors gracefully
            if r.status_code == 404:
                raise RuntimeError("Stocktwits API returned 404 — user not found or endpoint moved.")
            if r.status_code == 401 or r.status_code == 403:
                raise RuntimeError("Stocktwits API returned 401/403 — authentication required or token invalid. Add a valid access token in the sidebar or Streamlit Secrets.")
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After", "a bit")
                raise RuntimeError(f"Stocktwits API rate limit hit (429). Try again after {retry_after}, reduce Max posts, or add an access token.")
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            # Sanitize details for Streamlit Cloud
            raise RuntimeError(f"HTTP error from Stocktwits API (status {r.status_code}). Please check your parameters or add an access token.") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError("Network error contacting Stocktwits API. Check internet access and try again.") from e

    def iter_user_messages(self, username: str, max_posts: int = 2000, since: Optional[str] = None) -> Iterable[dict]:
        fetched = 0
        max_id = None
        cutoff = None
        if since:
            try:
                cutoff = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
            except Exception:
                cutoff = None
        while True:
            params = {"limit": 30}
            if max_id:
                params["max"] = max_id
            data = self._get(f"streams/user/{username}.json", **params)
            msgs = data.get("messages", [])
            if not msgs:
                break
            for m in msgs:
                fetched += 1
                yield m
                if fetched >= max_posts:
                    return
                if cutoff is not None:
                    created = dtp.parse(m.get("created_at", "")).astimezone(timezone.utc)
                    if created < cutoff:
                        return
            cur = data.get("cursor", {})
            if not cur or not cur.get("more"):
                break
            max_id = cur.get("max")

# ==============================
# Parsing helpers
# ==============================
def _extract_direction_and_meta(msg: dict) -> Tuple[Optional[int], str]:
    ent = msg.get("entities") or {}
    sent = (ent.get("sentiment") or {}).get("basic") if isinstance(ent, dict) else None
    if isinstance(sent, str):
        s = sent.lower()
        if s == "bullish":
            return +1, "sentiment"
        if s == "bearish":
            return -1, "sentiment"
    body = (msg.get("body") or "")
    if BULL_KEYWORDS.search(body):
        return +1, "heuristic"
    if BEAR_KEYWORDS.search(body):
        return -1, "heuristic"
    return None, "unknown"


def _extract_symbols(msg: dict) -> List[str]:
    syms: List[str] = []
    for s in msg.get("symbols") or []:
        sym = s.get("symbol") or s.get("id")
        if isinstance(sym, str):
            syms.append(sym.upper())
    body = msg.get("body", "") or ""
    for match in re.findall(r"\$([A-Za-z][A-Za-z0-9\.-]{0,9})", body):
        syms.append(match.upper())
    return sorted(set(syms))


def _extract_targets_and_multipliers(body: str) -> Tuple[Optional[float], Optional[float]]:
    body = body or ""
    m = PT_DOLLAR.search(body) or PT_ARROW.search(body)
    target = float(m.group(1)) if m else None
    mx = MULT_X.search(body)
    mult = None
    if mx:
        if mx.group(1):
            mult = float(mx.group(1))
        elif mx.group(2):
            mult = float(mx.group(2))
        else:
            word = mx.group(3).lower()
            mult = 2.0 if word == "double" else (3.0 if word == "triple" else None)
    return target, mult


def harvest_calls(client: StocktwitsClient, username: str, max_posts: int, since: Optional[str]) -> List[Call]:
    calls: List[Call] = []
    for m in client.iter_user_messages(username, max_posts=max_posts, since=since):
        direction, source = _extract_direction_and_meta(m)
        if direction is None:
            continue
        syms = _extract_symbols(m)
        if not syms:
            continue
        body = (m.get("body") or "").strip()
        target, mult = _extract_targets_and_multipliers(body)
        created = dtp.parse(m["created_at"]).astimezone(timezone.utc)
        for sym in syms:
            calls.append(Call(
                message_id=int(m.get("id")),
                created_at=created,
                username=username,
                symbol=sym,
                direction=direction,
                source=source,
                body=body,
                target_price=target,
                multiplier=mult,
            ))
    # de-dup by (message_id, symbol)
    key = set()
    dedup: List[Call] = []
    for c in calls:
        k = (c.message_id, c.symbol)
        if k not in key:
            key.add(k)
            dedup.append(c)
    return dedup

# ==============================
# Daily (EOD) scoring
# ==============================
def next_daily_close_after(ts_utc: datetime) -> datetime:
    close_hour_utc = 21  # ~16:00 ET during DST; simplification
    d = ts_utc.astimezone(timezone.utc)
    if d.hour < close_hour_utc:
        entry = d.date()
    else:
        entry = (d + timedelta(days=1)).date()
    return datetime(entry.year, entry.month, entry.day, close_hour_utc, 0, 0, tzinfo=timezone.utc)


def _yf_daily(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, start=start.date(), end=(end.date() + timedelta(days=1)), interval="1d", progress=False, auto_adjust=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.index = pd.DatetimeIndex(df.index.tz_localize("UTC"))
            return df
    except Exception:
        pass
    return None


def score_daily(calls: List[Call], horizons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    by_symbol: Dict[str, List[Call]] = {}
    for c in calls:
        by_symbol.setdefault(c.symbol, []).append(c)
    for sym, sym_calls in by_symbol.items():
        min_ts = min(c.created_at for c in sym_calls)
        max_ts = max(c.created_at for c in sym_calls) + timedelta(days=max(horizons) + 5)
        hist = _yf_daily(sym, min_ts - timedelta(days=3), max_ts)
        if hist is None or hist.empty:
            continue
        for c in sym_calls:
            entry_dt = next_daily_close_after(c.created_at)
            entry_row = hist[hist.index >= pd.Timestamp(entry_dt)].head(1)
            if entry_row.empty:
                continue
            entry_px = float(entry_row["Close"].iloc[0])
            entry_idx = entry_row.index[0]
            for h in horizons:
                exit_dt = entry_idx + pd.Timedelta(days=h)
                exit_row = hist[hist.index >= exit_dt].head(1)
                if exit_row.empty:
                    continue
                exit_px = float(exit_row["Close"].iloc[0])
                ret = (exit_px - entry_px) / entry_px
                rows.append({
                    "message_id": c.message_id,
                    "username": c.username,
                    "symbol": sym,
                    "created_at_utc": c.created_at,
                    "direction": c.direction,
                    "source": c.source,
                    "horizon_days": h,
                    "entry_date": entry_idx,
                    "exit_date": exit_row.index[0],
                    "entry": entry_px,
                    "exit": exit_px,
                    "ret": ret * (1.0 if c.direction > 0 else -1.0),
                })
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, trades, trades

    def agg(df: pd.DataFrame) -> pd.Series:
        signed = df["ret"].values
        wins = (signed > 0).mean() if len(signed) else np.nan
        avg = float(np.mean(signed)) if len(signed) else np.nan
        med = float(np.median(signed)) if len(signed) else np.nan
        pos = np.sum(signed[signed > 0]) if len(signed) else 0.0
        neg = -np.sum(signed[signed < 0]) if len(signed) else 0.0
        pf = (pos / neg) if neg > 0 else np.nan
        n = int(df.shape[0])
        return pd.Series({
            "trades": n,
            "hit_rate": wins,
            "avg_ret": avg,
            "median_ret": med,
            "profit_factor": pf,
        })

    summary = trades.groupby(["horizon_days"]).apply(agg).reset_index()
    per_symbol = trades.groupby(["symbol", "horizon_days"]).apply(agg).reset_index()
    return trades, summary, per_symbol

# ==============================
# Intraday utilities (optional section removed to keep this fix focused)
# ==============================

# ==============================
# UI
# ==============================
def main():
    st.set_page_config(page_title="Stocktwits Success-Rate", layout="wide")
    st.title("📈 Stocktwits Success-Rate Dashboard")
    st.caption("Research/education only. Uses Stocktwits REST API + yfinance.")

    with st.sidebar:
        st.header("Parameters")
        username = st.text_input("Stocktwits username", value=DEFAULT_USER)
        # Prefer secret if provided; fallback to sidebar input
        secret_token = st.secrets.get("ST_ACCESS_TOKEN", None)
        access_token = st.text_input("Access token (optional)", type="password", help="If you see 401/403/429 errors, add a Stocktwits OAuth token here or via Secrets.")
        if not access_token and secret_token:
            access_token = secret_token
            st.info("Using ST_ACCESS_TOKEN from Streamlit Secrets.")
        since = st.date_input("Since (UTC)", value=pd.to_datetime("2024-01-01")).isoformat()[:10]
        max_posts = st.slider("Max posts to scan", min_value=100, max_value=10000, value=2000, step=100)
        horizons = st.multiselect("Daily horizons (trading days)", options=[1, 3, 5, 10, 20, 60], default=[1, 5, 20])
        run = st.button("🚀 Fetch & Compute", type="primary")

    if not run:
        st.info("Set parameters in the sidebar and click **Fetch & Compute**.")
        st.stop()

    st.subheader(f"User: @{username}")
    client = StocktwitsClient(access_token=access_token)

    # Fetch & parse with friendly error handling
    try:
        with st.spinner("Contacting Stocktwits API…"):
            calls = harvest_calls(client, username, max_posts=max_posts, since=since)
    except RuntimeError as e:
        st.error(
            "Could not fetch data from Stocktwits.\n\n"
            f"**Reason:** {e}\n\n"
            "Possible fixes:\n"
            "- Verify the username is correct and public.\n"
            "- Reduce **Max posts** (rate limits) or try later.\n"
            "- Add a valid **access token** (sidebar) or set `ST_ACCESS_TOKEN` in **Secrets**.\n"
            "- Ensure outbound internet is allowed by your host."
        )
        st.stop()

    if not calls:
        st.warning("No calls harvested — consider increasing Max posts or changing the date range.")
        st.stop()

    calls_df = pd.DataFrame([c.__dict__ for c in calls])

    # Daily scoring
    with st.spinner("Scoring daily horizons…"):
        trades, summary, per_symbol = score_daily(calls, horizons)

    if summary.empty:
        st.warning("No tradable symbols could be scored on daily horizons.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        total_trades = int(trades.shape[0])
        c1.metric("Total trades", f"{total_trades}")
        c2.metric("Best hit-rate", f"{summary['hit_rate'].max():.1%}")
        c3.metric("Best avg return", f"{summary['avg_ret'].max():.2%}")
        c4.metric("Best profit factor", f"{summary['profit_factor'].max():.2f}")

        st.markdown("### Daily Summary (by horizon)")
        st.dataframe(summary, use_container_width=True)

        st.markdown("### Daily Per-Symbol")
        st.dataframe(per_symbol, use_container_width=True)

        # Downloads
        st.download_button("Download calls CSV", data=calls_df.to_csv(index=False), file_name=f"calls_{username}.csv", mime="text/csv")
        st.download_button("Download daily trades CSV", data=trades.to_csv(index=False), file_name=f"daily_trades_{username}.csv", mime="text/csv")
        st.download_button("Download daily summary CSV", data=summary.to_csv(index=False), file_name=f"daily_summary_{username}.csv", mime="text/csv")
        st.download_button("Download per-symbol CSV", data=per_symbol.to_csv(index=False), file_name=f"daily_per_symbol_{username}.csv", mime="text/csv")

    st.success("Done.")

if __name__ == "__main__":
    main()
