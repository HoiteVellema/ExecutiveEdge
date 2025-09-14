#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocktwits Success-Rate — Online Dashboard (Streamlit)

Features
- Enter a Stocktwits username (default: ExecutiveEdge) and optional access token
- Pull user messages via Stocktwits REST API (no scraping)
- Parse tickers, direction (Bullish/Bearish or keyword heuristic), price targets, multipliers
- Compute daily forward-return success rates (EOD horizons)
- Optional intraday backtests (1-minute) for recent ~7 days using yfinance
- Capture "first bullish mention" 1m price, plus parsed target/multiplier
- Interactive KPIs, tables, and CSV downloads

Run locally
    pip install -r requirements.txt
    streamlit run app.py
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
from dateutil import parser as dtp
from datetime import datetime, timedelta, timezone

# ==============================
# Config & Heuristics
# ==============================
BASE_URL = os.environ.get("ST_BASE_URL", "https://api.stocktwits.com/api/2")
DEFAULT_USER = "ExecutiveEdge"

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
# Stocktwits Client
# ==============================
class StocktwitsClient:
    def __init__(self, access_token: Optional[str] = None, base_url: str = BASE_URL, session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.token = access_token

    def _get(self, path: str, **params) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if self.token:
            params.setdefault("access_token", self.token)
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

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
# Intraday utilities
# ==============================

def _yf_intraday_1m(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    now_utc = datetime.now(timezone.utc)
    window_start = max(start - timedelta(days=1), now_utc - timedelta(days=7))
    window_end = min(end + timedelta(days=1), now_utc)
    try:
        df = yf.download(symbol, period="7d", interval="1m", prepost=True, progress=False, auto_adjust=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.index = pd.DatetimeIndex(df.index.tz_localize("UTC"))
            return df[(df.index >= pd.Timestamp(window_start)) & (df.index <= pd.Timestamp(window_end))]
    except Exception:
        pass
    return None


def _atr_tr(hi: pd.Series, lo: pd.Series, cl: pd.Series, n: int = 14) -> pd.Series:
    prev_close = cl.shift(1)
    tr = pd.concat([
        hi - lo,
        (hi - prev_close).abs(),
        (lo - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n//2)).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    if "Volume" not in df.columns:
        return df["Close"].copy()
    pv = (df["Close"] * df["Volume"]).cumsum()
    vv = (df["Volume"]).cumsum().replace(0, np.nan)
    return pv / vv


def backtest_fill(df: pd.DataFrame, entry_time: pd.Timestamp, entry_px: float, stop_px: float, target_px: float, direction: int, horizon_mins: int):
    start_idx = df.index.get_indexer([entry_time], method="nearest")[0]
    horizon_end = entry_time + pd.Timedelta(minutes=horizon_mins)
    path = df.iloc[start_idx:]
    for ts, row in path.iterrows():
        if ts > horizon_end:
            break
        hi, lo = float(row["High"]), float(row["Low"])
        if direction > 0:
            if lo <= stop_px:
                r = (stop_px - entry_px) / (entry_px - stop_px)
                return True, ts, float(stop_px), "stop", float(r)
            if hi >= target_px:
                r = (target_px - entry_px) / (entry_px - stop_px)
                return True, ts, float(target_px), "target", float(r)
        else:
            if hi >= stop_px:
                r = (entry_px - stop_px) / (stop_px - entry_px)
                return True, ts, float(stop_px), "stop", float(r)
            if lo <= target_px:
                r = (entry_px - target_px) / (stop_px - entry_px)
                return True, ts, float(target_px), "target", float(r)
    last = path[path.index <= horizon_end].tail(1)
    if last.empty:
        return False, None, None, None, None
    exit_px = float(last["Close"].iloc[0])
    if direction > 0:
        r = (exit_px - entry_px) / (entry_px - stop_px)
    else:
        r = (entry_px - exit_px) / (stop_px - entry_px)
    return True, last.index[0], exit_px, "time", float(r)


def simulate_intraday(call: Call, df: pd.DataFrame, horizon_mins: int, strategies: List[str]):
    rows = []
    after = df[df.index > pd.Timestamp(call.created_at)]
    if after.empty:
        return rows
    first_bar_idx = after.index[0]
    prev_bar = df.loc[:first_bar_idx].tail(2).iloc[0] if len(df.loc[:first_bar_idx]) >= 2 else None
    atr = _atr_tr(df["High"], df["Low"], df["Close"]).reindex(df.index)
    vwap = _vwap(df).reindex(df.index)

    def strategy_row(strategy, entry_time, entry_px, stop_px, target_px, rr):
        filled, exit_time, exit_px, exit_reason, r_realized = backtest_fill(df, entry_time, entry_px, stop_px, target_px, call.direction, horizon_mins)
        return {
            "strategy": strategy,
            "message_id": call.message_id,
            "symbol": call.symbol,
            "created_at_utc": call.created_at,
            "entry_time": entry_time,
            "entry_px": entry_px,
            "stop_px": stop_px,
            "target_px": target_px,
            "rr": rr,
            "filled": filled,
            "exit_time": exit_time,
            "exit_px": exit_px,
            "exit_reason": exit_reason,
            "r_realized": r_realized,
        }

    # nextclose
    if "nextclose" in strategies:
        entry_time = first_bar_idx
        entry_px = float(df.loc[entry_time, "Close"])
        risk = float(atr.loc[entry_time] or (0.005 * entry_px))
        stop_px = entry_px - (1.5 * risk) if call.direction > 0 else entry_px + (1.5 * risk)
        tgt_from_msg = call.target_price
        if call.multiplier and call.multiplier > 0:
            tgt_from_msg = max(tgt_from_msg or 0, entry_px * call.multiplier)
        target_px = (entry_px + 2.0 * (entry_px - stop_px)) if call.direction > 0 else (entry_px - 2.0 * (stop_px - entry_px))
        if tgt_from_msg:
            target_px = tgt_from_msg if call.direction > 0 else (entry_px - abs(tgt_from_msg - entry_px))
        rr = abs((target_px - entry_px) / (entry_px - stop_px)) if (entry_px != stop_px) else np.nan
        rows.append(strategy_row("nextclose", entry_time, entry_px, stop_px, target_px, rr))

    # breakout
    if "breakout" in strategies and prev_bar is not None:
        trigger = float(prev_bar["High"]) * 1.0005 if call.direction > 0 else float(prev_bar["Low"]) * 0.9995
        sub = df[df.index >= first_bar_idx]
        entry_time, entry_px = None, None
        for ts, row in sub.iterrows():
            if call.direction > 0 and row["High"] >= trigger:
                entry_time, entry_px = ts, float(max(trigger, row["Open"]))
                break
            if call.direction < 0 and row["Low"] <= trigger:
                entry_time, entry_px = ts, float(min(trigger, row["Open"]))
                break
        if entry_time is not None:
            risk = float(atr.loc[entry_time] or (0.006 * entry_px))
            stop_px = entry_px - (1.5 * risk) if call.direction > 0 else entry_px + (1.5 * risk)
            tgt_from_msg = call.target_price
            if call.multiplier and call.multiplier > 0:
                tgt_from_msg = max(tgt_from_msg or 0, entry_px * call.multiplier)
            target_px = (entry_px + 2.0 * (entry_px - stop_px)) if call.direction > 0 else (entry_px - 2.0 * (stop_px - entry_px))
            if tgt_from_msg:
                target_px = tgt_from_msg if call.direction > 0 else (entry_px - abs(tgt_from_msg - entry_px))
            rr = abs((target_px - entry_px) / (entry_px - stop_px)) if (entry_px != stop_px) else np.nan
            rows.append(strategy_row("breakout", entry_time, entry_px, stop_px, target_px, rr))

    # vwap
    if "vwap" in strategies:
        entry_time = first_bar_idx
        v = float(vwap.loc[entry_time]) if pd.notna(vwap.loc[entry_time]) else float(df.loc[entry_time, "Close"])
        bar = df.loc[entry_time]
        filled = (bar["Low"] <= v <= bar["High"])  # symmetric check
        if filled:
            entry_px = v
            risk = float(atr.loc[entry_time] or (0.005 * entry_px))
            stop_px = entry_px - (1.2 * risk) if call.direction > 0 else entry_px + (1.2 * risk)
            tgt_from_msg = call.target_price
            if call.multiplier and call.multiplier > 0:
                tgt_from_msg = max(tgt_from_msg or 0, entry_px * call.multiplier)
            target_px = (entry_px + 2.0 * (entry_px - stop_px)) if call.direction > 0 else (entry_px - 2.0 * (stop_px - entry_px))
            if tgt_from_msg:
                target_px = tgt_from_msg if call.direction > 0 else (entry_px - abs(tgt_from_msg - entry_px))
            rr = abs((target_px - entry_px) / (entry_px - stop_px)) if (entry_px != stop_px) else np.nan
            rows.append(strategy_row("vwap", entry_time, entry_px, stop_px, target_px, rr))
        else:
            rows.append({
                "strategy": "vwap", "message_id": call.message_id, "symbol": call.symbol,
                "created_at_utc": call.created_at, "entry_time": entry_time, "entry_px": float(v),
                "stop_px": np.nan, "target_px": np.nan, "rr": np.nan,
                "filled": False, "exit_time": None, "exit_px": None, "exit_reason": "no_fill", "r_realized": None,
            })
    return rows


def capture_first_mentions(calls: List[Call], intraday_fetcher, horizon_mins: int = 60) -> pd.DataFrame:
    df_rows: List[Dict] = []
    bullish = [c for c in calls if c.direction > 0]
    if not bullish:
        return pd.DataFrame()
    by_symbol: Dict[str, List[Call]] = {}
    for c in bullish:
        by_symbol.setdefault(c.symbol, []).append(c)
    for sym, lst in by_symbol.items():
        c0 = sorted(lst, key=lambda x: x.created_at)[0]
        start = c0.created_at - timedelta(minutes=10)
        end = c0.created_at + timedelta(minutes=horizon_mins)
        idata = intraday_fetcher(sym, start, end)
        if idata is None or idata.empty:
            continue
        after = idata[idata.index > pd.Timestamp(c0.created_at)]
        if after.empty:
            continue
        first_bar = after.iloc[0]
        first_close = float(first_bar["Close"])
        df_rows.append({
            "symbol": sym,
            "first_bullish_message_id": c0.message_id,
            "first_bullish_at_utc": c0.created_at,
            "first_mention_entry_1m": first_close,
            "text_target": c0.target_price,
            "text_multiplier": c0.multiplier,
        })
    return pd.DataFrame(df_rows)

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
        access_token = st.text_input("Access token (optional)", type="password", help="Stocktwits OAuth token if you have one.")
        since = st.date_input("Since (UTC)", value=pd.to_datetime("2024-01-01")).isoformat()[:10]
        max_posts = st.slider("Max posts to scan", min_value=100, max_value=10000, value=3000, step=100)
        horizons = st.multiselect("Daily horizons (trading days)", options=[1, 3, 5, 10, 20, 60], default=[1, 5, 20])
        st.divider()
        intraday = st.toggle("Enable intraday backtests (1m, recent ~7d)", value=True)
        i_horizon = st.number_input("Intraday horizon (mins)", min_value=30, max_value=1200, value=390, step=30)
        strategies = st.multiselect("Intraday strategies", options=["nextclose", "breakout", "vwap"], default=["nextclose", "breakout", "vwap"])
        run = st.button("🚀 Fetch & Compute", type="primary")

    if not run:
        st.info("Set parameters in the sidebar and click **Fetch & Compute**.")
        st.stop()

    # Fetch & parse
    st.subheader(f"User: @{username}")

    st.toast("Pulling Stocktwits messages…")
    client = StocktwitsClient(access_token=access_token)

    with st.spinner("Contacting Stocktwits API…"):
        calls = harvest_calls(client, username, max_posts=max_posts, since=since)
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

    # Intraday
    if intraday:
        st.markdown("---")
        st.markdown("## Intraday Backtests (1-minute)")
        st.caption("Only recent ~7 days are available via yfinance. Older posts may not produce intraday trades.")
        adv_rows = []
        first_mentions_df = pd.DataFrame()
        with st.spinner("Fetching 1m bars & simulating entries…"):
            def fetcher(sym: str, start: datetime, end: datetime):
                return _yf_intraday_1m(sym, start, end)
            first_mentions_df = capture_first_mentions(calls, fetcher, horizon_mins=60)
            for c in calls:
                start = c.created_at - timedelta(minutes=30)
                end = c.created_at + timedelta(minutes=i_horizon + 5)
                idata = fetcher(c.symbol, start, end)
                if idata is None or idata.empty:
                    continue
                adv_rows.extend(simulate_intraday(c, idata, i_horizon, strategies))
        if adv_rows:
            adv_df = pd.DataFrame(adv_rows)
            # Summary by strategy
            filled = adv_df[adv_df["filled"]]
            if not filled.empty:
                wins = filled[filled["exit_reason"] == "target"].groupby("strategy").size()
                losses = filled[filled["exit_reason"] == "stop"].groupby("strategy").size()
                totals = filled.groupby("strategy").size()
                hr = (wins / totals).fillna(0).rename("hit_rate").reset_index()
                avg_r = filled.groupby("strategy")["r_realized"].mean().rename("avg_R").reset_index()
                intraday_summary = pd.merge(hr, avg_r, on="strategy", how="outer").fillna(0)
            else:
                intraday_summary = pd.DataFrame({"strategy": strategies, "hit_rate": 0, "avg_R": 0})

            st.markdown("### Intraday Summary (by strategy)")
            st.dataframe(intraday_summary, use_container_width=True)

            st.markdown("### Intraday Trades (sample)")
            st.dataframe(adv_df.head(1000), use_container_width=True)

            st.download_button("Download intraday trades CSV", data=adv_df.to_csv(index=False), file_name=f"intraday_trades_{username}.csv", mime="text/csv")
            st.download_button("Download intraday summary CSV", data=intraday_summary.to_csv(index=False), file_name=f"intraday_summary_{username}.csv", mime="text/csv")
        else:
            st.info("No intraday rows produced (likely outside 7d window or missing 1m data).")

        if not first_mentions_df.empty:
            st.markdown("### First Bullish Mentions")
            st.dataframe(first_mentions_df, use_container_width=True)
            st.download_button("Download first-mentions CSV", data=first_mentions_df.to_csv(index=False), file_name=f"first_mentions_{username}.csv", mime="text/csv")

    st.success("Done.")


if __name__ == "__main__":
    main()
