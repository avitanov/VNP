#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_price_and_news.py  –  верзија 3
────────────────────────────────────────────────────────────────────────
Спој 5-минутни цени и Alpha-Vantage вести со следни правила

▪ 00 : 00 – 03 : 59  → 04 : 00 истиот ден
▪ 04 : 00            → 04 : 05
▪ 04 : 01 – 19 : 55  → претстојната 5-мин рамка (ceil)
▪ 20 : 00 – 23 : 59  → 04 : 00 првиот нареден трејдинг-ден
▪ Викенд / празник   → 04 : 00 првиот нареден трејдинг-ден

Агрегација по бар
────────────────
sent_mean        – просек на sentiment
sent_max_abs     – најекстремен |sentiment|
bullish_cnt      – број Bullish + Somewhat-Bullish вести
bearish_cnt      – број Bearish + Somewhat-Bearish вести
headline_count   – вкупно вести
"""

from __future__ import annotations
import os
import pandas as pd
from datetime import time

# ─── CONFIG ──────────────────────────────────────────────────────────
SYMBOL      = "AAPL"
DATA_DIR    = "../dataset/raw/merged"
PRICE_CSV   = os.path.join(DATA_DIR, f"{SYMBOL}_PRICE_FULL.csv")
NEWS_CSV    = os.path.join(DATA_DIR,  f"{SYMBOL}_NEWS_FULL.csv")
OUT_CSV     = os.path.join(DATA_DIR,  f"{SYMBOL}_PRICE_WITH_NEWS.csv")

TRADING_START = time(4, 0)     # 04:00
TRADING_END   = time(19, 55)   # последен бар почнува 19:55
# ─────────────────────────────────────────────────────────────────────


# ─── I/O ─────────────────────────────────────────────────────────────
def load_prices(path: str) -> pd.DataFrame:
    return (pd.read_csv(path, parse_dates=["datetime"])
              .set_index("datetime")
              .sort_index())


def load_news(path: str, symbol: str) -> pd.DataFrame:
    return (pd.read_csv(path, parse_dates=["time_published"])
              .query("symbol == @symbol")
              .sort_values("time_published"))


# ─── BAR-MAPPING HELPERS ─────────────────────────────────────────────
def next_session_open(ts: pd.Timestamp,
                      price_idx: pd.DatetimeIndex) -> pd.Timestamp | pd.NaT:
    """
    Враќа 04:00 на првиот ден ≥ ts што постои во price_idx.
    Ако нема таков (податоците завршуваат) → NaT.
    """
    target = ts.normalize() + pd.Timedelta(hours=4)
    # сите 04:00 барови во индексот
    open_bars = price_idx[price_idx.indexer_between_time("04:00", "04:00")]
    pos = open_bars.searchsorted(target, side="left")
    return open_bars[pos] if pos < len(open_bars) else pd.NaT


def map_to_bar(ts: pd.Timestamp,
               price_idx: pd.DatetimeIndex) -> pd.Timestamp | pd.NaT:
    """Бизнис­-логика за мапирање news-timestamp → 5-мин бар."""
    if ts.time() < TRADING_START:                         # ноќ
        return next_session_open(ts, price_idx)

    if ts.time() >= time(20, 0):                          # after-hours
        return next_session_open(ts + pd.Timedelta(days=1), price_idx)

    # внатре trading hours (04:00 – 19:55)
    if ts.time() == TRADING_START:                        # точно 04:00
        ts += pd.Timedelta(minutes=5)

    # ceil кон следна 5-мин рамка
    pos = price_idx.searchsorted(ts, side="left")
    if pos < len(price_idx):
        return price_idx[pos]

    # timestamp е по последниот бар во табелата
    return next_session_open(ts, price_idx)


# ─── NEWS → BAR AGGREGATION ──────────────────────────────────────────
def aggregate_news(news: pd.DataFrame,
                   price_idx: pd.DatetimeIndex) -> pd.DataFrame:
    news["bar_dt"] = news["time_published"].apply(
        lambda ts: map_to_bar(ts, price_idx)
    )
    news = news.dropna(subset=["bar_dt"])

    # One-hot за labels (Somewhat се спојуваат со Bullish/Bearish)
    one_hot = pd.get_dummies(
        news["overall_sentiment_label"]
            .replace({"Somewhat-Bullish": "Bullish",
                      "Somewhat-Bearish": "Bearish"}),
        prefix="lab"
    )
    news = pd.concat([news, one_hot], axis=1)

    agg = (news.groupby("bar_dt")
                 .agg(sent_mean   = ("overall_sentiment_score", "mean"),
                      sent_max_abs= ("overall_sentiment_score",
                                      lambda g: g.abs().max()),
                      bullish_cnt = ("lab_Bullish", "sum"),
                      bearish_cnt = ("lab_Bearish", "sum"),
                      headline_count = ("overall_sentiment_score", "size")))
    return agg


# ─── MAIN ────────────────────────────────────────────────────────────
def main() -> None:
    prices = load_prices(PRICE_CSV)
    news   = load_news(NEWS_CSV, SYMBOL)

    news_bar = aggregate_news(news, prices.index)

    merged = prices.join(news_bar, how="left")
    merged[["headline_count", "bullish_cnt", "bearish_cnt"]] = (
        merged[["headline_count", "bullish_cnt", "bearish_cnt"]].fillna(0)
    )

    merged.to_csv(OUT_CSV)
    print(f"✅  Saved → {OUT_CSV}   ({len(merged)} rows)")

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
