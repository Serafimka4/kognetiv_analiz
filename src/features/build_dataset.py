"""Этап 4: формирование единого датасета признаков для прогноза тренда IMOEX.

Признаки двух групп (по префиксу имени колонки):
  px_*  — ценовые / технические (только котировки индекса);
  tx_*  — текстовые (тональность, объём и темы новостей).
Цель: y = направление доходности IMOEX на СЛЕДУЮЩИЙ торговый день (1 — рост, 0 — падение).

Результат сохраняется в data/dataset.csv.
"""
import sqlite3

import numpy as np
import pandas as pd
import ta

import config

OUT_CSV = config.DATA_DIR / "dataset.csv"
MARKET_TOPICS = [1, 2, 4, 5, 6, 7]  # рыночно-значимые темы LDA


def _price_features(conn):
    px = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM index_quotes ORDER BY date",
        conn, parse_dates=["date"],
    )
    px["ret"] = px["close"].pct_change()
    # лаги доходности
    for k in (1, 2, 3, 5):
        px[f"px_ret_lag{k}"] = px["ret"].shift(k)
    # скользящая волатильность
    px["px_vol5"] = px["ret"].rolling(5).std()
    px["px_vol10"] = px["ret"].rolling(10).std()
    # технические индикаторы
    px["px_rsi14"] = ta.momentum.RSIIndicator(px["close"], window=14).rsi()
    px["px_macd"] = ta.trend.MACD(px["close"]).macd_diff()
    px["px_sma_ratio"] = px["close"] / px["close"].rolling(10).mean() - 1
    px["px_mom5"] = px["close"].pct_change(5)
    px["px_vol_chg"] = np.log1p(px["volume"]).diff()
    # ЦЕЛЬ: знак доходности следующего дня
    px["y"] = (px["ret"].shift(-1) > 0).astype(int)
    return px


def _text_features(conn, trading_days):
    """Агрегирует новости по торговым дням (новости выходных -> следующий торг. день)."""
    s = pd.read_sql(
        """SELECT s.date, s.score, s.label, t.topic_id
           FROM news_sentiment s LEFT JOIN news_topic t ON t.news_id = s.news_id""",
        conn, parse_dates=["date"],
    )
    s = s.sort_values("date")
    td = pd.DataFrame({"trade_date": trading_days}).sort_values("trade_date")
    # каждую новость относим к ближайшему торговому дню >= даты новости
    s = pd.merge_asof(
        s, td, left_on="date", right_on="trade_date", direction="forward"
    ).dropna(subset=["trade_date"])

    g = s.groupby("trade_date")
    feat = pd.DataFrame({
        "tx_sent_mean": g["score"].mean(),
        "tx_sent_std": g["score"].std(),
        "tx_neg_share": g["label"].apply(lambda x: (x == "NEGATIVE").mean()),
        "tx_pos_share": g["label"].apply(lambda x: (x == "POSITIVE").mean()),
        "tx_n_news": g.size(),
    })
    feat["tx_n_log"] = np.log1p(feat["tx_n_news"])
    # тональность по рыночным темам
    mkt = s[s["topic_id"].isin(MARKET_TOPICS)].groupby("trade_date")["score"].mean()
    feat["tx_sent_market"] = mkt
    # тональность по ключевым темам отдельно (нефть=4, валюта=2, санкции=1, банки=5)
    for tid, name in [(4, "oil"), (2, "fx"), (1, "sanc"), (5, "bank")]:
        feat[f"tx_sent_t{name}"] = (
            s[s["topic_id"] == tid].groupby("trade_date")["score"].mean()
        )
    feat = feat.reset_index().rename(columns={"trade_date": "date"})
    return feat


def build():
    conn = sqlite3.connect(config.DB_PATH)
    px = _price_features(conn)
    tx = _text_features(conn, px["date"])
    conn.close()

    df = px.merge(tx, on="date", how="left")

    # сглаживание тональности (накопленный фон)
    df["tx_sent_roll3"] = df["tx_sent_mean"].rolling(3).mean()
    df["tx_sent_roll5"] = df["tx_sent_mean"].rolling(5).mean()

    # дни без новостей: объём = 0, тональность = нейтральная (0)
    tx_cols = [c for c in df.columns if c.startswith("tx_")]
    df[tx_cols] = df[tx_cols].fillna(0)

    # убираем строки с NaN в ценовых признаках (начало ряда) и последнюю (нет y)
    df = df.dropna().reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    px_cols = [c for c in df.columns if c.startswith("px_")]
    print(f"Датасет: {len(df)} строк, {len(px_cols)} ценовых + {len(tx_cols)} текстовых признаков")
    print(f"Баланс классов y: {df['y'].mean():.3f} (доля роста)")
    print(f"Сохранено: {OUT_CSV}")
    return df


if __name__ == "__main__":
    print("Формирование датасета признаков...")
    build()
