"""Сбор котировок акций с Московской биржи через ISS MOEX API (apimoex)."""
import apimoex
import requests

import config
from src.storage import db


def fetch_quotes(ticker, start=None, end=None, board=None):
    """Загружает дневную историю по одной акции за окно [start, end].

    Возвращает список кортежей (ticker, date, open, high, low, close, volume).
    """
    start = start or config.START_DATE
    end = end or config.END_DATE
    board = board or config.MOEX_BOARD
    columns = ("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")

    with requests.Session() as session:
        data = apimoex.get_board_history(
            session, security=ticker, start=start, end=end,
            board=board, columns=columns,
        )

    return [
        (ticker, r["TRADEDATE"], r["OPEN"], r["HIGH"],
         r["LOW"], r["CLOSE"], r["VOLUME"])
        for r in data
    ]


def fetch_index(start=None, end=None):
    """Загружает дневную историю индекса IMOEX за окно [start, end].

    Возвращает список кортежей (security, date, open, high, low, close, volume).
    У индекса вместо объёма используется оборот (VALUE).
    """
    start = start or config.START_DATE
    end = end or config.END_DATE
    columns = ("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VALUE")

    with requests.Session() as session:
        data = apimoex.get_board_history(
            session,
            security=config.INDEX_SECURITY,
            start=start, end=end,
            board=config.INDEX_BOARD,
            market=config.INDEX_MARKET,
            engine=config.INDEX_ENGINE,
            columns=columns,
        )

    return [
        (config.INDEX_SECURITY, r["TRADEDATE"], r["OPEN"], r["HIGH"],
         r["LOW"], r["CLOSE"], r.get("VALUE"))
        for r in data
    ]


def collect_index():
    """Загружает и сохраняет историю индекса IMOEX."""
    db.init_db()
    rows = fetch_index()
    saved = db.save_index(rows)
    print(f"  {config.INDEX_SECURITY}: получено {len(rows)} строк, "
          f"новых сохранено {saved}")
    return saved


def collect_all(tickers=None):
    """Загружает индекс IMOEX и котировки всех акций из конфига."""
    tickers = tickers or config.TICKERS
    db.init_db()
    print("Индекс:")
    collect_index()
    print("Акции:")
    total = 0
    for ticker in tickers:
        rows = fetch_quotes(ticker)
        saved = db.save_quotes(rows)
        print(f"  {ticker}: получено {len(rows)} строк, новых сохранено {saved}")
        total += saved
    return total


if __name__ == "__main__":
    print("Сбор котировок MOEX...")
    collect_all()
