"""Загрузка исторического архива новостей Lenta.ru в БД.

Источник: https://github.com/yutkin/Lenta.Ru-News-Dataset (CSV: url,title,text,topic,tags,date).
Фильтруем по рубрике (config.LENTA_TOPICS) и окну дат [START_DATE, END_DATE].
"""
import pandas as pd

import config
from src.storage import db

CHUNK = 50_000


def load_archive(path=None, topics=None, start=None, end=None):
    """Читает csv.bz2 чанками, фильтрует и сохраняет релевантные новости.

    Возвращает число сохранённых записей.
    """
    path = path or config.LENTA_FILE
    topics = set(topics or config.LENTA_TOPICS)
    start = start or config.START_DATE
    end = end or config.END_DATE

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден архив Lenta: {path}\n"
            f"Скачайте его: curl -sL -o {path} {config.LENTA_URL}"
        )

    db.init_db()
    total_saved = 0
    total_seen = 0

    reader = pd.read_csv(
        path,
        compression="bz2",
        usecols=["url", "title", "text", "topic", "date"],
        chunksize=CHUNK,
    )
    for chunk in reader:
        total_seen += len(chunk)
        # нормализуем дату -> YYYY-MM-DD
        chunk["date"] = pd.to_datetime(
            chunk["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        mask = (
            chunk["topic"].isin(topics)
            & chunk["date"].notna()
            & (chunk["date"] >= start)
            & (chunk["date"] <= end)
        )
        sel = chunk.loc[mask, ["topic", "date", "title", "text", "url"]]
        if sel.empty:
            continue

        rows = [
            ("lenta", r.date, r.topic, r.title, r.text, r.url)
            for r in sel.itertuples(index=False)
        ]
        total_saved += db.save_news_archive(rows)

    print(f"  Lenta: просмотрено {total_seen} статей, "
          f"сохранено релевантных {total_saved} "
          f"(рубрики {sorted(topics)}, окно {start}..{end})")
    return total_saved


if __name__ == "__main__":
    print("Загрузка архива Lenta.ru...")
    load_archive()
