"""Сбор финансовых новостей из RSS-лент русскоязычных СМИ."""
from datetime import datetime, timezone

import feedparser
import requests

import config
from src.storage import db

# Часть СМИ отдаёт пустой ответ без браузерного User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _published(entry):
    """Извлекает дату публикации в ISO-формате, если она есть."""
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    if t:
        return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
    return entry.get("published") or entry.get("updated") or ""


def fetch_feed(source, url):
    """Загружает одну RSS-ленту. Возвращает список кортежей для БД."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    # Качаем через requests (использует certifi -> нет SSL-ошибок на macOS),
    # затем разбираем содержимое feedparser-ом.
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    rows = []
    for e in parsed.entries:
        rows.append(
            (
                source,
                e.get("title", "").strip(),
                e.get("summary", "").strip(),
                e.get("link", "").strip(),
                _published(e),
                fetched_at,
            )
        )
    return rows


def collect_all(feeds=None):
    """Загружает и сохраняет новости из всех лент конфига."""
    feeds = feeds or config.RSS_FEEDS
    db.init_db()
    total = 0
    for source, url in feeds.items():
        try:
            rows = fetch_feed(source, url)
            saved = db.save_news(rows)
            print(f"  {source}: получено {len(rows)} новостей, новых сохранено {saved}")
            total += saved
        except Exception as exc:  # лента может быть временно недоступна
            print(f"  {source}: ОШИБКА — {exc}")
    return total


if __name__ == "__main__":
    print("Сбор новостей RSS...")
    collect_all()
