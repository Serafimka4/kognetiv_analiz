"""Работа с SQLite: схема и сохранение котировок и новостей."""
import sqlite3
from contextlib import contextmanager

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    open    REAL,
    high    REAL,
    low     REAL,
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS news (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT,
    title      TEXT,
    summary    TEXT,
    link       TEXT UNIQUE,
    published  TEXT,
    fetched_at TEXT
);

-- Дневная история индекса IMOEX (целевой ряд)
CREATE TABLE IF NOT EXISTS index_quotes (
    security TEXT NOT NULL,
    date     TEXT NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    PRIMARY KEY (security, date)
);

-- Исторический архив новостей (Lenta.ru), отфильтрованный по рубрике
CREATE TABLE IF NOT EXISTS news_archive (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    date   TEXT,
    topic  TEXT,
    title  TEXT,
    text   TEXT,
    url    TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_news_archive_date ON news_archive(date);

-- Предобработанный текст новостей (результат Этапа 2)
CREATE TABLE IF NOT EXISTS news_clean (
    news_id    INTEGER PRIMARY KEY,
    date       TEXT,
    clean_text TEXT,   -- очищенный текст для RuBERT
    lemmas     TEXT,   -- леммы через пробел для LDA
    FOREIGN KEY (news_id) REFERENCES news_archive(id)
);
CREATE INDEX IF NOT EXISTS idx_news_clean_date ON news_clean(date);

-- Тональность каждой новости (Этап 3, RuBERT)
CREATE TABLE IF NOT EXISTS news_sentiment (
    news_id INTEGER PRIMARY KEY,
    date    TEXT,
    label   TEXT,            -- NEUTRAL / POSITIVE / NEGATIVE
    neu     REAL,
    pos     REAL,
    neg     REAL,
    score   REAL,            -- pos - neg, диапазон [-1, 1]
    FOREIGN KEY (news_id) REFERENCES news_archive(id)
);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_date ON news_sentiment(date);

-- Доминирующая тема каждой новости (Этап 3, LDA)
CREATE TABLE IF NOT EXISTS news_topic (
    news_id    INTEGER PRIMARY KEY,
    date       TEXT,
    topic_id   INTEGER,
    topic_prob REAL,
    FOREIGN KEY (news_id) REFERENCES news_archive(id)
);
CREATE INDEX IF NOT EXISTS idx_news_topic_date ON news_topic(date);

-- LIVE-мониторинг: свежие котировки IMOEX
CREATE TABLE IF NOT EXISTS live_index (
    date  TEXT PRIMARY KEY,
    close REAL
);

-- LIVE-мониторинг: свежие новости с тональностью и темой
CREATE TABLE IF NOT EXISTS live_news (
    link       TEXT PRIMARY KEY,
    source     TEXT,
    date       TEXT,
    title      TEXT,
    label      TEXT,
    neu        REAL,
    pos        REAL,
    neg        REAL,
    score      REAL,
    topic_id   INTEGER,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_live_news_date ON live_news(date);
"""


@contextmanager
def connect():
    """Контекстный менеджер соединения с БД (создаёт каталог data/ при нужде)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Создаёт таблицы, если их ещё нет."""
    with connect() as conn:
        conn.executescript(SCHEMA)


def save_quotes(rows):
    """Сохраняет котировки. rows — список кортежей
    (ticker, date, open, high, low, close, volume). Дубликаты игнорируются."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO quotes "
            "(ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def save_news(rows):
    """Сохраняет новости. rows — список кортежей
    (source, title, summary, link, published, fetched_at).
    Дубликаты по link игнорируются."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO news "
            "(source, title, summary, link, published, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def save_index(rows):
    """Сохраняет историю индекса. rows — список кортежей
    (security, date, open, high, low, close, volume)."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO index_quotes "
            "(security, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def save_news_archive(rows):
    """Сохраняет архивные новости. rows — список кортежей
    (source, date, topic, title, text, url). Дубликаты по url игнорируются."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO news_archive "
            "(source, date, topic, title, text, url) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def load_archive_all():
    """Читает весь архив новостей в память. Возвращает список кортежей
    (id, date, title, text). Соединение закрывается до записи результатов —
    иначе открытый курсор блокирует запись в ту же БД (SQLite)."""
    with connect() as conn:
        return conn.execute(
            "SELECT id, date, title, text FROM news_archive ORDER BY id"
        ).fetchall()


def save_news_clean(rows):
    """Сохраняет предобработанные новости. rows — кортежи
    (news_id, date, clean_text, lemmas)."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO news_clean "
            "(news_id, date, clean_text, lemmas) VALUES (?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def load_clean_all():
    """Читает предобработанные новости в память. Возвращает список кортежей
    (news_id, date, clean_text, lemmas)."""
    with connect() as conn:
        return conn.execute(
            "SELECT news_id, date, clean_text, lemmas FROM news_clean ORDER BY news_id"
        ).fetchall()


def save_news_sentiment(rows):
    """rows — кортежи (news_id, date, label, neu, pos, neg, score)."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO news_sentiment "
            "(news_id, date, label, neu, pos, neg, score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def save_news_topic(rows):
    """rows — кортежи (news_id, date, topic_id, topic_prob)."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO news_topic "
            "(news_id, date, topic_id, topic_prob) VALUES (?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def save_live_index(rows):
    """rows — кортежи (date, close)."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO live_index (date, close) VALUES (?, ?)", rows
        )
        return conn.total_changes


def save_live_news(rows):
    """rows — кортежи (link, source, date, title, label, neu, pos, neg, score,
    topic_id, fetched_at). Дубликаты по link обновляются."""
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO live_news "
            "(link, source, date, title, label, neu, pos, neg, score, "
            "topic_id, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes


def counts():
    """Возвращает счётчики по основным таблицам — для проверки."""
    with connect() as conn:
        return {
            "quotes": conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0],
            "index_quotes": conn.execute(
                "SELECT COUNT(*) FROM index_quotes"
            ).fetchone()[0],
            "news": conn.execute("SELECT COUNT(*) FROM news").fetchone()[0],
            "news_archive": conn.execute(
                "SELECT COUNT(*) FROM news_archive"
            ).fetchone()[0],
            "news_clean": conn.execute(
                "SELECT COUNT(*) FROM news_clean"
            ).fetchone()[0],
            "news_sentiment": conn.execute(
                "SELECT COUNT(*) FROM news_sentiment"
            ).fetchone()[0],
            "news_topic": conn.execute(
                "SELECT COUNT(*) FROM news_topic"
            ).fetchone()[0],
            "live_index": conn.execute(
                "SELECT COUNT(*) FROM live_index"
            ).fetchone()[0],
            "live_news": conn.execute(
                "SELECT COUNT(*) FROM live_news"
            ).fetchone()[0],
        }
