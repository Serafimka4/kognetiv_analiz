"""Конфигурация проекта: тикеры, периоды, источники данных, пути."""
from pathlib import Path

# --- Пути ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "market.db"

# --- Целевой объект прогноза: индекс Московской биржи ---
# Архив новостей Lenta покрывает 1999-2019, поэтому окно котировок совмещаем
# с этим периодом (берём 2015-2019 — стабильная и репрезентативная выборка).
# Рубрика «Экономика» в архиве Lenta заканчивается 2018 годом — совмещаем окно с ним.
START_DATE = "2015-01-01"
END_DATE = "2018-12-31"

# Индекс MOEX (основной целевой ряд)
INDEX_SECURITY = "IMOEX"
INDEX_ENGINE = "stock"
INDEX_MARKET = "index"
INDEX_BOARD = "SNDX"

# Отдельные акции — вторичная демонстрация (опционально)
TICKERS = ["SBER", "GAZP", "LKOH", "GMKN"]
MOEX_BOARD = "TQBR"  # основной режим торгов акциями T+

# --- Исторический архив новостей: Lenta.ru ---
LENTA_FILE = RAW_DIR / "lenta-ru-news.csv.bz2"
LENTA_URL = (
    "https://github.com/yutkin/Lenta.Ru-News-Dataset/"
    "releases/download/v1.1/lenta-ru-news.csv.bz2"
)
# Рубрики, которые считаем релевантными рынку
LENTA_TOPICS = ["Экономика"]

# --- Живые RSS-ленты (для демо/дообучения на свежих данных, опционально) ---
RSS_FEEDS = {
    "rbc": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "interfax": "https://www.interfax.ru/rss.asp",
    "prime": "https://1prime.ru/export/rss2/index.xml",
    "finam": "https://www.finam.ru/analysis/conews/rsspoint/",
}

# --- Новостные RSS-ленты (рус. финансовые СМИ) ---
# Если какая-то лента недоступна — закомментируйте её.
RSS_FEEDS = {
    "rbc": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "interfax": "https://www.interfax.ru/rss.asp",
    "prime": "https://1prime.ru/export/rss2/index.xml",
    "finam": "https://www.finam.ru/analysis/conews/rsspoint/",
}
