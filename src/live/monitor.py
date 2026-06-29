"""Live-мониторинг: свежие новости (RSS) + текущие котировки IMOEX.

Каждую свежую новость прогоняем через ту же модель тональности (RuBERT) и
присваиваем тему обученной ранее LDA-моделью. Результат — в таблицы
live_index и live_news. Дашборд показывает их на вкладке «Сейчас».

Запуск:  python -m src.live.monitor
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

import config
from src.collectors import moex, news
from src.nlp import sentiment
from src.preprocessing.clean import clean_text, lemmatize
from src.storage import db

LIVE_DAYS = 120  # глубина свежих котировок
MODELS_DIR = config.DATA_DIR / "models"


def refresh_quotes():
    """Тянет котировки IMOEX за последние LIVE_DAYS дней до сегодня."""
    start = (date.today() - timedelta(days=LIVE_DAYS)).isoformat()
    end = date.today().isoformat()
    rows = moex.fetch_index(start=start, end=end)
    # кортеж fetch_index: (security, date, open, high, low, close, value)
    saved = db.save_live_index([(r[1], r[5]) for r in rows])  # (date, close)
    print(f"  котировки IMOEX: {len(rows)} дней (новых/обновлено {saved})")
    return len(rows)


def _load_lda():
    """Загружает сохранённую LDA-модель и словарь (если есть)."""
    model_path = MODELS_DIR / "lda.model"
    dict_path = MODELS_DIR / "lda.dict"
    if model_path.exists() and dict_path.exists():
        return LdaModel.load(str(model_path)), corpora.Dictionary.load(str(dict_path))
    return None, None


def _to_date(published):
    """Парсит дату публикации в YYYY-MM-DD, иначе сегодня."""
    d = pd.to_datetime(published, errors="coerce", utc=True)
    if pd.isna(d):
        return date.today().isoformat()
    return d.date().isoformat()


def refresh_news():
    """Собирает свежие RSS-новости, считает тональность и тему, пишет в live_news."""
    lda, dictionary = _load_lda()

    # 1) собрать сырые новости из всех лент
    raw = []
    for source, url in config.RSS_FEEDS.items():
        try:
            for r in news.fetch_feed(source, url):
                # r = (source, title, summary, link, published, fetched_at)
                _src, title, summary, link, published, fetched = r
                if not link or not title:
                    continue
                raw.append({
                    "source": _src, "title": title, "summary": summary,
                    "link": link, "date": _to_date(published), "fetched": fetched,
                })
        except Exception as exc:
            print(f"  {source}: ОШИБКА — {exc}")
    if not raw:
        print("  свежих новостей не получено")
        return 0

    # 2) тональность (на заголовке + анонсе)
    texts = [clean_text(f"{n['title']}. {n['summary']}") for n in raw]
    scores = sentiment.score_texts(texts)

    # 3) тема по обученной LDA (если модель есть)
    rows = []
    for n, txt, s in zip(raw, texts, scores):
        topic_id = None
        if lda is not None:
            bow = dictionary.doc2bow(lemmatize(txt))
            if bow:
                topic_id = int(max(lda.get_document_topics(bow), key=lambda x: x[1])[0])
        rows.append((
            n["link"], n["source"], n["date"], n["title"],
            s["label"], s["neu"], s["pos"], s["neg"], s["score"],
            topic_id, n["fetched"],
        ))

    saved = db.save_live_news(rows)
    print(f"  свежих новостей обработано: {len(rows)} (новых/обновлено {saved})")
    return len(rows)


def run():
    db.init_db()
    print(f"Live-обновление {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print("Котировки:")
    refresh_quotes()
    print("Новости:")
    refresh_news()
    c = db.counts()
    print(f"Всего в live_news накоплено: {c.get('live_news', '—')}")


if __name__ == "__main__":
    run()
