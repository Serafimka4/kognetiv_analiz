"""Тематическое моделирование новостей (LDA, gensim).

На леммах из news_clean обучаем LDA, выделяем N тем (драйверов рынка),
для каждой новости определяем доминирующую тему -> news_topic.
Модель и словарь сохраняем в data/models/.
"""
from gensim import corpora
from gensim.models import LdaModel

import config
from src.storage import db

NUM_TOPICS = 8
NO_BELOW = 5       # лемма должна встретиться минимум в 5 документах
NO_ABOVE = 0.5     # и не более чем в 50% документов
PASSES = 10
MODELS_DIR = config.DATA_DIR / "models"


def run(num_topics=NUM_TOPICS):
    db.init_db()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows = db.load_clean_all()
    ids = [r[0] for r in rows]
    dates = [r[1] for r in rows]
    docs = [(r[3] or "").split() for r in rows]
    print(f"  документов: {len(docs)}", flush=True)

    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE)
    corpus = [dictionary.doc2bow(d) for d in docs]

    print(f"  обучение LDA ({num_topics} тем)...", flush=True)
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=PASSES,
        random_state=42,
        alpha="auto",
    )

    # доминирующая тема для каждой новости
    results = []
    for news_id, date, bow in zip(ids, dates, corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        topic_id, prob = max(dist, key=lambda x: x[1])
        results.append((news_id, date, int(topic_id), float(prob)))

    for s in range(0, len(results), 2000):
        db.save_news_topic(results[s:s + 2000])

    lda.save(str(MODELS_DIR / "lda.model"))
    dictionary.save(str(MODELS_DIR / "lda.dict"))

    print("\n  Выделенные темы (топ-слова):", flush=True)
    for tid in range(num_topics):
        words = ", ".join(w for w, _ in lda.show_topic(tid, topn=8))
        print(f"    Тема {tid}: {words}", flush=True)

    print(f"\nГотово: темы назначены {len(results)} новостям.", flush=True)
    return lda, dictionary


if __name__ == "__main__":
    print("Тематическое моделирование (LDA)...")
    run()
