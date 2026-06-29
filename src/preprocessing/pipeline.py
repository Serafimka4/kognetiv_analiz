"""Конвейер Этапа 2: предобработка архива новостей.

Берёт записи из news_archive, для каждой формирует:
  - clean_text (заголовок + текст, очищенные) — для модели тональности;
  - lemmas (леммы без стоп-слов) — для тематического моделирования.
Результат пишет в таблицу news_clean.
"""
from src.preprocessing.clean import clean_text, lemmatize
from src.storage import db

BATCH = 2000


def run():
    db.init_db()

    # 1) читаем весь архив в память и закрываем соединение
    archive = db.load_archive_all()
    total = len(archive)
    print(f"  к обработке: {total} новостей", flush=True)

    # 2) обрабатываем (CPU-bound, без открытых соединений к БД)
    results = []
    for i, (news_id, date, title, text) in enumerate(archive, 1):
        full = f"{title or ''}. {text or ''}".strip()
        clean = clean_text(full)
        lemmas = " ".join(lemmatize(clean))
        results.append((news_id, date, clean, lemmas))
        if i % 5000 == 0:
            print(f"  обработано {i}/{total}...", flush=True)

    # 3) пишем результат батчами
    for start in range(0, len(results), BATCH):
        db.save_news_clean(results[start:start + BATCH])

    print(f"Готово: предобработано {len(results)} новостей.", flush=True)
    return len(results)


if __name__ == "__main__":
    print("Предобработка новостей (очистка + лемматизация)...")
    run()
