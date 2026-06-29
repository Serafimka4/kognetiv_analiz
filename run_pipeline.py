"""Полный конвейер проекта: от сбора данных до модели и объяснимости.

Запускает этапы 1–6 по порядку. Дашборд (этап 7) запускается отдельно:
    streamlit run src/dashboard/app.py

Запуск:  python run_pipeline.py
Предварительно положите архив Lenta в data/raw/ (см. README).
"""
import time

from src.collectors import lenta, moex
from src.features import build_dataset
from src.models import train
from src.nlp import sentiment, topics
from src.preprocessing import pipeline as prep
from src.cognitive import explain
from src.storage import db


def step(title, fn):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    t = time.time()
    fn()
    print(f"[{title}] заняло {time.time() - t:.1f} c")


def main():
    db.init_db()
    step("Этап 1a. Котировки MOEX (индекс + акции)", moex.collect_all)
    step("Этап 1b. Архив новостей Lenta", lenta.load_archive)
    step("Этап 2. Предобработка текста", prep.run)
    step("Этап 3a. Тональность (RuBERT)", sentiment.run)
    step("Этап 3b. Темы (LDA)", topics.run)
    step("Этап 4. Датасет признаков", build_dataset.build)
    step("Этап 5. Модели (цена vs цена+текст)", train.run)
    step("Этап 6. Объяснимость (SHAP)", explain.run)
    print("\nГотово. Запустите дашборд: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
