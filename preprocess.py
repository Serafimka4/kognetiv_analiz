"""Точка входа Этапа 2: предобработка текста новостей.

Запуск:  python preprocess.py
"""
from src.preprocessing import pipeline
from src.storage import db


def main():
    print("=" * 60)
    print("Предобработка русского текста новостей")
    print("=" * 60)
    pipeline.run()

    c = db.counts()
    print("\nВ базе:")
    print(f"  архив новостей       : {c['news_archive']}")
    print(f"  предобработано (clean): {c['news_clean']}")


if __name__ == "__main__":
    main()
