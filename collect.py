"""Точка входа Этапа 1: сбор данных в SQLite.

Собирает:
  - индекс IMOEX и котировки акций MOEX за окно [START_DATE, END_DATE];
  - исторический архив новостей Lenta.ru (рубрика «Экономика»).

Запуск:  python collect.py
"""
import config
from src.collectors import lenta, moex
from src.storage import db


def main():
    db.init_db()

    print("=" * 60)
    print(f"Котировки MOEX  (окно {config.START_DATE}..{config.END_DATE})")
    print("=" * 60)
    moex.collect_all()

    print("\n" + "=" * 60)
    print("Архив новостей Lenta.ru")
    print("=" * 60)
    lenta.load_archive()

    print("\n" + "=" * 60)
    c = db.counts()
    print("Итого в базе:")
    print(f"  индекс IMOEX : {c['index_quotes']} дней")
    print(f"  акции        : {c['quotes']} строк")
    print(f"  архив новостей: {c['news_archive']} статей")
    print(f"База данных: {config.DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
