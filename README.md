# ПО для когнитивного анализа рыночных тенденций (IMOEX)

Дипломный проект. Система собирает котировки Московской биржи и русскоязычные
новости, извлекает из текстов «когнитивные» признаки (тональность, темы) методами
ML/NLP и анализирует/прогнозирует тенденции индекса **IMOEX** с объяснением факторов.

**Гипотеза:** учёт текстовой аналитики (тональность новостей) повышает точность
прогноза направления тренда по сравнению с моделью только на ценовых данных.

## Стек

Python · apimoex · pandas · razdel · pymorphy3 · transformers (RuBERT) · gensim (LDA)
· scikit-learn · xgboost · shap · Streamlit · plotly

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS: для xgboost нужен OpenMP
brew install libomp
```

Скачайте архив новостей Lenta.ru и положите в `data/raw/`:
```bash
mkdir -p data/raw
curl -L -o data/raw/lenta-ru-news.csv.bz2 \
  https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2
```

## Запуск

Полный конвейер (этапы 1–6):
```bash
python run_pipeline.py
```
Дашборд (этап 7):
```bash
streamlit run src/dashboard/app.py
```

Live-мониторинг (свежие новости RSS + текущие котировки IMOEX → вкладка «Сейчас»):
```bash
python -m src.live.monitor
```
Запускайте периодически — данные накапливаются в `live_news`/`live_index`.

Этапы можно запускать и по отдельности:
```bash
python collect.py                      # 1. сбор данных
python preprocess.py                   # 2. предобработка текста
python -m src.nlp.sentiment            # 3a. тональность (RuBERT)
python -m src.nlp.topics               # 3b. темы (LDA)
python -m src.features.build_dataset   # 4. признаки
python -m src.models.train             # 5. модели
python -m src.cognitive.explain        # 6. SHAP
```

## Docker

```bash
docker compose build                  # собрать образ
docker compose run --rm pipeline      # прогнать конвейер (этапы 1–6)
docker compose up dashboard           # дашборд на http://localhost:8501
docker compose run --rm live          # освежить live-данные (RSS + котировки)
```

Каталог `./data` монтируется как том — БД, архив Lenta, LDA-модель и кэш модели
RuBERT сохраняются на хосте между запусками. Перед первым прогоном конвейера
положите архив Lenta в `data/raw/` (см. выше). Модель тональности RuBERT (~700 МБ)
скачивается при первом запуске в `data/hf-cache/`.

## Структура

```
kog_anal/
├── config.py                 # тикеры, окно дат, источники
├── run_pipeline.py           # полный конвейер 1–6
├── collect.py, preprocess.py # запуск этапов 1 и 2
├── data/                     # SQLite БД, архив, результаты (gitignore)
└── src/
    ├── collectors/  moex.py, news.py, lenta.py     # этап 1
    ├── preprocessing/ clean.py, stopwords.py, pipeline.py  # этап 2
    ├── nlp/         sentiment.py, topics.py         # этап 3
    ├── features/    build_dataset.py                # этап 4
    ├── models/      train.py                        # этап 5
    ├── cognitive/   explain.py                      # этап 6
    ├── dashboard/   app.py                          # этап 7
    └── storage/     db.py                           # SQLite
```

## Основные результаты

- Данные: индекс IMOEX (1008 дней, 2015–2018), 14 767 новостей рубрики «Экономика».
- Модели (walk-forward): price=0.480, text=0.506, **price+text=0.507** (baseline 0.489).
- Гипотеза подтверждается слабо (Δacc≈+2.7 п.п.), абсолютная точность ≈ случайной
  (ROC-AUC≈0.51) — согласуется с гипотезой эффективного рынка.
- SHAP: текстовые факторы дают ~48% влияния; информативнее отраслевые настроения
  (банки, валюта, санкции, нефть), а не общий тон новостей.

Подробный план и статус этапов — в [PLAN.md](PLAN.md).
