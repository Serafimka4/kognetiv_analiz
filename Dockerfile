# ПО для когнитивного анализа рыночных тенденций (IMOEX)
# Базовый образ: Python 3.12 slim
FROM python:3.12-slim

# Системные зависимости:
#   libgomp1 — OpenMP-рантайм для xgboost
#   curl, bzip2 — для скачивания/распаковки архива новостей Lenta
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — кэшируется отдельным слоем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код проекта
COPY . .

# Streamlit: запуск headless на 0.0.0.0:8501
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf-cache

EXPOSE 8501

# По умолчанию — дашборд. Конвейер: docker run ... python run_pipeline.py
CMD ["streamlit", "run", "src/dashboard/app.py"]
