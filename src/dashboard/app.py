"""Этап 7: дашборд когнитивного анализа рыночных тенденций (Streamlit).

Запуск:  streamlit run src/dashboard/app.py
"""
import faulthandler
import sqlite3
import sys
from pathlib import Path

# при нативном сбое (SIGSEGV) выведет точную строку Python в stderr
faulthandler.enable()

# чтобы работали импорты config/src при запуске через streamlit
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402

TOPIC_LABELS = {
    0: "Проекты/инфраструктура", 1: "США / санкции / газ",
    2: "Доллар / валюта / экономика", 3: "Товары / продукция",
    4: "Нефть / ОПЕК", 5: "Банки / Сбербанк / кредиты",
    6: "Бюджет / пенсии / правительство", 7: "Украина / долги",
}


def _read_sql(conn, query):
    """Безопасное чтение: если таблицы ещё нет, возвращает пустой DataFrame."""
    try:
        return pd.read_sql(query, conn, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_data():
    if not config.DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(config.DB_PATH)
    idx = _read_sql(conn, "SELECT date, close FROM index_quotes ORDER BY date")
    sent = _read_sql(conn,
        """SELECT date, AVG(score) sent, COUNT(*) n_news,
           AVG(CASE WHEN label='NEGATIVE' THEN 1.0 ELSE 0 END) neg_share
           FROM news_sentiment GROUP BY date""")
    topics = _read_sql(conn, "SELECT date, topic_id FROM news_topic")
    conn.close()
    if not sent.empty:
        sent["sent_roll7"] = sent["sent"].rolling(7).mean()
    return idx, sent, topics


@st.cache_data
def load_results():
    res_path = config.DATA_DIR / "model_results.csv"
    imp_path = config.DATA_DIR / "shap_importance.csv"
    res = pd.read_csv(res_path) if res_path.exists() else None
    imp = pd.read_csv(imp_path) if imp_path.exists() else None
    return res, imp


@st.cache_data
def load_live():
    if not config.DB_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(config.DB_PATH)
    lidx = _read_sql(conn, "SELECT date, close FROM live_index ORDER BY date")
    lnews = _read_sql(conn,
        "SELECT date, source, title, label, score, topic_id FROM live_news")
    conn.close()
    return lidx, lnews


def render_live(lidx, lnews):
    """Отрисовка раздела актуальных данных (используется и как вкладка,
    и как облегчённый режим при отсутствии исторических данных)."""
    st.subheader("Текущее настроение рынка (свежие новости + котировки)")
    if lnews.empty:
        st.warning("Live-данные ещё не собраны. Запустите: "
                   "`python -m src.live.monitor`")
        return
    last_day = lnews["date"].max()
    today = lnews[lnews["date"] == last_day]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Обновлено", last_day.strftime("%Y-%m-%d"))
    c2.metric("Новостей всего", len(lnews))
    mood = today["score"].mean() if len(today) else 0
    c3.metric("Настроение (посл. день)", f"{mood:+.3f}",
              delta="негатив" if mood < -0.05 else
              ("позитив" if mood > 0.05 else "нейтрально"))
    if not lidx.empty:
        c4.metric("IMOEX", f"{lidx['close'].iloc[-1]:.0f}")

    daily = lnews.groupby("date")["score"].mean().reset_index()
    fig = go.Figure()
    if not lidx.empty:
        fig.add_trace(go.Scatter(x=lidx.date, y=lidx.close, name="IMOEX",
                                 line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=daily.date, y=daily.score,
                             name="Настроение (день)", yaxis="y2",
                             line=dict(color="#d62728")))
    fig.update_layout(
        height=420, title="IMOEX и настроение свежих новостей",
        yaxis=dict(title="IMOEX"),
        yaxis2=dict(title="Настроение", overlaying="y", side="right"),
        legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    colA, colB = st.columns(2)
    with colA:
        by_src = (lnews.groupby("source")["score"].mean()
                  .sort_values().reset_index())
        fig2 = px.bar(by_src, x="score", y="source", orientation="h",
                      title="Среднее настроение по источникам",
                      color="score", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig2, width="stretch")
    with colB:
        lnews2 = lnews.copy()
        lnews2["Тема"] = lnews2["topic_id"].map(TOPIC_LABELS).fillna("—")
        by_t = (lnews2.groupby("Тема")["score"].mean()
                .sort_values().reset_index())
        fig3 = px.bar(by_t, x="score", y="Тема", orientation="h",
                      title="Среднее настроение по темам",
                      color="score", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig3, width="stretch")

    st.subheader("Свежие заголовки")
    show = (today if len(today) else lnews).sort_values("date", ascending=False)
    show = show[["date", "source", "label", "score", "title"]].head(30)
    st.dataframe(show, width="stretch", hide_index=True)


def main():
    st.set_page_config(page_title="Когнитивный анализ рынка (IMOEX)",
                       layout="wide")
    st.title("📈 Когнитивный анализ рыночных тенденций (IMOEX)")
    st.caption("Дипломный проект · ML/NLP · новости Lenta.ru (2015–2018) + индекс MOEX")

    idx, sent, topics = load_data()
    res, imp = load_results()
    lidx0, lnews0 = load_live()

    # для исторических вкладок нужны и котировки, и результаты NLP
    hist_ready = not idx.empty and not sent.empty and not topics.empty

    # данных ещё нет — показываем инструкцию вместо падения
    if not hist_ready and lnews0.empty:
        st.warning("Данные ещё не сформированы или неполны — каталог `data/` пуст "
                   "либо не пройдены все этапы конвейера.")
        st.markdown("""
Дашборд отображает результаты работы конвейера, которых пока нет в этой среде
(каталог `data/` не входит в репозиторий). Чтобы наполнить систему данными:

**Вариант 1 — прогнать конвейер на сервере** (нужен архив новостей в `data/raw/`):
```bash
mkdir -p data/raw
curl -L -o data/raw/lenta-ru-news.csv.bz2 \\
  https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2
python run_pipeline.py            # или: docker compose run --rm pipeline
```

**Вариант 2 — быстро показать актуальные данные** (свежие новости + котировки):
```bash
python -m src.live.monitor        # или: docker compose run --rm live
```
После этого обновите страницу. Вкладка «🔴 Сейчас» заработает уже после варианта 2.

**Вариант 3 — перенести готовые данные** с рабочей машины: скопируйте локальный
каталог `data/` (файл `market.db`, `model_results.csv`, `shap_importance.csv`,
папки `figures/`, `models/`) в `data/` на сервере.
        """)
        st.stop()

    # исторические данные неполны (нет тональности/тем) — облегчённый режим
    if not hist_ready:
        st.info("Полные исторические данные недоступны (нет результатов анализа "
                "тональности или тем). Для исторических вкладок прогоните весь "
                "конвейер: `python run_pipeline.py`. Ниже — доступный режим «Сейчас».")
        render_live(lidx0, lnews0)
        return

    # фильтр по датам
    dmin, dmax = idx["date"].min().date(), idx["date"].max().date()
    d1, d2 = st.sidebar.slider("Период", dmin, dmax, (dmin, dmax))
    mask_i = (idx.date.dt.date >= d1) & (idx.date.dt.date <= d2)
    mask_s = (sent.date.dt.date >= d1) & (sent.date.dt.date <= d2)
    mask_t = (topics.date.dt.date >= d1) & (topics.date.dt.date <= d2)
    idx, sent, topics = idx[mask_i], sent[mask_s], topics[mask_t]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Обзор", "Настроения", "Темы-драйверы", "Прогноз и факторы",
         "Выводы", "🔴 Сейчас (live)"])

    # --- Обзор: индекс + настроение ---
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Дней (индекс)", len(idx))
        c2.metric("Новостей", int(sent["n_news"].sum()))
        c3.metric("Средн. настроение", f"{sent['sent'].mean():.3f}")
        c4.metric("IMOEX закрытие", f"{idx['close'].iloc[-1]:.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx.date, y=idx.close, name="IMOEX",
                                 line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=sent.date, y=sent.sent_roll7, name="Настроение (7д)",
                                 yaxis="y2", line=dict(color="#d62728")))
        fig.update_layout(
            height=480, title="Индекс IMOEX и индекс настроений новостей",
            yaxis=dict(title="IMOEX"),
            yaxis2=dict(title="Настроение", overlaying="y", side="right"),
            legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.info("Индекс настроений — средний score (pos−neg) новостей по дням, "
                "сглаженный за 7 дней.")

    # --- Настроения ---
    with tab2:
        fig = px.line(sent, x="date", y="neg_share",
                      title="Доля негативных новостей по дням",
                      labels={"neg_share": "Доля негатива"})
        fig.update_traces(line_color="#d62728")
        st.plotly_chart(fig, width="stretch")

        fig2 = px.bar(sent, x="date", y="n_news",
                      title="Объём новостей по дням",
                      labels={"n_news": "Кол-во новостей"})
        st.plotly_chart(fig2, width="stretch")

    # --- Темы ---
    with tab3:
        st.subheader("Тематические драйверы (LDA)")
        topics = topics.copy()
        topics["label"] = topics["topic_id"].map(TOPIC_LABELS)
        share = (topics.groupby("label").size()
                 .sort_values(ascending=False).reset_index(name="Новостей"))
        st.dataframe(share, width="stretch", hide_index=True)

        topics["month"] = topics["date"].dt.to_period("M").dt.to_timestamp()
        ts = (topics.groupby(["month", "label"]).size()
              .reset_index(name="n"))
        fig = px.area(ts, x="month", y="n", color="label",
                      title="Динамика тем во времени (новостей в месяц)")
        st.plotly_chart(fig, width="stretch")

    # --- Прогноз и факторы ---
    with tab4:
        if res is None or imp is None:
            st.warning("Результаты модели ещё не сформированы. Запустите этапы 5–6: "
                       "`python -m src.models.train` и `python -m src.cognitive.explain` "
                       "(или `python run_pipeline.py`).")
        else:
            st.subheader("Сравнение моделей прогноза тренда (walk-forward)")
            st.dataframe(res.round(4), width="stretch", hide_index=True)
            st.caption("Гипотеза: «цена+текст» точнее, чем «цена». Подтверждается слабо "
                       "(Δacc≈+2.7 п.п.), абсолютная точность ≈ случайной (AUC≈0.51).")

            st.subheader("Влияние факторов на прогноз (SHAP)")
            top = imp.head(15).iloc[::-1]
            fig = px.bar(top, x="importance", y="name_ru", color="group",
                         orientation="h",
                         color_discrete_map={"текст": "#d62728", "цена": "#1f77b4"},
                         labels={"importance": "Среднее |SHAP|", "name_ru": "",
                                 "group": "Группа"})
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch")
            share = imp.groupby("group")["importance"].sum()
            share = (share / share.sum() * 100).round(1)
            st.metric("Вклад текстовых факторов",
                      f"{share.get('текст', 0)}%",
                      help="Доля суммарного |SHAP| у текстовых признаков")

    # --- Выводы ---
    with tab5:
        st.markdown("""
### Основные выводы

1. **Конвейер работает end-to-end:** сбор данных (индекс IMOEX + 14 767 новостей
   Lenta) → предобработка → тональность (RuBERT) → темы (LDA) → признаки →
   прогноз (XGBoost) → объяснимость (SHAP) → дашборд.

2. **Гипотеза подтверждается в слабой форме.** Модель «цена+текст» точнее модели
   «только цена» (Δaccuracy ≈ +2.7 п.п.), а «только текст» даёт лучший ROC-AUC.
   Но абсолютная точность близка к случайной (AUC ≈ 0.51).

3. **Когнитивная интерпретация (SHAP):** текстовые факторы дают ~48% совокупного
   влияния; среди них информативнее **отраслевые** настроения (банки, валюта,
   санкции, нефть), а не общий тон новостей.

4. **Ограничения:** новости Lenta «Экономика» — общий макрофон, не специфичный
   для рынка РФ; модель тональности смещена в негатив. Это согласуется с
   гипотезой эффективного рынка и объясняет слабый сигнал.

5. **Развитие:** рыночно-специфичные источники новостей (РБК Quote, Finam,
   Telegram), прогноз волатильности вместо направления, дообучение модели
   тональности на финансовых текстах.
        """)

    # --- Live: текущее состояние ---
    with tab6:
        render_live(lidx0, lnews0)


if __name__ == "__main__":
    main()
else:
    main()
