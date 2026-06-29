"""Этап 6: когнитивный слой — объяснимость прогноза (SHAP).

Обучаем модель price+text на всех данных и считаем вклад каждого признака
(средний |SHAP|). Это показывает, КАКИЕ факторы (ценовые и текстовые, включая
тематические настроения) сильнее всего влияют на прогноз тренда — «когнитивная»
интерпретация модели.

Результаты:
  data/shap_importance.csv   — таблица важности признаков;
  data/figures/shap_importance.png — график топ-признаков.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

import config

DATA = config.DATA_DIR / "dataset.csv"
FIG_DIR = config.DATA_DIR / "figures"
IMP_CSV = config.DATA_DIR / "shap_importance.csv"

# человекочитаемые названия признаков
RU_NAMES = {
    "tx_sent_mean": "Настроение (среднее)",
    "tx_sent_std": "Настроение (разброс)",
    "tx_neg_share": "Доля негатива",
    "tx_pos_share": "Доля позитива",
    "tx_n_news": "Объём новостей",
    "tx_n_log": "Объём новостей (log)",
    "tx_sent_market": "Настроение рыночных тем",
    "tx_sent_toil": "Настроение: нефть",
    "tx_sent_tfx": "Настроение: валюта",
    "tx_sent_tsanc": "Настроение: санкции",
    "tx_sent_tbank": "Настроение: банки",
    "tx_sent_roll3": "Настроение (сглаж. 3 дня)",
    "tx_sent_roll5": "Настроение (сглаж. 5 дней)",
    "px_ret_lag1": "Доходность (вчера)",
    "px_ret_lag2": "Доходность (2 дня назад)",
    "px_ret_lag3": "Доходность (3 дня назад)",
    "px_ret_lag5": "Доходность (5 дней назад)",
    "px_vol5": "Волатильность (5 дней)",
    "px_vol10": "Волатильность (10 дней)",
    "px_rsi14": "RSI(14)",
    "px_macd": "MACD",
    "px_sma_ratio": "Отклонение от SMA(10)",
    "px_mom5": "Моментум (5 дней)",
    "px_vol_chg": "Изменение объёма торгов",
}


def run(top_n=15):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["date"])
    cols = [c for c in df.columns if c.startswith(("px_", "tx_"))]
    X, y = df[cols], df["y"]

    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)

    imp = (
        pd.DataFrame({"feature": cols, "importance": mean_abs})
        .assign(
            group=lambda d: np.where(d.feature.str.startswith("tx_"), "текст", "цена"),
            name_ru=lambda d: d.feature.map(RU_NAMES).fillna(d.feature),
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(IMP_CSV, index=False)

    # доля вклада текстовых vs ценовых признаков
    share = imp.groupby("group")["importance"].sum()
    share = (share / share.sum() * 100).round(1)

    print("Топ признаков по влиянию на прогноз (средний |SHAP|):")
    for r in imp.head(top_n).itertuples():
        print(f"  {r.importance:.4f}  [{r.group:5}] {r.name_ru}")
    print(f"\nСуммарный вклад групп: цена {share.get('цена',0)}% | "
          f"текст {share.get('текст',0)}%")

    # график
    top = imp.head(top_n).iloc[::-1]
    colors = ["#d62728" if g == "текст" else "#1f77b4" for g in top["group"]]
    plt.figure(figsize=(9, 6))
    plt.barh(top["name_ru"], top["importance"], color=colors)
    plt.xlabel("Среднее |SHAP| (влияние на прогноз)")
    plt.title("Влияние факторов на прогноз тренда IMOEX\n(синий — цена, красный — текст)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_importance.png", dpi=130)
    plt.close()
    print(f"\nГрафик: {FIG_DIR / 'shap_importance.png'}")
    print(f"Таблица: {IMP_CSV}")
    return imp


if __name__ == "__main__":
    print("Анализ важности признаков (SHAP)...")
    run()
