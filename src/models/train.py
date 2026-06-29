"""Этап 5: прогноз направления тренда IMOEX и проверка гипотезы диплома.

Сравниваем наборы признаков на ОДНОЙ схеме walk-forward валидации:
  - baseline   — всегда «рост» (наивный прогноз);
  - price      — только ценовые/технические признаки (px_*);
  - text       — только текстовые признаки (tx_*);
  - price+text — объединённые признаки.

Гипотеза: набор price+text даёт точность выше, чем price.
Метрики усредняются по разбиениям TimeSeriesSplit (расширяющееся окно).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

import config

DATA = config.DATA_DIR / "dataset.csv"
N_SPLITS = 5
RESULTS_CSV = config.DATA_DIR / "model_results.csv"


def _model():
    return XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )


def _evaluate(df, cols, name):
    """Walk-forward оценка XGBoost на наборе признаков cols."""
    X, y = df[cols].values, df["y"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs, f1s, aucs = [], [], []
    for tr, te in tscv.split(X):
        m = _model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        proba = m.predict_proba(X[te])[:, 1]
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, zero_division=0))
        try:
            aucs.append(roc_auc_score(y[te], proba))
        except ValueError:
            aucs.append(np.nan)
    return {
        "model": name, "n_features": len(cols),
        "accuracy": np.mean(accs), "f1": np.mean(f1s),
        "roc_auc": np.nanmean(aucs),
    }


def _baseline(df):
    """Наивный прогноз: всегда мажоритарный класс на тесте каждого фолда."""
    y = df["y"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    accs = []
    for tr, te in tscv.split(y):
        pred = 1 if y[tr].mean() >= 0.5 else 0
        accs.append(accuracy_score(y[te], np.full(len(te), pred)))
    return {"model": "baseline (always-majority)", "n_features": 0,
            "accuracy": np.mean(accs), "f1": np.nan, "roc_auc": np.nan}


def run():
    df = pd.read_csv(DATA, parse_dates=["date"])
    px_cols = [c for c in df.columns if c.startswith("px_")]
    tx_cols = [c for c in df.columns if c.startswith("tx_")]

    rows = [
        _baseline(df),
        _evaluate(df, px_cols, "price (только цена)"),
        _evaluate(df, tx_cols, "text (только текст)"),
        _evaluate(df, px_cols + tx_cols, "price+text (цена+текст)"),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_CSV, index=False)

    print(f"\nДанные: {len(df)} дней | walk-forward, {N_SPLITS} разбиений")
    print(f"Признаки: цена={len(px_cols)}, текст={len(tx_cols)}\n")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    acc_base = res.loc[res.model.str.startswith("baseline"), "accuracy"].values[0]
    acc_px = res.loc[res.model.str.startswith("price ("), "accuracy"].values[0]
    acc_pt = res.loc[res.model.str.startswith("price+text"), "accuracy"].values[0]
    auc_pt = res.loc[res.model.str.startswith("price+text"), "roc_auc"].values[0]
    delta = (acc_pt - acc_px) * 100

    print(f"\nГипотеза (price+text > price): Δaccuracy = {delta:+.2f} п.п.")
    if delta > 0.3:
        print("• Текстовые признаки УЛУЧШАЮТ прогноз относительно только ценовых.")
    else:
        print("• Текстовые признаки не дают значимого улучшения.")
    # честная оговорка: насколько модель вообще выше случайной
    print(f"• Абсолютная сила прогноза НИЗКАЯ: accuracy={acc_pt:.3f} при baseline="
          f"{acc_base:.3f}, ROC-AUC={auc_pt:.3f} (≈0.5 — близко к случайной).")
    print("• Это согласуется с гипотезой эффективного рынка и слабой корреляцией\n"
          "  настроений общих новостей Lenta с доходностью IMOEX.")
    return res


if __name__ == "__main__":
    print("Обучение и сравнение моделей...")
    run()
