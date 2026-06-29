"""Генерация диаграмм UML и BPMN для пояснительной записки.

Создаёт PNG в data/figures/: usecase.png, component.png, bpmn.png.
Запуск: python scripts/make_diagrams.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Rectangle

import config

FIG_DIR = config.DATA_DIR / "figures"
BLUE, GREY, DARK = "#1f77b4", "#f0f0f0", "#333333"


def _box(ax, x, y, w, h, text, fc="#eaf2fb", ec=BLUE, fs=10, rounded=True, bold=False):
    style = "round,pad=0.02,rounding_size=0.12" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def _arrow(ax, p1, p2, style="-|>", color=DARK, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=1.3, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- UML use case
def usecase():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")

    # актёр (аналитик)
    ax.add_patch(Circle((1.0, 4.7), 0.18, fc="white", ec=DARK, lw=1.5))
    ax.plot([1.0, 1.0], [4.52, 3.9], color=DARK, lw=1.5)
    ax.plot([0.6, 1.4], [4.3, 4.3], color=DARK, lw=1.5)
    ax.plot([1.0, 0.65], [3.9, 3.4], color=DARK, lw=1.5)
    ax.plot([1.0, 1.35], [3.9, 3.4], color=DARK, lw=1.5)
    ax.text(1.0, 3.1, "Аналитик", ha="center", fontsize=11, weight="bold")

    # граница системы
    ax.add_patch(Rectangle((3.0, 0.4), 6.7, 5.7, fc="none", ec=GREY, lw=1.5))
    ax.text(6.35, 5.8, "Система когнитивного анализа рыночных тенденций",
            ha="center", fontsize=10, style="italic", color="#666")

    cases = [
        (5.0, "Сбор котировок и новостей"),
        (4.2, "Анализ тональности новостей"),
        (3.4, "Выделение тем-драйверов"),
        (2.6, "Прогноз направления тренда"),
        (1.8, "Просмотр факторов влияния (SHAP)"),
        (1.0, "Мониторинг актуальных данных"),
    ]
    for y, label in cases:
        ax.add_patch(Ellipse((6.35, y), 4.8, 0.62, fc="#eaf2fb", ec=BLUE, lw=1.4))
        ax.text(6.35, y, label, ha="center", va="center", fontsize=9.5)
        _arrow(ax, (1.5, 3.95), (3.95, y), style="-", color="#888")

    fig.savefig(FIG_DIR / "usecase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- UML component
def component():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")

    comps = {
        "collect": (0.4, 5.2, 2.6, 0.9, "«component»\nСбор данных"),
        "store":   (3.7, 5.2, 2.6, 0.9, "«component»\nХранилище (SQLite)"),
        "prep":    (7.0, 5.2, 2.6, 0.9, "«component»\nОбработка текста"),
        "nlp":     (7.0, 3.6, 2.6, 0.9, "«component»\nNLP: тональность, темы"),
        "feat":    (3.7, 3.6, 2.6, 0.9, "«component»\nФормирование признаков"),
        "model":   (3.7, 2.0, 2.6, 0.9, "«component»\nПрогноз (XGBoost)"),
        "explain": (0.4, 2.0, 2.6, 0.9, "«component»\nИнтерпретация (SHAP)"),
        "dash":    (0.4, 0.4, 2.6, 0.9, "«component»\nПанель мониторинга"),
        "live":    (7.0, 2.0, 2.6, 0.9, "«component»\nРежим live"),
    }
    for k, (x, y, w, h, t) in comps.items():
        _box(ax, x, y, w, h, t, fs=9)
        # значок компонента UML
        ax.add_patch(Rectangle((x + w - 0.45, y + h - 0.32), 0.32, 0.2,
                               fc="white", ec=BLUE, lw=1))

    def c(k, side):
        x, y, w, h, _ = comps[k]
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
                "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]

    _arrow(ax, c("collect", "r"), c("store", "l"))
    _arrow(ax, c("prep", "l"), c("store", "r"))
    _arrow(ax, c("prep", "b"), c("nlp", "t"))
    _arrow(ax, c("nlp", "l"), c("feat", "r"))
    _arrow(ax, c("store", "b"), c("feat", "t"))
    _arrow(ax, c("feat", "b"), c("model", "t"))
    _arrow(ax, c("model", "l"), c("explain", "r"))
    _arrow(ax, c("explain", "b"), c("dash", "t"))
    _arrow(ax, c("store", "b"), c("live", "t"), ls="--")
    _arrow(ax, c("live", "b"), c("model", "r"), ls="--")

    fig.savefig(FIG_DIR / "component.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------- BPMN
def bpmn():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")
    y = 1.6

    # стартовое событие
    ax.add_patch(Circle((0.5, y), 0.28, fc="#e7f6e7", ec="#2ca02c", lw=2))
    ax.text(0.5, 0.95, "Старт", ha="center", fontsize=9)

    tasks = ["Сбор\nданных", "Предобработка\nтекста",
             "Тональность\nи темы", "Формирование\nпризнаков",
             "Прогноз\nтренда", "Интерпретация\n(SHAP)", "Визуа-\nлизация"]
    x = 1.2
    w, h, gap = 1.18, 0.9, 0.18
    centers = []
    for t in tasks:
        _box(ax, x, y - h / 2, w, h, t, fs=8.5)
        centers.append((x, x + w))
        x += w + gap

    # конечное событие
    ex = x + 0.1
    ax.add_patch(Circle((ex, y), 0.28, fc="#fde7e7", ec="#d62728", lw=3))
    ax.text(ex, 0.95, "Готово", ha="center", fontsize=9)

    # стрелки потока
    _arrow(ax, (0.78, y), (centers[0][0], y))
    for i in range(len(tasks) - 1):
        _arrow(ax, (centers[i][1], y), (centers[i + 1][0], y))
    _arrow(ax, (centers[-1][1], y), (ex - 0.28, y))

    ax.text(5.5, 2.9, "Процесс когнитивного анализа рыночных тенденций (нотация BPMN)",
            ha="center", fontsize=10, style="italic", color="#666")
    fig.savefig(FIG_DIR / "bpmn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    usecase(); component(); bpmn()
    print("Диаграммы созданы в", FIG_DIR)
