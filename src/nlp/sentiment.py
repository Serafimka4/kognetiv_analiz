"""Анализ тональности новостей моделью RuBERT.

Модель: blanchefort/rubert-base-cased-sentiment (3 класса:
0=NEUTRAL, 1=POSITIVE, 2=NEGATIVE). Для каждой новости считаем вероятности
классов и итоговый score = pos - neg. Результат пишем в news_sentiment.
"""
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.storage import db

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
# Длину можно уменьшить для ускорения на CPU: заголовок+анонс обычно короткие.
MAX_LEN = int(os.getenv("SENTIMENT_MAX_LEN", "192"))
BATCH = int(os.getenv("SENTIMENT_BATCH", "32"))


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_CACHE = {}


def load_model():
    """Лениво загружает и кэширует модель + токенизатор + устройство."""
    if "model" not in _CACHE:
        device = _device()
        if device.type == "cpu":
            # задействуем все ядра процессора
            torch.set_num_threads(os.cpu_count() or 1)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval().to(device)
        _CACHE.update(model=model, tok=tok, device=device)
    return _CACHE["model"], _CACHE["tok"], _CACHE["device"]


def score_texts(texts, progress=False):
    """Оценивает список текстов. Возвращает список словарей
    {label, neu, pos, neg, score}. Переиспользуется для архива и live-новостей.
    progress=True печатает ход выполнения с оценкой оставшегося времени."""
    model, tok, device = load_model()
    out = []
    total = len(texts)
    t0 = time.time()
    for start in range(0, total, BATCH):
        batch = [t or "" for t in texts[start:start + BATCH]]
        enc = tok(batch, return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1).cpu()
        for p in probs:
            neu, pos, neg = float(p[0]), float(p[1]), float(p[2])
            out.append({
                "label": model.config.id2label[int(p.argmax())],
                "neu": neu, "pos": pos, "neg": neg, "score": pos - neg,
            })
        if progress and ((start // BATCH) % 10 == 0 or start + BATCH >= total):
            done = len(out)
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed else 0
            eta = (total - done) / speed if speed else 0
            print(f"  тональность: {done}/{total} "
                  f"({speed:.0f} новостей/с, осталось ~{eta/60:.1f} мин)",
                  flush=True)
    return out


def run():
    db.init_db()
    rows = db.load_clean_all()
    total = len(rows)
    print(f"  устройство: {load_model()[2]} | max_len={MAX_LEN} batch={BATCH}",
          flush=True)
    print(f"  к анализу: {total} новостей", flush=True)

    scores = score_texts([r[2] for r in rows], progress=True)
    results = [
        (r[0], r[1], s["label"], s["neu"], s["pos"], s["neg"], s["score"])
        for r, s in zip(rows, scores)
    ]
    for s in range(0, len(results), 2000):
        db.save_news_sentiment(results[s:s + 2000])

    print(f"Готово: тональность по {len(results)} новостям.", flush=True)
    return len(results)


if __name__ == "__main__":
    print("Анализ тональности (RuBERT)...")
    run()
