"""Анализ тональности новостей моделью RuBERT.

Модель: blanchefort/rubert-base-cased-sentiment (3 класса:
0=NEUTRAL, 1=POSITIVE, 2=NEGATIVE). Для каждой новости считаем вероятности
классов и итоговый score = pos - neg. Результат пишем в news_sentiment.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.storage import db

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
MAX_LEN = 256
BATCH = 32


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
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval().to(device)
        _CACHE.update(model=model, tok=tok, device=device)
    return _CACHE["model"], _CACHE["tok"], _CACHE["device"]


def score_texts(texts):
    """Оценивает список текстов. Возвращает список словарей
    {label, neu, pos, neg, score}. Переиспользуется для архива и live-новостей."""
    model, tok, device = load_model()
    out = []
    for start in range(0, len(texts), BATCH):
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
    return out


def run():
    db.init_db()
    rows = db.load_clean_all()
    total = len(rows)
    print(f"  устройство: {load_model()[2]}", flush=True)
    print(f"  к анализу: {total} новостей", flush=True)

    scores = score_texts([r[2] for r in rows])
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
