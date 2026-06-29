"""Очистка и лемматизация русского текста новостей.

Готовим два представления:
  - clean_text() — естественный текст (для модели тональности RuBERT);
  - lemmatize()  — список лемм без стоп-слов (для тематического моделирования LDA).
"""
import re
from functools import lru_cache

import pymorphy3
from razdel import tokenize

from src.preprocessing.stopwords import RU_STOPWORDS

_morph = pymorphy3.MorphAnalyzer()

_RE_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_SPACE = re.compile(r"\s+")
_RE_WORD = re.compile(r"^[а-яё-]+$")  # только русские слова (для лемм)


def clean_text(text):
    """Лёгкая очистка для RuBERT: убираем теги, ссылки, лишние пробелы.
    Регистр и пунктуацию сохраняем — модель тональности работает с живым текстом."""
    if not text:
        return ""
    text = _RE_TAG.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = text.replace("\xa0", " ")
    return _RE_SPACE.sub(" ", text).strip()


@lru_cache(maxsize=200_000)
def _lemma(word):
    return _morph.parse(word)[0].normal_form


def lemmatize(text, min_len=3):
    """Возвращает список лемм: нижний регистр, только русские слова,
    длиной >= min_len, без стоп-слов."""
    if not text:
        return []
    lemmas = []
    for token in tokenize(text.lower()):
        w = token.text
        if len(w) < min_len or not _RE_WORD.match(w):
            continue
        lemma = _lemma(w)
        if lemma in RU_STOPWORDS or len(lemma) < min_len:
            continue
        lemmas.append(lemma)
    return lemmas
