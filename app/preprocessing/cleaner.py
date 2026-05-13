from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Capitalized Ukrainian/Latin words (proper nouns) and all-caps abbreviations (ЗСУ, НАТО)
_ENTITY_RE = re.compile(r"\b([А-ЯІЇЄҐA-Z][а-яіїєґa-z'\-]{2,}|[А-ЯІЇЄҐ]{2,})\b")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\s\u0400-\u04FF\u0500-\u052F\-]")

STOPWORDS = {
    "це", "цей", "ця", "ці", "та", "і", "й", "або", "але", "що", "як", "у", "в", "на",
    "до", "за", "про", "по", "не", "від", "із", "з", "для", "при", "після", "під",
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
    "это", "этот", "эта", "эти", "что", "как", "в", "на", "по", "за", "не", "из", "с", "для"
}

_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer(lang='uk')
    return _morph


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.replace("\n", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def lemmatize_text(text: str) -> str:
    """Clean and normalize each token to its dictionary form (for TF-IDF input)."""
    cleaned = clean_text(text)
    morph = _get_morph()
    lemmas = [morph.parse(t)[0].normal_form for t in cleaned.split() if t]
    return " ".join(lemmas)


def extract_entities(text: str) -> set[str]:
    """Extract named entity candidates: capitalized words and all-caps abbreviations, lemmatized."""
    t = URL_RE.sub(" ", text)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    morph = _get_morph()
    entities: set[str] = set()
    for m in _ENTITY_RE.finditer(t):
        word = m.group()
        lemma = morph.parse(word)[0].normal_form
        if len(lemma) >= 3:
            entities.add(lemma)
    return entities


def tokenize_keywords(text: str) -> set[str]:
    cleaned = clean_text(text)
    morph = _get_morph()
    result = set()
    for token in cleaned.split():
        if len(token) >= 4 and token not in STOPWORDS:
            lemma = morph.parse(token)[0].normal_form
            if len(lemma) >= 4 and lemma not in STOPWORDS:
                result.add(lemma)
    return result
