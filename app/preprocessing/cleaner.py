from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
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


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.replace("\n", " ")
    text = NON_WORD_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def tokenize_keywords(text: str) -> set[str]:
    cleaned = clean_text(text)
    tokens = cleaned.split()
    return {
        token for token in tokens
        if len(token) >= 4 and token not in STOPWORDS
    }