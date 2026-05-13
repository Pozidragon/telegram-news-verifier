# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Master's thesis project for automated formation of a verified Ukrainian-language corpus from Telegram messages and trusted news sources. The system collects posts from Ukrainian Telegram channels, fetches RSS news articles, and verifies whether posts match real news using two independent approaches (TF-IDF and semantic embeddings).

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env            # Fill in TELEGRAM_API_ID and TELEGRAM_API_HASH
```

## Running Scripts

All entry points live in `scripts/` and are run directly:

```bash
# Data collection
python scripts/collect_telegram.py       # Requires .env Telegram credentials
python scripts/collect_news.py

# Verification
python scripts/run_verification.py               # TF-IDF cosine similarity
python scripts/run_semantic_verification.py      # Sentence-transformers (downloads ~400 MB model on first run)

# Evaluation (requires manually labeled sample in data/experiments/labeled_sample.jsonl)
python scripts/run_experiment.py
python scripts/run_semantic_experiment.py

# Annotation workflow
python scripts/select_annotation_sample.py       # Stratified sampling for human labeling
python scripts/merge_annotation_samples.py
python scripts/build_labeled_sample.py
```

No test suite or linter is configured.

## Architecture

### Data flow

```
Telegram API (Telethon) → collect_telegram.py → data/raw/telegram/*.jsonl
RSS feeds (feedparser)  → collect_news.py     → data/raw/news/*.jsonl
                                                         ↓
                                          run_verification.py (TF-IDF)
                                          run_semantic_verification.py (embeddings)
                                                         ↓
                                         data/processed/verification_results.jsonl
                                                         ↓
                               human annotation → data/experiments/labeled_sample.jsonl
                                                         ↓
                                         run_experiment.py → experiment_results.json
```

### Core modules (`app/`)

| Module | Role |
|---|---|
| `models.py` | Pydantic models: `TelegramPost`, `NewsArticle`, `VerificationResult` |
| `config.py` | All settings loaded from env vars with defaults |
| `collectors/telegram_collector.py` | Async Telethon client; fetches from 3 hardcoded channels |
| `collectors/news_collector.py` | RSS + full-page HTML extraction via BeautifulSoup |
| `preprocessing/cleaner.py` | Ukrainian text cleaning: `clean_text`, `lemmatize_text` (pymorphy3 normalisation), `tokenize_keywords` |
| `verification/verifier.py` | TF-IDF verifier: time-window filter → keyword check → cosine similarity |
| `verification/semantic_verifier.py` | Semantic verifier: keyword pre-filter → top-K → MiniLM embeddings |
| `storage/json_repository.py` | `save_jsonl` / `load_jsonl` helpers |
| `evaluation/metrics.py` | Accuracy, precision/recall/F1 for binary and 3-class tasks |

### Verification logic

Both verifiers produce one of three labels — `"verified"`, `"uncertain"`, `"unverified"` — controlled by thresholds in `config.py`:

**TF-IDF (`NewsVerifier`)**: computes cosine similarity between the post and each candidate article's title, body, and title+body combined; takes the max; penalises low keyword overlap (×0.65 if fewer than `MIN_KEYWORD_OVERLAP` matching keywords).

Two improvements were made to the TF-IDF verifier:
1. **Corpus-level IDF**: the `TfidfVectorizer` is now fit once across the post + all candidate texts together, so IDF is meaningful. Previously it was fit on 2 documents per comparison, making IDF near-useless.
2. **Ukrainian morphological lemmatization**: `pymorphy3` (with `pymorphy3-dicts-uk`) normalises inflected word forms before vectorization and keyword matching (e.g. "Зеленського" → "зеленський"). This is applied via `lemmatize_text()` in `cleaner.py`, which is used as TF-IDF input. `tokenize_keywords()` also lemmatizes, so the semantic verifier's keyword pre-filter benefits too.

These changes required recalibrating thresholds (old 0.16/0.08 were tuned for per-pair TF-IDF scores). New thresholds 0.45/0.35 were found by grid search on the 200-item labeled sample and are set in `.env`.

**Semantic (`SemanticNewsVerifier`)**: pre-filters candidates by keyword overlap, then encodes post and top-K articles with `paraphrase-multilingual-MiniLM-L12-v2`; classifies by cosine similarity thresholds.

### Key configuration knobs (set in `.env`)

```
TIME_WINDOW_HOURS          # news age window relative to post (default 36)
SIMILARITY_THRESHOLD_VERIFIED / UNCERTAIN   # TF-IDF thresholds (0.45 / 0.35)
SEMANTIC_THRESHOLD_VERIFIED / UNCERTAIN     # embedding thresholds (0.55 / 0.40)
MIN_POST_LENGTH            # skip short posts (default 40 chars)
MIN_KEYWORD_OVERLAP        # minimum shared keywords (default 2)
SEMANTIC_TOP_K             # candidates passed to encoder (default 8)
SEMANTIC_MODEL_NAME        # HuggingFace model id
```

### Ukrainian NLP notes

- `clean_text` preserves Cyrillic range `\u0400–\u04FF` and strips URLs/mentions/hashtags.
- `tokenize_keywords` removes stopwords in Ukrainian, Russian, and English; drops tokens shorter than 4 characters.
- All data is stored as UTF-8 JSONL.
