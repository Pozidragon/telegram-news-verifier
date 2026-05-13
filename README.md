# Telegram News Verifier

Master's thesis project for automated formation of a verified Ukrainian-language corpus from Telegram messages and trusted news sources. The system collects posts from Ukrainian Telegram channels, fetches RSS news articles, and verifies whether posts match real news using two independent approaches (TF-IDF and semantic embeddings).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
cp .env.example .env            # Then fill in TELEGRAM_API_ID and TELEGRAM_API_HASH
```

## Running the pipeline

```bash
# 1. Collect data (Telegram requires API credentials in .env)
python scripts/collect_telegram.py
python scripts/collect_news.py

# 2. Run verification
python scripts/run_verification.py            # TF-IDF cosine similarity
python scripts/run_semantic_verification.py   # Sentence-transformers (~400 MB model on first run)

# 3. Evaluate against labeled sample
python scripts/run_experiment.py              # TF-IDF evaluation → data/experiments/experiment_results.json
python scripts/run_semantic_experiment.py     # Semantic evaluation → data/experiments/semantic_experiment_results.json
```

## Tech stack

- **Data collection:** Telethon (Telegram MTProto), feedparser, BeautifulSoup
- **Ukrainian NLP:** pymorphy3 + pymorphy3-dicts-uk (morphological lemmatization)
- **TF-IDF verification:** scikit-learn TfidfVectorizer + cosine similarity
- **Semantic verification:** sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Storage:** JSONL (one JSON object per line)

## Project structure

```
app/                    Core modules (collectors, preprocessing, verification, evaluation)
scripts/                Entry-point scripts
data/raw/               Collected Telegram posts and news articles
data/processed/         Verification results
data/experiments/       Labeled samples, grid search results, experiment outputs
docs/                   Thesis documents and defense notes
```

See `CLAUDE.md` for full architecture documentation.
