from pathlib import Path
from collections import Counter

from app.config import Settings
from app.models import TelegramPost, NewsArticle
from app.storage.json_repository import load_jsonl, save_jsonl
from app.verification.verifier import NewsVerifier


def load_all(directory: Path) -> list[dict]:
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {directory}")
    rows = []
    for f in files:
        rows.extend(load_jsonl(f))
    return rows


def main() -> None:
    Settings.ensure_dirs()

    telegram_posts = [TelegramPost.model_validate(item) for item in load_all(Settings.TELEGRAM_RAW_DIR)]
    news_articles = [NewsArticle.model_validate(item) for item in load_all(Settings.NEWS_RAW_DIR)]

    verifier = NewsVerifier()
    print(f"Precomputing lemmatized article cache ({len(news_articles)} articles)...")
    article_cache = verifier.precompute_articles(news_articles)
    print(f"Verifying {len(telegram_posts)} posts...")
    results = [verifier.verify_post(post, news_articles, article_cache).model_dump(mode="json") for post in telegram_posts]

    out_path = Settings.PROCESSED_DIR / "verification_results.jsonl"
    save_jsonl(out_path, results)

    counts = Counter(item["status"] for item in results)

    print(f"Saved {len(results)} verification results to {out_path}")
    print("Status distribution:")
    for status, count in counts.items():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()