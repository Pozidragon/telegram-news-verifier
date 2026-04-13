from pathlib import Path
from collections import Counter

from app.config import Settings
from app.models import TelegramPost, NewsArticle
from app.storage.json_repository import load_jsonl, save_jsonl
from app.verification.semantic_verifier import SemanticNewsVerifier


def get_latest_file(directory: Path) -> Path:
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {directory}")
    return files[-1]


def main() -> None:
    Settings.ensure_dirs()

    telegram_file = get_latest_file(Settings.TELEGRAM_RAW_DIR)
    news_file = get_latest_file(Settings.NEWS_RAW_DIR)

    telegram_posts = [TelegramPost.model_validate(item) for item in load_jsonl(telegram_file)]
    news_articles = [NewsArticle.model_validate(item) for item in load_jsonl(news_file)]

    verifier = SemanticNewsVerifier()
    results = [verifier.verify_post(post, news_articles).model_dump(mode="json") for post in telegram_posts]

    out_path = Settings.PROCESSED_DIR / "semantic_verification_results.jsonl"
    save_jsonl(out_path, results)

    counts = Counter(item["status"] for item in results)

    print(f"Saved {len(results)} semantic verification results to {out_path}")
    print("Status distribution:")
    for status in ("verified", "uncertain", "unverified"):
        print(f"  {status}: {counts.get(status, 0)}")


if __name__ == "__main__":
    main()