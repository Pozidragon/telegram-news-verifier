from pathlib import Path
from collections import Counter

from app.config import Settings
from app.models import TelegramPost, NewsArticle
from app.storage.json_repository import load_jsonl, save_jsonl
from app.verification.semantic_verifier import SemanticNewsVerifier


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