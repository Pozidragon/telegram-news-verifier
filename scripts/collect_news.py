from datetime import datetime

from app.collectors.news_collector import NewsCollector
from app.config import Settings
from app.storage.json_repository import save_jsonl


RSS_SOURCES = {
    "Suspilne": "https://suspilne.media/rss/all.rss",
    "Korespondent": "http://k.img.com.ua/rss/ua/all_news2.0.xml",
    "RBC": "https://www.rbc.ua/static/rss/all.ukr.rss.xml"
}


def main() -> None:
    Settings.ensure_dirs()

    collector = NewsCollector()
    all_articles = []

    NEWS_LIMIT_PER_SOURCE = 150

    for source_name, rss_url in RSS_SOURCES.items():
        articles = collector.collect_from_rss(source_name, rss_url, limit=NEWS_LIMIT_PER_SOURCE)
        print(f"{source_name}: collected {len(articles)} articles")
        all_articles.extend([article.model_dump(mode="json") for article in articles])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Settings.NEWS_RAW_DIR / f"news_articles_{timestamp}.jsonl"
    save_jsonl(out_path, all_articles)

    print(f"Saved {len(all_articles)} news articles to {out_path}")


if __name__ == "__main__":
    main()