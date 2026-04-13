import asyncio
from datetime import datetime

from app.collectors.telegram_collector import TelegramCollector
from app.config import Settings
from app.storage.json_repository import save_jsonl


CHANNELS = [
    "@suspilnenews",
    "@times_ukraina",
    "@truexanewsua"
]


async def main() -> None:
    Settings.ensure_dirs()

    collector = TelegramCollector()
    all_posts = []

    for channel in CHANNELS:
        posts = await collector.collect_channel_posts(channel, limit=200)
        all_posts.extend([post.model_dump(mode="json") for post in posts])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Settings.TELEGRAM_RAW_DIR / f"telegram_posts_{timestamp}.jsonl"
    save_jsonl(out_path, all_posts)

    print(f"Saved {len(all_posts)} telegram posts to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())