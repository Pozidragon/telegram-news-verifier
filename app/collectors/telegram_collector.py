from __future__ import annotations

from telethon import TelegramClient
from telethon.tl.types import Message
from typing import Iterable

from app.config import Settings
from app.models import TelegramPost


class TelegramCollector:
    def __init__(self) -> None:
        if not Settings.TELEGRAM_API_ID or not Settings.TELEGRAM_API_HASH:
            raise ValueError("Telegram API credentials are not configured in .env")

        self.client = TelegramClient(
            Settings.TELEGRAM_SESSION_NAME,
            Settings.TELEGRAM_API_ID,
            Settings.TELEGRAM_API_HASH,
        )

    async def collect_channel_posts(self, channel: str, limit: int = 100) -> list[TelegramPost]:
        posts: list[TelegramPost] = []

        async with self.client:
            async for message in self.client.iter_messages(channel, limit=limit):
                if not self._is_valid_message(message):
                    continue

                post_url = self._build_post_url(channel, message.id)

                posts.append(
                    TelegramPost(
                        channel=channel,
                        message_id=message.id,
                        text=message.message,
                        published_at=message.date,
                        post_url=post_url,
                    )
                )

        return posts

    @staticmethod
    def _is_valid_message(message: Message) -> bool:
        return bool(message.message and message.message.strip())

    @staticmethod
    def _build_post_url(channel: str, message_id: int) -> str | None:
        if channel.startswith("@"):
            return f"https://t.me/{channel[1:]}/{message_id}"
        return None