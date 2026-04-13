from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

from app.models import NewsArticle


class NewsCollector:
    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    def collect_from_rss(self, source_name: str, rss_url: str, limit: int = 50) -> list[NewsArticle]:
        feed = feedparser.parse(rss_url)
        articles: list[NewsArticle] = []

        for entry in feed.entries[:limit]:
            url = getattr(entry, "link", None)
            title = getattr(entry, "title", "").strip()

            if not url or not title:
                continue

            article_text = self._extract_article_text(url)
            if not article_text:
                article_text = self._extract_rss_fallback_text(entry)

            published_at = self._parse_published(entry)

            if not article_text:
                continue

            articles.append(
                NewsArticle(
                    source_name=source_name,
                    title=title,
                    text=article_text,
                    url=url,
                    published_at=published_at,
                )
            )

        return articles

    def _extract_article_text(self, url: str) -> str:
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
            )
            response.raise_for_status()
        except requests.RequestException:
            return ""

        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paragraphs if p)
        return text.strip()

    def _extract_rss_fallback_text(self, entry) -> str:
        parts: list[str] = []

        summary = getattr(entry, "summary", "")
        if summary:
            parts.append(self._strip_html(summary))

        description = getattr(entry, "description", "")
        if description and description != summary:
            parts.append(self._strip_html(description))

        contents = getattr(entry, "content", [])
        if contents:
            for item in contents:
                value = item.get("value", "")
                if value:
                    parts.append(self._strip_html(value))

        text = " ".join(part for part in parts if part).strip()
        return text

    @staticmethod
    def _strip_html(html: str) -> str:
        if "<" not in html and ">" not in html:
            return html.strip()
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(" ", strip=True)

    @staticmethod
    def _parse_published(entry) -> Optional[datetime]:
        published_parsed = getattr(entry, "published_parsed", None)
        if not published_parsed:
            return None
        return datetime(*published_parsed[:6], tzinfo=timezone.utc)