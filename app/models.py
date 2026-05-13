from __future__ import annotations

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, HttpUrl


class TelegramPost(BaseModel):
    source_type: Literal["telegram"] = "telegram"
    channel: str
    message_id: int
    text: str
    published_at: datetime
    post_url: Optional[HttpUrl] = None


class NewsArticle(BaseModel):
    source_type: Literal["news"] = "news"
    source_name: str
    title: str
    text: str
    url: HttpUrl
    published_at: Optional[datetime] = None


class VerificationResult(BaseModel):
    telegram_channel: str
    telegram_message_id: int
    telegram_text: str

    matched_news_url: Optional[HttpUrl] = None
    matched_news_source: Optional[str] = None
    matched_news_title: Optional[str] = None

    similarity_score: float = Field(ge=0.0, le=1.0)
    threshold_verified: float = Field(ge=0.0, le=1.0)
    threshold_uncertain: float = Field(ge=0.0, le=1.0)

    keyword_overlap: int = 0
    candidate_count: int = 0
    top3_scores: list[float] = Field(default_factory=list)

    # Intermediate values for bonus grid search (base scores before entity/corroboration bonuses)
    top5_base_scores: list[float] = Field(default_factory=list)
    top5_entity_overlaps: list[int] = Field(default_factory=list)

    status: Literal["verified", "uncertain", "unverified"]