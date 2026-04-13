from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.models import TelegramPost, NewsArticle, VerificationResult
from app.preprocessing.cleaner import clean_text, tokenize_keywords


class SemanticNewsVerifier:
    _model: SentenceTransformer | None = None

    def __init__(
        self,
        threshold_verified: float | None = None,
        threshold_uncertain: float | None = None,
        time_window_hours: int | None = None,
        model_name: str | None = None,
    ) -> None:
        self.threshold_verified = (
            threshold_verified
            if threshold_verified is not None
            else Settings.SEMANTIC_THRESHOLD_VERIFIED
        )
        self.threshold_uncertain = (
            threshold_uncertain
            if threshold_uncertain is not None
            else Settings.SEMANTIC_THRESHOLD_UNCERTAIN
        )
        self.time_window_hours = (
            time_window_hours if time_window_hours is not None else Settings.TIME_WINDOW_HOURS
        )
        self.model_name = model_name if model_name is not None else Settings.SEMANTIC_MODEL_NAME

        if SemanticNewsVerifier._model is None:
            SemanticNewsVerifier._model = SentenceTransformer(self.model_name)

        self.model = SemanticNewsVerifier._model

    def verify_post(self, post: TelegramPost, news_articles: Sequence[NewsArticle]) -> VerificationResult:
        if len(post.text.strip()) < Settings.MIN_POST_LENGTH:
            return VerificationResult(
                telegram_channel=post.channel,
                telegram_message_id=post.message_id,
                telegram_text=post.text,
                similarity_score=0.0,
                threshold_verified=self.threshold_verified,
                threshold_uncertain=self.threshold_uncertain,
                keyword_overlap=0,
                candidate_count=0,
                status="unverified",
            )

        candidates = self._filter_by_time(post, news_articles)
        if not candidates:
            return VerificationResult(
                telegram_channel=post.channel,
                telegram_message_id=post.message_id,
                telegram_text=post.text,
                similarity_score=0.0,
                threshold_verified=self.threshold_verified,
                threshold_uncertain=self.threshold_uncertain,
                keyword_overlap=0,
                candidate_count=0,
                status="unverified",
            )

        post_clean = clean_text(post.text)
        post_keywords = tokenize_keywords(post.text)

        prefiltered: list[tuple[NewsArticle, int]] = []
        for article in candidates:
            article_keywords = tokenize_keywords(f"{article.title} {article.text[:600]}")
            overlap = len(post_keywords & article_keywords)
            if overlap >= 2:
                prefiltered.append((article, overlap))

        if not prefiltered:
            return VerificationResult(
                telegram_channel=post.channel,
                telegram_message_id=post.message_id,
                telegram_text=post.text,
                similarity_score=0.0,
                threshold_verified=self.threshold_verified,
                threshold_uncertain=self.threshold_uncertain,
                keyword_overlap=0,
                candidate_count=0,
                status="unverified",
            )

        prefiltered = sorted(prefiltered, key=lambda x: x[1], reverse=True)[: Settings.SEMANTIC_TOP_K]

        candidate_articles = [item[0] for item in prefiltered]
        overlaps = [item[1] for item in prefiltered]

        post_embedding = self.model.encode([post_clean], normalize_embeddings=True)
        candidate_texts = [
            clean_text(f"{article.title}. {article.text[:1500]}")
            for article in candidate_articles
        ]
        candidate_embeddings = self.model.encode(candidate_texts, normalize_embeddings=True)

        similarities = cosine_similarity(post_embedding, candidate_embeddings)[0]

        scored = []
        for article, overlap, score in zip(candidate_articles, overlaps, similarities):
            final_score = float(score)
            final_score = max(0.0, min(1.0, final_score))
            scored.append((article, final_score, overlap))

        best_article, best_score, best_overlap = max(scored, key=lambda x: x[1])

        if best_score >= self.threshold_verified:
            status = "verified"
        elif best_score >= self.threshold_uncertain:
            status = "uncertain"
        else:
            status = "unverified"

        return VerificationResult(
            telegram_channel=post.channel,
            telegram_message_id=post.message_id,
            telegram_text=post.text,
            matched_news_url=best_article.url,
            matched_news_source=best_article.source_name,
            matched_news_title=best_article.title,
            similarity_score=float(best_score),
            threshold_verified=self.threshold_verified,
            threshold_uncertain=self.threshold_uncertain,
            keyword_overlap=best_overlap,
            candidate_count=len(candidate_articles),
            status=status,
        )

    def _filter_by_time(self, post: TelegramPost, news_articles: Sequence[NewsArticle]) -> list[NewsArticle]:
        window = timedelta(hours=self.time_window_hours)
        candidates: list[NewsArticle] = []

        for article in news_articles:
            if article.published_at is None:
                candidates.append(article)
                continue

            if abs(post.published_at - article.published_at) <= window:
                candidates.append(article)

        return candidates