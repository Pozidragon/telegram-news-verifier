from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.models import TelegramPost, NewsArticle, VerificationResult
from app.preprocessing.cleaner import clean_text, tokenize_keywords


class NewsVerifier:
    def __init__(
        self,
        threshold_verified: float | None = None,
        threshold_uncertain: float | None = None,
        time_window_hours: int | None = None,
    ) -> None:
        self.threshold_verified = (
            threshold_verified
            if threshold_verified is not None
            else Settings.SIMILARITY_THRESHOLD_VERIFIED
        )
        self.threshold_uncertain = (
            threshold_uncertain
            if threshold_uncertain is not None
            else Settings.SIMILARITY_THRESHOLD_UNCERTAIN
        )
        self.time_window_hours = (
            time_window_hours if time_window_hours is not None else Settings.TIME_WINDOW_HOURS
        )

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

        scored_candidates: list[tuple[NewsArticle, float, int]] = []

        for article in candidates:
            article_title = clean_text(article.title)
            article_body = clean_text(article.text[:1500])
            article_combo = clean_text(f"{article.title}. {article.text[:1500]}")
            article_keywords = tokenize_keywords(f"{article.title} {article.text[:600]}")

            keyword_overlap = len(post_keywords & article_keywords)

            title_score = self._similarity(post_clean, article_title)
            body_score = self._similarity(post_clean, article_body)
            combo_score = self._similarity(post_clean, article_combo)

            final_score = max(
                title_score * 1.15,
                body_score,
                combo_score * 1.10,
            )

            if keyword_overlap < Settings.MIN_KEYWORD_OVERLAP:
                final_score *= 0.65

            scored_candidates.append((article, final_score, keyword_overlap))

        if not scored_candidates:
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

        best_article, best_score, best_overlap = max(scored_candidates, key=lambda x: x[1])

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
            candidate_count=len(candidates),
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

    @staticmethod
    def _similarity(text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
        )
        matrix = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])