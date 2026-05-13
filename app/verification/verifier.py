from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.models import TelegramPost, NewsArticle, VerificationResult
from app.preprocessing.cleaner import lemmatize_text, tokenize_keywords, extract_entities


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

    def precompute_articles(self, news_articles: Sequence[NewsArticle]) -> dict:
        """Lemmatize and tokenize all articles once so verify_post can reuse results."""
        cache: dict[str, dict] = {}
        for a in news_articles:
            key = a.url or a.title
            if key not in cache:
                cache[key] = {
                    "title": lemmatize_text(a.title),
                    "body": lemmatize_text(a.text[:1500]),
                    "combo": lemmatize_text(f"{a.title}. {a.text[:1500]}"),
                    "keywords": tokenize_keywords(f"{a.title} {a.text[:600]}"),
                    "entities": extract_entities(f"{a.title} {a.text[:600]}"),
                }
        return cache

    def verify_post(
        self,
        post: TelegramPost,
        news_articles: Sequence[NewsArticle],
        article_cache: dict | None = None,
    ) -> VerificationResult:
        if len(post.text.strip()) < Settings.MIN_POST_LENGTH:
            return self._empty_result(post)

        candidates = self._filter_by_time(post, news_articles)
        if not candidates:
            return self._empty_result(post)

        post_lemmatized = lemmatize_text(post.text)
        post_keywords = tokenize_keywords(post.text)
        post_entities = extract_entities(post.text)

        n = len(candidates)
        if article_cache:
            def _get(a: NewsArticle, field: str) -> str:
                return article_cache[a.url or a.title][field]
            article_titles = [_get(a, "title") for a in candidates]
            article_bodies = [_get(a, "body") for a in candidates]
            article_combos = [_get(a, "combo") for a in candidates]
            keyword_overlaps = [len(post_keywords & article_cache[a.url or a.title]["keywords"]) for a in candidates]
            entity_overlaps = [len(post_entities & article_cache[a.url or a.title]["entities"]) for a in candidates]
        else:
            article_titles = [lemmatize_text(a.title) for a in candidates]
            article_bodies = [lemmatize_text(a.text[:1500]) for a in candidates]
            article_combos = [lemmatize_text(f"{a.title}. {a.text[:1500]}") for a in candidates]
            keyword_overlaps = [
                len(post_keywords & tokenize_keywords(f"{a.title} {a.text[:600]}"))
                for a in candidates
            ]
            entity_overlaps = [
                len(post_entities & extract_entities(f"{a.title} {a.text[:600]}"))
                for a in candidates
            ]

        # Fit one vectorizer across post + all candidate representations so IDF is meaningful
        all_texts = [post_lemmatized] + article_titles + article_bodies + article_combos
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(all_texts)

        post_vec = matrix[0:1]
        title_scores = cosine_similarity(post_vec, matrix[1:n + 1])[0]
        body_scores = cosine_similarity(post_vec, matrix[n + 1:2 * n + 1])[0]
        combo_scores = cosine_similarity(post_vec, matrix[2 * n + 1:3 * n + 1])[0]

        # Compute base scores (TF-IDF + keyword penalty) before any bonuses
        pre_bonus = []
        for i, article in enumerate(candidates):
            overlap = keyword_overlaps[i]
            base = max(
                title_scores[i] * 1.15,
                body_scores[i],
                combo_scores[i] * 1.10,
            )
            if overlap < Settings.MIN_KEYWORD_OVERLAP:
                base *= 0.65
            pre_bonus.append((article, base, overlap, entity_overlaps[i]))

        # Apply entity bonus (grid-search optimal: 0.14 per matching named entity)
        scored_candidates = []
        for article, base, overlap, ent_overlap in pre_bonus:
            score = min(1.0, base * (1 + 0.14 * ent_overlap)) if ent_overlap > 0 else base
            scored_candidates.append((article, score, overlap))

        best_article, best_score, best_overlap = max(scored_candidates, key=lambda x: x[1])
        best_score = min(1.0, best_score)

        all_scores = sorted([s for _, s, _ in scored_candidates], reverse=True)
        top3_scores = [round(s, 4) for s in all_scores[:3]]

        # Store top-5 base scores + entity overlaps for bonus grid search
        top5_pre = sorted(pre_bonus, key=lambda x: x[1], reverse=True)[:5]
        top5_base_scores = [round(x[1], 4) for x in top5_pre]
        top5_entity_overlaps = [x[3] for x in top5_pre]

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
            top3_scores=top3_scores,
            top5_base_scores=top5_base_scores,
            top5_entity_overlaps=top5_entity_overlaps,
            status=status,
        )

    def _empty_result(self, post: TelegramPost) -> VerificationResult:
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
