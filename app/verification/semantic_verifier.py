from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import Settings
from app.models import TelegramPost, NewsArticle, VerificationResult
from app.preprocessing.cleaner import clean_text, tokenize_keywords, extract_entities


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
        post_entities = extract_entities(post.text)

        prefiltered: list[tuple[NewsArticle, int]] = []
        for article in candidates:
            article_keywords = tokenize_keywords(f"{article.title} {article.text[:600]}")
            overlap = len(post_keywords & article_keywords)
            prefiltered.append((article, overlap))

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

        # Compute base scores (semantic similarity) before any bonuses
        pre_bonus = []
        for article, overlap, score in zip(candidate_articles, overlaps, similarities):
            base = max(0.0, min(1.0, float(score)))
            article_entities = extract_entities(f"{article.title} {article.text[:600]}")
            ent_overlap = len(post_entities & article_entities)
            pre_bonus.append((article, base, overlap, ent_overlap))

        # Apply entity bonus (grid-search optimal: 0.04 per matching named entity)
        scored = []
        for article, base, overlap, ent_overlap in pre_bonus:
            s = min(1.0, base * (1 + 0.04 * ent_overlap)) if ent_overlap > 0 else base
            scored.append((article, s, overlap))

        best_article, best_score, best_overlap = max(scored, key=lambda x: x[1])

        # Corroboration bonus (grid-search optimal: 0.07 per corroborating article)
        all_scores = sorted([s for _, s, _ in scored], reverse=True)
        soft_threshold = self.threshold_uncertain * 0.7
        n_corroborating = sum(1 for s in all_scores[1:5] if s >= soft_threshold)
        best_score = min(1.0, best_score + 0.07 * n_corroborating)

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
            candidate_count=len(candidate_articles),
            top3_scores=top3_scores,
            top5_base_scores=top5_base_scores,
            top5_entity_overlaps=top5_entity_overlaps,
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