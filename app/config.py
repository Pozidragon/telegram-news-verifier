from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    TELEGRAM_API_ID: int = int(os.getenv("TELEGRAM_API_ID", "0"))
    TELEGRAM_API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")
    TELEGRAM_SESSION_NAME: str = os.getenv("TELEGRAM_SESSION_NAME", "telegram_news_session")

    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    TELEGRAM_RAW_DIR: Path = Path(os.getenv("TELEGRAM_RAW_DIR", "data/raw/telegram"))
    NEWS_RAW_DIR: Path = Path(os.getenv("NEWS_RAW_DIR", "data/raw/news"))
    PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", "data/processed"))
    EXPERIMENTS_DIR: Path = Path(os.getenv("EXPERIMENTS_DIR", "data/experiments"))

    TIME_WINDOW_HOURS: int = int(os.getenv("TIME_WINDOW_HOURS", "36"))
    SIMILARITY_THRESHOLD_VERIFIED: float = float(os.getenv("SIMILARITY_THRESHOLD_VERIFIED", "0.16"))
    SIMILARITY_THRESHOLD_UNCERTAIN: float = float(os.getenv("SIMILARITY_THRESHOLD_UNCERTAIN", "0.08"))
    MIN_POST_LENGTH: int = int(os.getenv("MIN_POST_LENGTH", "40"))
    MIN_KEYWORD_OVERLAP: int = int(os.getenv("MIN_KEYWORD_OVERLAP", "2"))

    SEMANTIC_MODEL_NAME: str = os.getenv(
        "SEMANTIC_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    SEMANTIC_THRESHOLD_VERIFIED: float = float(os.getenv("SEMANTIC_THRESHOLD_VERIFIED", "0.55"))
    SEMANTIC_THRESHOLD_UNCERTAIN: float = float(os.getenv("SEMANTIC_THRESHOLD_UNCERTAIN", "0.40"))
    SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "8"))

    @classmethod
    def ensure_dirs(cls) -> None:
        for path in [
            cls.DATA_DIR,
            cls.TELEGRAM_RAW_DIR,
            cls.NEWS_RAW_DIR,
            cls.PROCESSED_DIR,
            cls.EXPERIMENTS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)