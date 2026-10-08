"""Settings read from the environment, with sensible defaults."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    chat_model: str = os.getenv("LAIKA_CHAT_MODEL", "gemini-flash-latest")
    embedding_model: str = os.getenv("LAIKA_EMBEDDING_MODEL", "models/gemini-embedding-001")
    temperature: float = 0.3
    request_timeout: int = 60

    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 5
    query_variants: int = 3
    similar_question_threshold: float = 0.8

    # Summarisation
    summary_chunk_size: int = 10_000

    data_dir: Path = Path(os.getenv("LAIKA_DATA_DIR", "data"))


settings = Settings()
