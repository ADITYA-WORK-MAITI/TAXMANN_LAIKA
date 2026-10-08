"""Offline stand-ins for the Gemini models, so tests need no API key."""

import hashlib

import numpy as np
from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    """Bag-of-words vectors built by hashing each word. Similar texts get similar vectors."""

    def __init__(self, size: int = 256):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.size)
        for word in text.lower().split():
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ScriptedModel:
    """Records every prompt and returns preset replies in order."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""
