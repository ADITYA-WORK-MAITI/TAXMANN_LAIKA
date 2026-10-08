"""Factories for the Gemini chat model and embedding model."""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from laika.config import settings


def require_api_key() -> str:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to a .env file or the environment.")
    return settings.google_api_key


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=require_api_key(),
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_retries=3,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=require_api_key())


def ask(prompt: str) -> str:
    """Send a single prompt to the chat model and return the text reply."""
    reply = get_chat_model().invoke(prompt)
    return reply.text.strip()
