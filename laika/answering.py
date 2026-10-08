"""The question-answering pipeline.

1. Answer simple questions about Laika itself without calling the model.
2. Retrieve relevant chunks from the documents and answer from them.
3. If the documents do not contain the answer, fall back to a web search.
4. If that also fails, fall back to Wikipedia.
"""

import logging
from collections.abc import Callable, Sequence

import wikipedia
from ddgs import DDGS

from laika import prompts
from laika.config import settings
from laika.retrieval import HybridIndex, cosine_similarity, generate_query_variants

logger = logging.getLogger(__name__)

Ask = Callable[[str], str]
Embed = Callable[[str], Sequence[float]]

ABOUT_LAIKA = {
    "who are you": "I am Laika, an AI assistant that answers questions about your PDF documents.",
    "what is your name": "My name is Laika.",
    "what can you do": (
        "I read the PDF documents you upload and answer questions about them. "
        "I can also summarise them, and I search the web when the documents do not have the answer."
    ),
}


def canned_answer(question: str, document_count: int) -> str | None:
    """Answer questions about Laika itself without calling the model."""
    q = question.lower()
    if "how many" in q and ("pdf" in q or "document" in q):
        return f"You have uploaded {document_count} document(s)."
    for phrase, answer in ABOUT_LAIKA.items():
        if phrase in q:
            return answer
    return None


def find_similar_turn(
    question: str, history: Sequence[dict], embed: Embed, threshold: float = settings.similar_question_threshold
) -> dict | None:
    """Return the earlier turn whose question is closest to this one, if it is close enough."""
    if not history:
        return None
    target = embed(question)
    score, best = max(
        ((cosine_similarity(target, embed(turn["question"])), turn) for turn in history),
        key=lambda pair: pair[0],
    )
    return best if score >= threshold else None


def answer_from_documents(question: str, chunks: Sequence[str], ask: Ask, previous: dict | None = None) -> str | None:
    """Answer from the retrieved chunks. Return None if they do not contain the answer."""
    history = ""
    if previous:
        history = f"A similar earlier question was:\nQ: {previous['question']}\nA: {previous['response']}\n\n"
    reply = ask(
        prompts.DOCUMENT_QA.format(
            not_found=prompts.NOT_FOUND,
            history=history,
            context="\n\n---\n\n".join(chunks),
            question=question,
        )
    )
    return None if not reply or prompts.NOT_FOUND in reply else reply


def answer_from_web(question: str, ask: Ask) -> str | None:
    try:
        results = DDGS().text(question, max_results=5)
    except Exception as error:
        logger.warning("Web search failed: %s", error)
        return None
    if not results:
        return None
    formatted = "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)
    return ask(prompts.WEB_ANSWER.format(question=question, results=formatted))


def answer_from_wikipedia(question: str) -> str | None:
    try:
        return wikipedia.summary(question, sentences=5, auto_suggest=True)
    except Exception as error:
        logger.warning("Wikipedia lookup failed: %s", error)
        return None


def answer_question(
    question: str,
    index: HybridIndex | None,
    history: Sequence[dict],
    ask: Ask,
    embed: Embed,
    document_count: int = 0,
) -> str:
    """Run the full pipeline and return a Markdown answer."""
    if canned := canned_answer(question, document_count):
        return canned

    if index is not None:
        queries = generate_query_variants(question, ask)
        chunks = index.search(queries)
        previous = find_similar_turn(question, history, embed)
        if answer := answer_from_documents(question, chunks, ask, previous):
            return answer

    if answer := answer_from_web(question, ask):
        return f"*I could not find this in your documents. Here is what I found on the web.*\n\n{answer}"

    if answer := answer_from_wikipedia(question):
        return f"*I could not find this in your documents. Here is a summary from Wikipedia.*\n\n{answer}"

    return "Sorry, I could not find a reliable answer to that question."
