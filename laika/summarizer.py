"""Document summaries using a map-reduce approach.

Long texts are split into pieces, each piece is summarised, and the piece
summaries are then combined into one.
"""

from collections.abc import Callable

from laika import prompts
from laika.config import settings
from laika.documents import split_fixed

Ask = Callable[[str], str]


def summarize(text: str, ask: Ask, chunk_size: int = settings.summary_chunk_size) -> str:
    pieces = split_fixed(text, chunk_size)
    if not pieces:
        return ""
    summaries = [ask(prompts.SUMMARY.format(text=piece)) for piece in pieces]
    if len(summaries) == 1:
        return summaries[0]
    return combine(summaries, ask)


def combine(summaries: list[str], ask: Ask) -> str:
    """Merge several summaries into one."""
    if len(summaries) == 1:
        return summaries[0]
    return ask(prompts.COMBINE_SUMMARIES.format(text="\n\n".join(summaries)))
