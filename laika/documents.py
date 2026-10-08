"""PDF text extraction and chunking."""

import logging
from typing import BinaryIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from laika.config import settings

logger = logging.getLogger(__name__)


def extract_text(pdf: BinaryIO) -> str:
    """Return the text of every page of a PDF file."""
    pdf.seek(0)
    reader = PdfReader(pdf)
    pages = [page.extract_text() or "" for page in reader.pages]
    logger.info("Extracted %d pages from %s", len(pages), getattr(pdf, "name", "document"))
    return "\n".join(pages)


def split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_text(text)


def split_fixed(text: str, size: int) -> list[str]:
    """Split text into consecutive pieces of at most `size` characters."""
    return [text[i : i + size] for i in range(0, len(text), size)]
