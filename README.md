# Laika

[![CI](https://github.com/ADITYA-WORK-MAITI/TAXMANN_LAIKA/actions/workflows/ci.yml/badge.svg)](https://github.com/ADITYA-WORK-MAITI/TAXMANN_LAIKA/actions/workflows/ci.yml)

Laika is a chat assistant for PDF documents. You upload tax or legal documents and ask questions about them. Laika finds the relevant passages and answers from them. When the documents do not contain the answer, it searches the web.

It is built with Streamlit, Google Gemini, FAISS and BM25.

I built Laika during my internship at [Taxmann](https://www.taxmann.com), a publisher of tax and legal content in India. This is why the interface carries the Taxmann name.

## Features

- Questions and answers over one or more uploaded PDFs.
- Hybrid retrieval that combines meaning-based search and keyword search.
- Summaries of each document and of the whole set.
- Web search and Wikipedia as fallbacks when the documents have no answer.
- Saved conversations and CSV export.

## How it works

```
PDF files ──► text extraction ──► chunks ──┬─► Gemini embeddings ─► FAISS index ─┐
                                           └─► tokens ────────────► BM25 index ──┤
                                                                                 │
question ──► Gemini writes 3 rephrasings ──► each one searches both indexes ─────┤
                                                                                 ▼
                                               Reciprocal Rank Fusion ─► top 5 chunks
                                                                                 │
                                                                                 ▼
                                    Gemini answers from the chunks, or replies NOT_FOUND
                                                                                 │
                                                       NOT_FOUND ─► web search ─► Wikipedia
```

**Hybrid retrieval.** Vector search finds passages with a similar meaning, even when the wording differs. BM25 finds passages that share exact terms, such as section numbers and form names. Tax documents need both.

**RAG Fusion.** One question can be phrased in many ways. The model writes a few rephrasings, and each one is run against both indexes. Reciprocal Rank Fusion then merges all the ranked lists. A passage that ranks well in many lists ends up first.

**Grounded answers.** The model must answer only from the retrieved passages. If they do not contain the answer, it returns a fixed `NOT_FOUND` marker. Only then does Laika use the web.

**Conversation memory.** If a new question is very similar to an earlier one (cosine similarity of 0.8 or more), the earlier answer is added to the prompt as extra context.

**Summaries.** Long documents are split into pieces. Each piece is summarised, and the piece summaries are then combined (map-reduce).

## Project structure

```
.
├── app.py                 Streamlit interface: pages, sidebar and session state
├── laika/
│   ├── config.py          Settings read from environment variables
│   ├── llm.py             Gemini chat and embedding models
│   ├── documents.py       PDF text extraction and chunking
│   ├── retrieval.py       FAISS + BM25 hybrid index and Reciprocal Rank Fusion
│   ├── answering.py       Question-answering pipeline with fallbacks
│   ├── summarizer.py      Map-reduce summaries
│   ├── prompts.py         Prompt templates
│   └── storage.py         Saved conversations and CSV export
├── assets/style.css       Interface styling
├── tests/                 Unit tests that run offline with fake models
├── render.yaml            Deployment settings for Render
└── .github/workflows/     Lint and test pipeline
```

The `laika` package does not depend on Streamlit. The model is passed into the pipeline as a function. This keeps the logic separate from the interface, and it lets the tests replace Gemini with simple fakes.

## Running locally

You need Python 3.11 or newer and a Google Gemini API key. You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

```bash
git clone https://github.com/ADITYA-WORK-MAITI/TAXMANN_LAIKA.git
cd TAXMANN_LAIKA
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your key in .env
streamlit run app.py
```

The app opens at http://localhost:8501.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | none (required) | Gemini API key |
| `LAIKA_CHAT_MODEL` | `gemini-flash-latest` | Model used for answers and summaries |
| `LAIKA_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model used for embeddings |
| `LAIKA_DATA_DIR` | `data` | Folder for saved conversations |

## Tests

The tests use fake models, so they need no API key or network access.

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

GitHub Actions runs the linter and the tests on every push and pull request.

## Deployment

The repository includes a `render.yaml` file for [Render](https://render.com). Create a new Blueprint from the repository and set `GOOGLE_API_KEY` when Render asks for it.

## Limitations

- Indexes are kept in memory for each browser session. They are rebuilt when the page is reloaded.
- Scanned PDFs without a text layer are not supported, because there is no OCR step.
- Saved conversations are stored in a local JSON file. On hosts with temporary disks, such as Render's free plan, they are lost on restart.
