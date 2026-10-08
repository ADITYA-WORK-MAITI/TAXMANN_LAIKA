"""Prompt templates sent to the chat model."""

NOT_FOUND = "NOT_FOUND"

DOCUMENT_QA = """You are Laika, an assistant that answers questions about the user's documents.
Answer using only the context below. Be accurate and concise, and use Markdown
where it helps (short paragraphs, bullet points, bold key terms).
If the context does not contain the answer, reply with exactly {not_found} and nothing else.

{history}Context:
{context}

Question: {question}

Answer:"""

WEB_ANSWER = """Answer the question below using the web search results provided.
Use Markdown with a short "Key points" list followed by a brief explanation.
End with a "Sources" list that contains the links you relied on.
Do not invent facts that are not in the results.

Question: {question}

Search results:
{results}

Answer:"""

SUMMARY = """Write a concise summary of the following text.
Focus on the main points and key figures. Use short paragraphs or bullet points.

Text:
{text}

Summary:"""

COMBINE_SUMMARIES = """Below are summaries of several parts of one or more documents.
Combine them into a single, coherent summary without repeating points.

{text}

Combined summary:"""
