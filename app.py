"""Streamlit interface for Laika. Run with: streamlit run app.py"""

import base64
import logging
from pathlib import Path

import streamlit as st

from laika import llm
from laika.answering import answer_question
from laika.documents import extract_text, split_into_chunks
from laika.retrieval import HybridIndex
from laika.storage import ConversationStore, to_csv
from laika.summarizer import combine, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STYLE = Path(__file__).parent / "assets" / "style.css"
store = ConversationStore()


@st.cache_data(show_spinner=False, max_entries=1000)
def embed(text: str) -> list[float]:
    return llm.get_embeddings().embed_query(text)


def init_state() -> None:
    defaults = {
        "page": "upload",
        "documents": [],  # list of {"name": str, "data": bytes, "text": str}
        "index": None,
        "conversation": [],
        "conversation_name": None,
        "summaries": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_to(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def logo() -> None:
    st.markdown(
        '<div class="logo"><h1>Laika</h1><p>by TAXMANN</p></div>',
        unsafe_allow_html=True,
    )


# ---------- Pages ----------


def upload_page() -> None:
    logo()
    files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if not st.button("Process documents", use_container_width=True, type="primary"):
        return
    if not files:
        st.warning("Please upload at least one PDF file.")
        return

    known = {doc["name"] for doc in st.session_state.documents}
    with st.spinner("Reading and indexing documents..."):
        try:
            for file in files:
                if file.name not in known:
                    st.session_state.documents.append(
                        {"name": file.name, "data": file.getvalue(), "text": extract_text(file)}
                    )
            full_text = "\n\n".join(doc["text"] for doc in st.session_state.documents)
            st.session_state.index = HybridIndex(split_into_chunks(full_text), llm.get_embeddings())
            st.session_state.summaries = None
        except Exception as error:
            logger.exception("Failed to process documents")
            st.error(f"Could not process the documents: {error}")
            return
    go_to("chat")


def chat_page() -> None:
    viewer, chat = st.columns(2)

    with viewer:
        documents = st.session_state.documents
        if documents:
            doc = st.selectbox("Document", documents, format_func=lambda d: d["name"], label_visibility="collapsed")
            encoded = base64.b64encode(doc["data"]).decode()
            st.markdown(
                f'<iframe class="pdf-viewer" src="data:application/pdf;base64,{encoded}"></iframe>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No documents uploaded. Laika will answer from the web.")

    with chat:
        if st.session_state.conversation_name:
            st.caption(st.session_state.conversation_name)
        for turn in st.session_state.conversation:
            st.chat_message("user").markdown(turn["question"])
            st.chat_message("assistant").markdown(turn["response"])

        question = st.chat_input("Message Laika")
        if not question:
            return
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                response = answer_question(
                    question,
                    index=st.session_state.index,
                    history=st.session_state.conversation,
                    ask=llm.ask,
                    embed=embed,
                    document_count=len(st.session_state.documents),
                )
            except Exception as error:
                logger.exception("Failed to answer question")
                response = f"Sorry, something went wrong while answering: {error}"
            st.markdown(response)
        st.session_state.conversation.append({"question": question, "response": response})


def summary_page() -> None:
    st.title("Document summaries")
    documents = st.session_state.documents
    if not documents:
        st.warning("Upload documents first.")
        return

    if st.session_state.summaries is None:
        progress = st.progress(0.0, text="Summarising...")
        try:
            summaries = {}
            for i, doc in enumerate(documents):
                progress.progress(i / len(documents), text=f"Summarising {doc['name']}")
                summaries[doc["name"]] = summarize(doc["text"], llm.ask)
            progress.progress(1.0, text="Writing the overall summary")
            overall = combine(list(summaries.values()), llm.ask)
            st.session_state.summaries = {"documents": summaries, "overall": overall}
        except Exception as error:
            logger.exception("Failed to summarise documents")
            st.error(f"Could not summarise the documents: {error}")
            return
        finally:
            progress.empty()

    summaries = st.session_state.summaries
    for name, summary in summaries["documents"].items():
        st.subheader(name)
        st.markdown(summary)
        st.divider()
    if len(summaries["documents"]) > 1:
        st.subheader("Overall summary")
        st.markdown(summaries["overall"])


def faq_page() -> None:
    st.markdown(
        """
### Frequently asked questions

**How do I upload PDFs?**
Click "Upload PDFs" in the sidebar, choose your files and press "Process documents".

**What can I ask?**
Anything about the content of your documents. If the answer is not in them, Laika searches the web.

**How accurate are the answers?**
Answers come from your documents or from web sources. Always check important details against official sources.

**Can I upload several PDFs?**
Yes. Laika searches across all of them.

**How do I start over?**
Use "New session" in the sidebar to remove all documents and the conversation.
"""
    )


def about_page() -> None:
    st.markdown(
        """
### About Laika

Laika is an AI assistant for reading tax and legal documents.

**What it does**
- Answers questions about uploaded PDF documents.
- Summarises each document and the whole set.
- Falls back to web search and Wikipedia when the documents do not have the answer.

**Limitations**
- Answers can be wrong. Check important details against official sources.
- Laika does not give personal legal or financial advice.
"""
    )


PAGES = {
    "upload": upload_page,
    "chat": chat_page,
    "summary": summary_page,
    "faq": faq_page,
    "about": about_page,
}


# ---------- Sidebar ----------


def sidebar() -> None:
    with st.sidebar:
        logo()
        if st.session_state.documents:
            st.markdown("**Documents**")
            for doc in st.session_state.documents:
                st.markdown(f"- {doc['name']}")
        st.divider()

        for label, page in [("Chat", "chat"), ("Upload PDFs", "upload"), ("Summarise documents", "summary")]:
            if st.button(label, use_container_width=True):
                go_to(page)

        conversation = st.session_state.conversation
        st.download_button(
            "Export conversation (CSV)",
            data=to_csv(conversation),
            file_name="laika_conversation.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not conversation,
        )
        if st.button("Save conversation", use_container_width=True, disabled=not conversation):
            st.session_state.conversation_name = store.save(conversation, st.session_state.conversation_name)
            st.toast(f"Saved as '{st.session_state.conversation_name}'")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.conversation_name = None
            st.rerun()
        if st.button("New session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        with st.expander("Saved conversations"):
            saved = store.load_all()
            if not saved:
                st.caption("None yet.")
            for name, turns in saved.items():
                if st.button(name, key=f"load-{name}", use_container_width=True):
                    st.session_state.conversation = list(turns)
                    st.session_state.conversation_name = name
                    go_to("chat")

        st.divider()
        if st.button("FAQ", use_container_width=True):
            go_to("faq")
        if st.button("About Laika", use_container_width=True):
            go_to("about")


def main() -> None:
    st.set_page_config(page_title="Laika", page_icon="📄", layout="wide")
    st.markdown(f"<style>{STYLE.read_text()}</style>", unsafe_allow_html=True)
    init_state()

    try:
        llm.require_api_key()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    sidebar()
    PAGES[st.session_state.page]()


if __name__ == "__main__":
    main()
