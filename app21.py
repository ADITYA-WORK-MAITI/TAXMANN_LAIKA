# app21.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import traceback
from rank_bm25 import BM25Okapi
import numpy as np
import pickle
from langchain.schema import Document
from chatbot_prompts import get_conversational_chain
import csv
import io
from typing import List, Tuple, Dict
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

# Setting up logging configuration 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY is required.")

genai.configure(api_key=GOOGLE_API_KEY)

MAX_CHUNK_SIZE = 10000  # Adjust this value based on the API's limits

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Handle both string paths and file-like objects
        pdf_name = os.path.basename(pdf) if isinstance(pdf, str) else pdf.name
        logger.info(f"Processing document: {pdf_name}")
        try:
            # If pdf is a string (file path), open it; otherwise, use it directly
            pdf_file = open(pdf, 'rb') if isinstance(pdf, str) else pdf
            pdf_reader = PdfReader(pdf_file)
            for i, page in enumerate(pdf_reader.pages):
                logger.info(f"Extracting text from page {i+1}")
                text += page.extract_text()
            if isinstance(pdf, str):
                pdf_file.close()
        except Exception as e:
            logger.error(f"Error processing document {pdf_name}: {str(e)}")
            raise
    logger.info(f"Total extracted text length: {len(text)}")
    return text

@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error creating text chunks: {str(e)}")
        raise

def create_bm25_index(text_chunks):
    tokenized_corpus = [chunk.split() for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

@st.cache_data(show_spinner=False)
def bm25_search(bm25, query, top_k=5):
    tokenized_query = query.split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_doc_indices = np.argsort(doc_scores)[::-1][:top_k]
    return top_doc_indices, [doc_scores[i] for i in top_doc_indices]

@st.cache_resource(show_spinner=False)
def get_vector_store(text_chunks):
    try:
        logger.info(f"Creating vector store with {len(text_chunks)} chunks")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        
        # Create and save BM25 index
        bm25 = create_bm25_index(text_chunks)
        with open("bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
        
        logger.info("Vector store and BM25 index saved successfully")
    except Exception as e:
        logger.error(f"Error creating vector store and BM25 index: {str(e)}")
        raise

def improved_web_search(question):
    search = DuckDuckGoSearchRun()
    raw_results = search.run(question)
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    structured_response = llm.invoke(f"""
    Summarize and structure the following information about '{question}':
    {raw_results}
    
    Format your response using the following template:
    
    # {question}
    
    ## Key Points
    - Point 1
    - Point 2
    - Point 3
    
    ## Detailed Explanation
    [Provide a concise, professional explanation here]
    
    ## Sources
    - [List of sources, if available]
    
    Note: Ensure the response is factual, well-organized, and professional in tone.
    """)
    
    return structured_response.content

def enhanced_search(question, context_docs):
    # Step 1: Search in the PDF content
    pdf_answer = search_pdf_content(question, context_docs)
    if pdf_answer:
        return pdf_answer

    # Step 2: Improved web search
    web_answer = improved_web_search(question)
    if web_answer:
        return f"I couldn't find an answer in the uploaded documents, but here's what I found from reliable sources:\n\n{web_answer}"

    # Step 3: Wikipedia search (as a last resort)
    wikipedia_answer = wikipedia_search(question)
    if wikipedia_answer:
        return f"I couldn't find a specific answer, but here's some related information from Wikipedia:\n\n{wikipedia_answer}"

    return "I'm sorry, but I couldn't find a reliable answer to your question from any of the available sources."

def search_pdf_content(question, context_docs):
    # Use your existing method to search through the PDF content
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": context_docs, "question": question}, return_only_outputs=True)
    answer = response["output_text"]
    
    # Check if the answer is meaningful
    if answer.strip() and "I don't have enough information" not in answer.lower() and "cannot answer" not in answer.lower() and "does not mention" not in answer.lower():
        return answer
    return None

def wikipedia_search(question):
    try:
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        results = wikipedia_tool.run(question)
        return results
    except Exception as e:
        logger.error(f"Wikipedia search error: {str(e)}")
        return None

def generate_queries(original_query: str, num_queries: int = 3) -> List[str]:
    """Generate multiple queries based on the original query using Google's AI."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    prompt = f"Generate {num_queries} different ways to ask the following question:\n{original_query}\n\nQueries:"
    response = llm.invoke(prompt)
    generated_queries = response.content.strip().split('\n')
    return [original_query] + generated_queries[:num_queries]

def reciprocal_rank_fusion(results: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """Apply Reciprocal Rank Fusion to re-rank the results."""
    fused_scores = {}
    for query_results in results:
        for rank, (doc, score) in enumerate(query_results):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + k)
    
    fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results

def rag_fusion_search(user_question: str, vector_store, num_queries: int = 3, top_k: int = 5) -> List[Document]:
    """Perform RAG Fusion search."""
    queries = generate_queries(user_question, num_queries)
    all_results = []
    
    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        all_results.append([(doc.page_content, score) for doc, score in results])
    
    fused_results = reciprocal_rank_fusion(all_results)
    return [Document(page_content=doc) for doc, _ in fused_results[:top_k]]

@st.cache_data(show_spinner=False)
def compute_embedding(text: str) -> np.ndarray:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings.embed_query(text)

# Modify the user_input function to include context maintenance
def user_input(user_question):
    try:
        logger.info(f"Received question: {user_question}")

        # Handle metadata queries separately
        metadata_response = handle_metadata_query(user_question)
        if metadata_response:
            return metadata_response

        # Initialize embeddings and load the FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Compute embedding for the current question
        current_embedding = compute_embedding(user_question)

        # Find the most similar previous question
        max_similarity = -1
        most_similar_question = None
        most_similar_response = None

        for entry in st.session_state.conversation:
            prev_embedding = compute_embedding(entry['question'])
            similarity = cosine_similarity([current_embedding], [prev_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_question = entry['question']
                most_similar_response = entry['response']

        # If a similar question is found, include it in the context
        context = ""
        if max_similarity > 0.8:  # You can adjust this threshold
            context = f"Previously asked similar question: {most_similar_question}\nPrevious response: {most_similar_response}\n\n"

        # Use RAG Fusion for document retrieval
        context_docs = rag_fusion_search(user_question, new_db)

        # Perform enhanced search using a thread pool for timeout handling
        with ThreadPoolExecutor() as executor:
            future = executor.submit(enhanced_search, context + user_question, context_docs)
            try:
                answer = future.result(timeout=30)
            except TimeoutError:
                logger.error("API call timed out")
                return "I apologize, but the request timed out. Please try again or rephrase your question."

        # Post-process and format the answer
        answer = post_process_answer(answer)
        formatted_answer = format_response(answer)

        return formatted_answer

    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        logger.error(traceback.format_exc())
        return f"I apologize, but an error occurred while processing your request. Please try again later. Error: {str(e)}"

def format_response(answer: str) -> str:
    """Format the answer into clear, well-spaced sections with bolded titles."""
    sections = [section.strip() for section in answer.split('\n') if section.strip()]
    formatted_response = []

    for section in sections:
        if section.startswith("1."):
            section = section.replace("1.", "")
        if section.lower().startswith("key points"):
            formatted_response.append(f"**Key Points**\n")
        elif section.lower().startswith("detailed explanation"):
            formatted_response.append(f"**Detailed Explanation**\n")
        elif section.lower().startswith("sources"):
            formatted_response.append(f"**Sources**\n")
        else:
            formatted_response.append(section)

    return "\n\n".join(formatted_response)

def post_process_answer(answer: str) -> str:
    """Post-process the answer to remove unwanted characters and refine the content."""
    # Remove dashes and hashtags
    refined_answer = answer.replace('-', '').replace('#', '')
    
    # Ensure the answer does not end abruptly
    if not refined_answer.endswith('.'):
        refined_answer += '.'

    return refined_answer

def handle_metadata_query(question):
    question = question.lower()
    if "who are you" in question:
        return "I am Laika, an AI assistant created by TAXMANN to help with tax-related queries and document analysis."
    elif "what do you do" in question:
        return "I analyze PDF documents and answer questions based on their content. I specialize in tax-related information and can provide insights from the documents you upload."
    elif "who created you" in question:
        return "I was created by TAXMANN, a leading company in tax and accounting solutions."
    elif "how many pdfs" in question:
        return f"Currently, I have access to {len(st.session_state.pdf_docs)} uploaded PDF documents."
    elif "what is your name" in question:
        return "My name is Laika."
    elif "what can you do" in question:
        return "I can analyze PDF documents, answer questions based on their content, provide tax-related information, and assist with general queries about tax laws and regulations."
     elif "Who made you?" in question:
        return "I was made by Aditya Maiti, an intern at Taxmann Technologies."
     elif "What is Laika" in question:
        return "I am Laika. I can chat with multiple pdfs."
    return None

def export_conversation():
    if st.session_state.conversation:
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(['Question', 'Response'])
        for entry in st.session_state.conversation:
            csv_writer.writerow([entry['question'], entry['response']])
        
        st.download_button(
            label="Download Conversation as CSV",
            data=csv_buffer.getvalue(),
            file_name="conversation_export.csv",
            mime="text/csv"
        )
    else:
        st.warning("No conversation to export.")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(ResourceExhausted)
)
def generate_summary_with_retry(text):
    try:
        return generate_summary(text)
    except ResourceExhausted:
        logger.warning("API rate limit hit. Retrying...")
        raise
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise

def chunk_text(text, max_chunk_size):
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def summarize_documents():
    if st.session_state.pdf_docs:
        summaries = []
        overall_text = ""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            for i, pdf in enumerate(st.session_state.pdf_docs):
                status_text.text(f"Processing document {i+1} of {len(st.session_state.pdf_docs)}: {pdf.name}")
                
                # Extract text from PDF
                text = get_pdf_text([pdf])
                overall_text += text + "\n\n"
                
                # Chunk the text if it's too large
                chunks = chunk_text(text, MAX_CHUNK_SIZE)
                chunk_summaries = []
                
                for j, chunk in enumerate(chunks):
                    status_text.text(f"Summarizing chunk {j+1} of {len(chunks)} for document {i+1}")
                    chunk_summary = generate_summary_with_retry(chunk)
                    chunk_summaries.append(chunk_summary)
                    time.sleep(2)  # Add a delay between chunk processing
                
                # Combine chunk summaries
                full_summary = " ".join(chunk_summaries)
                summaries.append({"name": pdf.name, "summary": full_summary})
                
                progress = (i + 1) / len(st.session_state.pdf_docs)
                progress_bar.progress(progress)
                time.sleep(2)  # Add a delay between document processing
            
            status_text.text("Generating overall summary...")
            overall_chunks = chunk_text(overall_text, MAX_CHUNK_SIZE)
            overall_chunk_summaries = []
            
            for j, chunk in enumerate(overall_chunks):
                status_text.text(f"Summarizing overall chunk {j+1} of {len(overall_chunks)}")
                chunk_summary = generate_summary_with_retry(chunk)
                overall_chunk_summaries.append(chunk_summary)
                time.sleep(2)  # Add a delay between overall chunk processing
            
            overall_summary = " ".join(overall_chunk_summaries)
            
            # Store summaries in session state
            st.session_state.summaries = summaries
            st.session_state.overall_summary = overall_summary
            
            # Change page to 'summary'
            st.session_state.page = 'summary'
            st.rerun()
        except Exception as e:
            logger.error(f"Error in summarize_documents: {str(e)}")
            st.error(f"An error occurred while summarizing documents: {str(e)}. Please try again later.")
        finally:
            progress_bar.empty()
            status_text.empty()
    else:
        st.warning("No documents uploaded to summarize.")

def generate_summary(text):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        summary = llm.invoke(f"Please provide a concise summary of the following text, focusing on the main points and key information:\n\n{text}")
        return summary.content  # Extract the content from the AIMessage
    except ResourceExhausted as e:
        logger.warning(f"Resource exhausted error in generate_summary: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_summary: {str(e)}")
        raise

def summary_page():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("Document Summaries")
    
    for summary in st.session_state.summaries:
        st.subheader(f"Summary of {summary['name']}")
        st.write(summary['summary'])
        st.markdown("---")
    
    st.subheader("Overall Summary")
    st.write(st.session_state.overall_summary)
    
    # Modified button with a unique key
    if st.button("Back to Chat", use_container_width=True, key="summary_back_to_chat"):
        st.session_state.page = 'chat'
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def save_current_conversation():
    if st.session_state.conversation:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_name = f"Conversation_{timestamp}"
        st.session_state.saved_conversations[conversation_name] = st.session_state.conversation
        save_conversations_to_file()
        st.success(f"Conversation saved as '{conversation_name}'")

def save_conversations_to_file():
    with open("saved_conversations.json", "w") as f:
        json.dump(st.session_state.saved_conversations, f)

def load_conversations_from_file():
    if os.path.exists("saved_conversations.json"):
        with open("saved_conversations.json", "r") as f:
            return json.load(f)
    return {}

def load_saved_conversation(conversation_name):
    st.session_state.conversation = st.session_state.saved_conversations[conversation_name]
    st.session_state.current_conversation = conversation_name
    st.rerun()

def show_faq():
    st.markdown("""
    ### Frequently Asked Questions

    1. **How do I upload PDFs?**
       Click on the "Upload New PDFs" button in the sidebar and select your PDF files or enter URLs.

    2. **What types of questions can I ask?**
       You can ask questions related to the content of the uploaded PDFs or general tax-related queries.

    3. **How accurate are the responses?**
       Laika uses advanced AI to provide accurate information, but always verify important details with official sources.

    4. **Can I upload multiple PDFs?**
       Yes, you can upload multiple PDFs at once.

    5. **How do I start a new conversation?**
       Click the "Clear Conversation" button in the sidebar to start fresh.
    """)

def show_about_laika():
    st.markdown("""
    ### About Laika

    Laika is an AI-powered assistant created by TAXMANN to help with tax-related queries and document analysis. 

    **Capabilities:**
    - Analyze uploaded PDF documents and web content
    - Answer questions based on document content
    - Provide general tax-related information
    - Assist with understanding tax laws and regulations

    **Limitations:**
    - Laika's knowledge is based on its training data and may not include the very latest tax updates.
    - While Laika strives for accuracy, always verify critical information with official sources.
    - Laika cannot provide personalized legal or financial advice.

    For more information, please visit [TAXMANN's website](https://www.taxmann.com).
    """)

def upload_page():
    st.markdown('<div class="upload-page">', unsafe_allow_html=True)
    st.markdown("""
        <div class="logo-container">
            <h1 class="logo-text">Laika</h1>
            <p class="logo-subtext">by TAXMANN</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
           
    uploaded_files = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"], label_visibility="collapsed")
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            # Add new files to existing ones
            st.session_state.pdf_docs.extend(uploaded_files)
            with st.spinner("Processing documents..."):
                try:
                    raw_text = get_pdf_text(st.session_state.pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.page = 'chat'
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error processing documents: {str(e)}")
                    st.error("An error occurred while processing the documents. Please try again.")
        else:
            st.warning("Please upload PDF files before processing.")
    st.markdown('</div>', unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="logo-container">
            <h1 class="logo-text">Laika</h1>
            <p class="logo-subtext">by TAXMANN</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Upload New PDFs button
        if st.button("Upload New PDFs", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()
        
        # Display current PDFs
        if st.session_state.pdf_docs:
            st.markdown("### Uploaded PDFs")
            for pdf in st.session_state.pdf_docs:
                st.write(f"- {pdf.name}")
            st.markdown("---")
        
        # Main action buttons
        if st.button("Back to Chat", use_container_width=True):
            st.session_state.page = 'chat'
            st.rerun()
        
        if st.button("Summarize Documents", use_container_width=True):
            summarize_documents()
        
        if st.button("Export Conversation", use_container_width=True):
            export_conversation()
        
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.current_conversation = None
            st.rerun()
        
        if st.button("New Conversation", use_container_width=True):
            st.session_state.pdf_docs = []
            st.session_state.conversation = []
            st.session_state.current_conversation = None
            st.session_state.page = 'upload'
            st.success("Starting a new conversation. You can now upload new documents.")
            st.rerun()
        
        if st.button("Save Current Conversation", use_container_width=True):
            save_current_conversation()
        
        st.markdown("---")
        
        # Saved Conversations in an expander
        with st.expander("Saved Conversations"):
            for conv_name in st.session_state.saved_conversations.keys():
                if st.button(conv_name, key=f"load_{conv_name}", use_container_width=True):
                    load_saved_conversation(conv_name)
        
        st.markdown("---")
        
        # Help and About buttons
        if st.button("FAQ / Help", use_container_width=True):
            st.session_state.page = 'faq'
            st.rerun()
        
        if st.button("About Laika", use_container_width=True):
            st.session_state.page = 'about'
            st.rerun()
        
        

def chat_page():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.session_state.pdf_docs:
            selected_doc = st.selectbox("Select a document", options=st.session_state.pdf_docs, format_func=lambda x: x.name, label_visibility="collapsed")
            if selected_doc.name.startswith('http'):
                st.markdown(f"[View source]({selected_doc.name})")
            else:
                selected_doc.seek(0)
                base64_pdf = base64.b64encode(selected_doc.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" class="pdf-viewer" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.current_conversation:
            st.markdown(f"### Current Conversation: {st.session_state.current_conversation}")
        
        chat_container = st.container()
        with chat_container:
            for entry in st.session_state.conversation:
                with st.chat_message("user"):
                    st.write(entry['question'])
                with st.chat_message("assistant"):
                    st.write(entry['response'])
        
        user_question = st.chat_input("Message Laika")
        
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            
            with st.spinner("Laika is processing your request..."):
                try:
                    response = user_input(user_question)
                    with st.chat_message("assistant"):
                        st.write(response)
                    st.session_state.conversation.append({
                        'question': user_question,
                        'response': response
                    })
                    save_conversations_to_file()
                except Exception as e:
                    logger.error(f"Error processing chat input: {str(e)}")
                    st.error("An error occurred while processing your request. Please try again.")
            st.rerun()

def main():
    st.set_page_config(page_title="Laika by TAXMANN", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');

    body {
        background-color: #F5FFE0;
        color: #333333;
        font-family: 'Roboto', sans-serif;
    }

    .stApp {
        background-color: #F5FFE0;
    }

    .main-content, .chat-content {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border: 5px solid #32CD32;
        box-sizing: border-box;
        padding: 20px;
        overflow-y: auto;
        background: linear-gradient(135deg, #F5FFE0, #FAFFD1);
    }

    .stButton>button {
        color: #FFFFFF;
        background-color: #32CD32;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 16px;
        height: 40px;
        margin-top: 1px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #228B22;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(50, 205, 50, 0.3);
    }

    .stTextInput>div>div>input {
        color: #333333;
        background-color: rgba(255, 255, 255, 0.7);
        border: 2px solid #32CD32;
        border-radius: 25px;
        height: 40px;
        padding: 0 15px;
    }

    .stTextInput>div>div>input:focus {
        border-color: #228B22;
        box-shadow: 0 0 0 0.2rem rgba(50, 205, 50, 0.25);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #228B22;
        font-family: 'Playfair Display', serif;
    }

    .stAlert {
        background-color: rgba(50, 205, 50, 0.1);
        color: #333333;
        border: 1px solid #32CD32;
        border-radius: 15px;
    }

    .stSelectbox {
        border-radius: 25px;
    }

    .stSelectbox>div>div>div {
        background-color: #F5FFE0;
        color: #333333;
        border: 2px solid #32CD32;
    }

    [data-testid="stChatMessage"] {
        background-color: rgba(50, 205, 50, 0.1);
        border-radius: 20px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #32CD32;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        color: #333333;
    }

    .stChatInputContainer {
        padding-bottom: 20px;
    }

    .stChatInputContainer textarea {
        border-radius: 25px;
        background-color: rgba(255, 255, 255, 0.7);
        color: #333333;
        border: 2px solid #32CD32;
        padding: 10px 15px;
    }

    .upload-page {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
        max-width: 600px;
        margin: 0 auto;
    }

    .pdf-viewer {
        height: 100vh;
        width: 100%;
        border: none;
    }

    .stFileUploader {
        margin-bottom: 20px;
    }

    .sidebar .sidebar-content {
        background-color: #FAFFD1;
    }

    .logo-container {
        text-align: center;
        margin-bottom: 30px;
    }

    .logo-text {
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        color: #228B22;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }

    .logo-subtext {
        font-family: 'Roboto', sans-serif;
        font-size: 14px;
        color: #32CD32;
        margin-top: -10px;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    body::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(-45deg, #F5FFE0, #FAFFD1, #F0FFE0, #FFFDE7);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        z-index: -1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'upload'
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = []
    if 'saved_conversations' not in st.session_state:
        st.session_state.saved_conversations = load_conversations_from_file()
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    
    # Display sidebar
    sidebar()
    
    if st.session_state.page == 'upload':
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        upload_page()
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.page == 'chat':
        st.markdown('<div class="chat-content">', unsafe_allow_html=True)
        chat_page()
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.page == 'faq':
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        show_faq()
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.page == 'about':
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        show_about_laika()
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.page == 'summary':
        summary_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("An unexpected error occurred. Please try again later or contact support.")
