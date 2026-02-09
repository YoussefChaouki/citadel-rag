"""
CITADEL RAG Frontend

Streamlit-based user interface for the CITADEL retrieval-augmented
generation system. Provides document upload and conversational Q&A.

Run locally:
    streamlit run ui/main.py

Run in Docker:
    docker compose up rag-ui
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8001")
RAG_ENDPOINT = f"{API_URL}/api/v1/rag"

# Timeouts: ingestion can be slow (PDF parsing + embedding)
INGEST_TIMEOUT = 60.0
ASK_TIMEOUT = 45.0


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------


class SourceRef(TypedDict):
    """Source reference from RAG response."""

    filename: str
    chunk_index: int
    score: float
    preview: str


class ChatMessage(TypedDict):
    """Chat message structure for session state."""

    role: str  # "user" | "assistant"
    content: str
    sources: list[SourceRef] | None
    is_mocked: bool


class DocumentRecord(TypedDict):
    """Document record from list_documents endpoint."""

    document_id: str
    filename: str
    chunks_count: int
    created_at: str


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CITADEL — RAG Assistant",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }

    /* Source cards with improved spacing */
    .source-card {
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        line-height: 1.6;
    }
    .source-filename {
        font-weight: 600;
        color: #1E40AF;
        font-size: 0.95rem;
    }
    .source-meta {
        display: inline-block;
        margin-left: 0.5rem;
    }
    .source-score-container {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    .source-score {
        color: #059669;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .source-score-help {
        cursor: help;
        color: #0891B2;
        font-weight: bold;
    }
    .source-preview {
        color: #4B5563;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-style: italic;
    }

    /* Mock mode warning */
    .mock-banner {
        background: linear-gradient(90deg, #FEF3C7, #FDE68A);
        border: 1px solid #F59E0B;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }

    /* Chat styling with better padding */
    .stChatMessage {
        padding: 1.5rem 1rem;
        margin: 0.5rem 0;
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    .document-item {
        background: #F3F4F6;
        border-left: 2px solid #6366F1;
        padding: 0.75rem 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0 0.25rem 0.25rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
    }

    .document-name {
        flex: 1;
        color: #1F2937;
        font-weight: 500;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .document-chunks {
        color: #9CA3AF;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }

    /* Status badge styling (subtle) */
    .status-subtle {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.9rem;
        color: #059669;
    }

    /* Button styling */
    .delete-button {
        color: #DC2626;
        cursor: pointer;
        font-weight: bold;
        padding: 0.25rem 0.5rem;
    }

    /* Improved divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #E5E7EB, transparent);
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[ChatMessage]
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []  # list[str]
    if "num_sources" not in st.session_state:
        st.session_state.num_sources = 5
    if "refresh_documents" not in st.session_state:
        st.session_state.refresh_documents = True


init_session_state()


# ---------------------------------------------------------------------------
# API Client Functions
# ---------------------------------------------------------------------------


def ingest_file(file_name: str, file_bytes: bytes) -> dict[str, Any]:
    """
    Upload a file to the RAG ingestion endpoint.

    Args:
        file_name: Original filename.
        file_bytes: Raw file content.

    Returns:
        API response as dict.

    Raises:
        httpx.HTTPError: On network or API errors.
    """
    with httpx.Client(timeout=INGEST_TIMEOUT) as client:
        response = client.post(
            f"{RAG_ENDPOINT}/ingest",
            files={"file": (file_name, file_bytes)},
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


def ask_question(query: str, k: int = 5) -> dict[str, Any]:
    """
    Send a question to the RAG /ask endpoint.

    Args:
        query: User's question.
        k: Number of context chunks to retrieve.

    Returns:
        API response with answer, sources, and mock status.

    Raises:
        httpx.HTTPError: On network or API errors.
    """
    with httpx.Client(timeout=ASK_TIMEOUT) as client:
        response = client.post(
            f"{RAG_ENDPOINT}/ask",
            json={"query": query, "k": k},
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


def get_documents() -> list[DocumentRecord]:
    """
    Retrieve list of all documents in the system.

    Returns:
        List of DocumentRecord dicts.

    Raises:
        httpx.HTTPError: On network or API errors.
    """
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"{RAG_ENDPOINT}/documents")
        response.raise_for_status()
        result: list[DocumentRecord] = response.json()
        return result


def delete_document(filename: str) -> dict[str, Any]:
    """
    Delete a document from the RAG system.

    Args:
        filename: Filename to delete.

    Returns:
        API response with deletion details.

    Raises:
        httpx.HTTPError: On network or API errors.
    """
    with httpx.Client(timeout=10.0) as client:
        response = client.delete(f"{RAG_ENDPOINT}/documents/{filename}")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


def check_api_health() -> bool:
    """Check if the RAG API is reachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_URL}/health")
            return response.status_code == 200
    except httpx.RequestError:
        return False


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the sidebar with file upload, documents list, and settings."""
    with st.sidebar:
        # --- API Status (subtle) ---
        api_healthy = check_api_health()
        if api_healthy:
            st.markdown(
                '<p style="color: #059669; font-size: 0.9rem;">✓ API Connected</p>',
                unsafe_allow_html=True,
            )
        else:
            st.error("❌ API Unreachable", icon="🔴")
            st.caption(f"Endpoint: `{API_URL}`")
            return

        # --- Upload Section ---
        st.markdown(
            '<p class="sidebar-header">📁 Upload Documents</p>', unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a PDF or Markdown file",
            type=["pdf", "md"],
            help="Supported formats: PDF (.pdf), Markdown (.md)",
            key="file_uploader",
        )

        if uploaded_file is not None:
            file_name = uploaded_file.name

            if st.button(
                "🚀 Ingest Document", type="primary", use_container_width=True
            ):
                with st.spinner(f"Processing '{file_name}'..."):
                    try:
                        result = ingest_file(file_name, uploaded_file.getvalue())

                        if result.get("status") == "duplicate":
                            st.warning(
                                f"⚠️ Duplicate: {result.get('message', 'File already exists')}"
                            )
                        else:
                            st.success(f"✅ '{file_name}' queued for processing!")
                            st.session_state.refresh_documents = True

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 422:
                            detail = e.response.json().get("detail", "Invalid file")
                            st.error(f"Validation Error: {detail}")
                        else:
                            st.error(f"API Error: {e.response.status_code}")
                    except httpx.RequestError as e:
                        st.error(f"Connection Error: {e}")

        # --- Documents List ---
        st.markdown(
            '<p class="sidebar-header">📚 Ingested Documents</p>',
            unsafe_allow_html=True,
        )

        try:
            documents = get_documents()

            if documents:
                for doc in documents:
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(
                            f"""
                            <div class="document-item">
                                <span class="document-name" title="{doc["filename"]}">{doc["filename"]}</span>
                                <span class="document-chunks">{doc["chunks_count"]} chunks</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        if st.button(
                            "🗑️",
                            key=f"delete_{doc['document_id']}",
                            help=f"Delete {doc['filename']}",
                            use_container_width=True,
                        ):
                            try:
                                delete_document(doc["filename"])
                                st.success(f"Deleted '{doc['filename']}'")
                                st.session_state.refresh_documents = True
                                st.rerun()
                            except httpx.HTTPStatusError as e:
                                if e.response.status_code == 404:
                                    st.error("Document not found")
                                else:
                                    st.error(f"Delete failed: {e.response.status_code}")
                            except httpx.RequestError as e:
                                st.error(f"Connection error: {e}")
            else:
                st.info(
                    "No documents uploaded yet. Start by uploading a PDF or Markdown file."
                )

        except httpx.RequestError as e:
            st.warning(f"Could not load documents: {e}")

        # --- Settings ---
        st.markdown('<p class="sidebar-header">⚙️ Settings</p>', unsafe_allow_html=True)

        st.session_state.num_sources = st.slider(
            "Context chunks (k)",
            min_value=1,
            max_value=15,
            value=st.session_state.num_sources,
            help="Number of document chunks to retrieve for context",
        )

        # --- Clear Conversation ---
        if st.button("🔄 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.success("Conversation cleared!")
            st.rerun()


def render_sources(sources: list[SourceRef]) -> None:
    """Render source citations with tooltips for explainability."""
    if not sources:
        return

    with st.expander(f"📖 View Sources ({len(sources)} chunks)", expanded=False):
        for _i, source in enumerate(sources, 1):
            score_pct = source["score"] * 100

            # Create the HTML with tooltip
            st.markdown(
                f"""
                <div class="source-card">
                    <span class="source-filename">📄 {source["filename"]}</span>
                    <span class="source-meta">
                        Chunk #{source["chunk_index"]}
                        <span class="source-score-container">
                            • <span class="source-score">{score_pct:.1f}%</span>
                            <span class="source-score-help" title="Relevance Score: Cosine similarity (0-100%). Indicates how mathematically close the document's meaning is to your question.">ℹ️</span>
                        </span class="source-score-container">
                    </span>
                    <p class="source-preview">"{source["preview"]}"</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_history() -> None:
    """Render the chat message history with improved spacing."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Show mock mode warning for assistant messages
            if message["role"] == "assistant" and message.get("is_mocked"):
                st.warning(
                    "⚠️ **Mock Mode** — AI Engine unavailable. Showing simulated response."
                )

            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])


def render_main_chat() -> None:
    """Render the main chat interface."""
    # Header
    st.markdown('<p class="main-title">🏰 CITADEL</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Ask questions about your documents using AI-powered semantic search</p>',
        unsafe_allow_html=True,
    )

    # Check API health
    if not check_api_health():
        st.error(
            "**Cannot connect to CITADEL API**\n\n"
            f"The backend at `{API_URL}` is not responding. "
            "Please ensure the Docker stack is running:\n"
            "```bash\nmake up\n```"
        )
        return

    # Chat history
    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        user_message: ChatMessage = {
            "role": "user",
            "content": prompt,
            "sources": None,
            "is_mocked": False,
        }
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    k = st.session_state.get("num_sources", 5)
                    result = ask_question(prompt, k=k)

                    answer = result.get("answer", "No response received.")
                    sources = result.get("sources", [])
                    is_mocked = result.get("is_mocked", False)

                    # Show mock warning if applicable
                    if is_mocked:
                        st.warning(
                            "⚠️ **Mock Mode** — AI Engine unavailable. Showing simulated response."
                        )

                    # Display answer
                    st.markdown(answer)

                    # Display sources
                    render_sources(sources)

                    # Add to history
                    assistant_message: ChatMessage = {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "is_mocked": is_mocked,
                    }
                    st.session_state.messages.append(assistant_message)

                except httpx.HTTPStatusError as e:
                    error_msg = f"API returned error: {e.response.status_code}"
                    st.error(f"❌ {error_msg}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "sources": None,
                            "is_mocked": False,
                        }
                    )
                except httpx.RequestError as e:
                    error_msg = f"Connection failed: {e}"
                    st.error(f"❌ {error_msg}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "sources": None,
                            "is_mocked": False,
                        }
                    )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main application entry point."""
    render_sidebar()
    render_main_chat()


if __name__ == "__main__":
    main()
