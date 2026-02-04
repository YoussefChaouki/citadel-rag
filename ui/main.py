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

    /* Source cards */
    .source-card {
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .source-filename {
        font-weight: 600;
        color: #1E40AF;
    }
    .source-score {
        color: #059669;
        font-size: 0.85rem;
    }
    .source-preview {
        color: #4B5563;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    /* Mock mode warning */
    .mock-banner {
        background: linear-gradient(90deg, #FEF3C7, #FDE68A);
        border: 1px solid #F59E0B;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }

    /* Chat styling */
    .stChatMessage {
        padding: 1rem;
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 1rem;
    }

    /* Success/Error badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-success {
        background: #D1FAE5;
        color: #065F46;
    }
    .status-error {
        background: #FEE2E2;
        color: #991B1B;
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
    """Render the sidebar with file upload and status."""
    with st.sidebar:
        st.markdown(
            '<p class="sidebar-header">📁 Document Upload</p>', unsafe_allow_html=True
        )

        # API Status indicator
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Connected", icon="🟢")
        else:
            st.error("❌ API Unreachable", icon="🔴")
            st.caption(f"Endpoint: `{API_URL}`")
            return

        st.divider()

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload PDF or Markdown",
            type=["pdf", "md"],
            help="Supported formats: PDF (.pdf), Markdown (.md)",
        )

        if uploaded_file is not None:
            file_name = uploaded_file.name

            # Check if already ingested this session
            if file_name in st.session_state.ingested_files:
                st.info(f"📄 '{file_name}' already uploaded this session.")
            else:
                # Ingest button
                if st.button(
                    "🚀 Ingest Document", type="primary", use_container_width=True
                ):
                    with st.spinner(f"Ingesting '{file_name}'..."):
                        try:
                            result = ingest_file(file_name, uploaded_file.getvalue())

                            if result.get("status") == "duplicate":
                                st.warning(
                                    f"⚠️ Duplicate: {result.get('message', 'File already exists')}"
                                )
                            else:
                                st.success(f"✅ '{file_name}' accepted for processing!")
                                st.session_state.ingested_files.append(file_name)

                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 422:
                                detail = e.response.json().get("detail", "Invalid file")
                                st.error(f"❌ Validation Error: {detail}")
                            else:
                                st.error(f"❌ API Error: {e.response.status_code}")
                        except httpx.RequestError as e:
                            st.error(f"❌ Connection Error: {e}")

        st.divider()

        # Ingested files list
        if st.session_state.ingested_files:
            st.markdown("**📚 Uploaded this session:**")
            for fname in st.session_state.ingested_files:
                st.caption(f"• {fname}")

        # Settings
        st.divider()
        st.markdown("**⚙️ Settings**")
        st.session_state.num_sources = st.slider(
            "Context chunks (k)",
            min_value=1,
            max_value=15,
            value=5,
            help="Number of document chunks to retrieve for context",
        )


def render_sources(sources: list[SourceRef]) -> None:
    """Render source citations in an expander."""
    if not sources:
        return

    with st.expander(f"📖 View Sources ({len(sources)} chunks)", expanded=False):
        for _i, source in enumerate(sources, 1):
            score_pct = source["score"] * 100
            st.markdown(
                f"""
                <div class="source-card">
                    <span class="source-filename">📄 {source["filename"]}</span>
                    <span class="source-score"> — Chunk #{source["chunk_index"]} • {score_pct:.1f}% relevance</span>
                    <p class="source-preview">"{source["preview"]}"</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_history() -> None:
    """Render the chat message history."""
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
