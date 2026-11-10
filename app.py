
import streamlit as st

from src.config import APP_TITLE, DEFAULT_REPO_URL
from src.generation.llm import configure_models
from src.pipeline import answer_question, build_repository_index
from src.ingestion.github_loader import (
    RepositoryLoadError,
    validate_github_url,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .status-label {
        color: #22C55E;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: #202127;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #34363F;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    .section-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading local AI models...")
def load_models():
    return configure_models()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "query_engine": None,
    "repo_url": "",
    "messages": [],
    "repo_stats": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🔎 GitHub RAG")
    st.caption("Repository Intelligence Assistant")

    st.divider()

    st.subheader("Repository")

    repo_url = st.text_input(
        "Public GitHub URL",
        value=st.session_state.repo_url or DEFAULT_REPO_URL,
        placeholder="https://github.com/owner/repository",
    )

    if st.button(
        "🚀 Index Repository",
        type="primary",
        use_container_width=True,
    ):

        if not validate_github_url(repo_url):
            st.error("Please enter a valid public GitHub URL.")

        else:
            try:
                load_models()

                with st.spinner(
                    "Cloning, parsing and indexing repository..."
                ):
                    engine, stats = build_repository_index(repo_url)

                st.session_state.query_engine = engine
                st.session_state.repo_url = repo_url.rstrip("/")
                st.session_state.repo_stats = stats
                st.session_state.messages = []

                st.success("Repository indexed successfully!")

            except RepositoryLoadError as exc:
                st.error(str(exc))

            except Exception as exc:
                st.error("Repository indexing failed.")
                st.exception(exc)

    # Repository status

    if st.session_state.query_engine is not None:

        st.divider()
        st.subheader("Index Status")

        st.markdown(
            '<p class="status-label">● Repository Ready</p>',
            unsafe_allow_html=True,
        )

        stats = st.session_state.repo_stats or {}

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Files",
                stats.get("files_indexed", 0),
            )

        with col2:
            st.metric(
                "Chunks",
                stats.get("chunks_indexed", 0),
            )

        st.caption(
            f"Skipped files: {stats.get('files_skipped', 0)}"
        )

        st.markdown(
            f"[View GitHub Repository]({st.session_state.repo_url})"
        )

    else:
        st.info("No repository indexed yet.")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with Python · LlamaIndex · ChromaDB · Ollama")


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.markdown(
    '<div class="main-title">GitHub RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Explore codebases using natural language and AI-powered retrieval.'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

if st.session_state.query_engine is None:

    st.info(
        "👈 Enter a public GitHub repository URL and click "
        "**Index Repository** to get started."
    )

    st.subheader("What can this assistant do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📂 Repository Ingestion")
        st.write(
            "Clone and index supported files from public GitHub repositories."
        )

    with col2:
        st.markdown("### 🧠 Semantic Search")
        st.write(
            "Find relevant code using embeddings and ChromaDB."
        )

    with col3:
        st.markdown("### 💬 AI Answers")
        st.write(
            "Ask questions and inspect the source files behind answers."
        )

    st.stop()


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg.get("latency_ms") is not None:
            st.caption(
                f"Response time: {msg['latency_ms'] / 1000:.2f} seconds"
            )

        if msg.get("sources"):

            with st.expander(
                f"📄 Retrieved Sources ({len(msg['sources'])})"
            ):

                for source in msg["sources"]:

                    file_path = source["file_path"]
                    url = source.get("url", "")
                    score = source.get("score")

                    if url:
                        st.markdown(f"📄 [{file_path}]({url})")
                    else:
                        st.write(f"📄 {file_path}")

                    if score is not None:
                        st.caption(
                            f"Retrieval score: {score:.4f}"
                        )


# --------------------------------------------------
# USER QUERY
# --------------------------------------------------

question = st.chat_input(
    "Ask anything about the indexed repository..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching source code and generating answer..."
        ):

            try:

                result = answer_question(
                    st.session_state.query_engine,
                    question,
                )

                answer = result["answer"]
                sources = result["sources"]
                latency_ms = result["latency_ms"]

                st.markdown(answer)

                st.caption(
                    f"⏱️ Generated in {latency_ms / 1000:.2f} seconds"
                )

                if sources:

                    with st.expander(
                        f"📄 Retrieved Sources ({len(sources)})",
                        expanded=True,
                    ):

                        for source in sources:

                            file_path = source["file_path"]
                            url = source.get("url", "")
                            score = source.get("score")

                            if url:
                                st.markdown(
                                    f"📄 [{file_path}]({url})"
                                )
                            else:
                                st.write(f"📄 {file_path}")

                            if score is not None:
                                st.caption(
                                    f"Retrieval score: {score:.4f}"
                                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency_ms": latency_ms,
                })

            except Exception as exc:

                st.error(
                    "Unable to generate an answer. "
                    "Check Ollama and the application logs."
                )

                st.exception(exc)
