import logging
import re
import uuid

import streamlit as st
from gitingest import ingest
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_TITLE = "GitHub RAG Assistant"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"
MAX_REPO_CHARS = 2_000_000


class RepositoryError(Exception):
    """Raised when a repository cannot be loaded or indexed."""


def validate_github_url(url: str) -> bool:
    """Accept standard public GitHub repository URLs only."""
    pattern = r"^https?://github\.com/[\w.-]+/[\w.-]+/?$"
    return bool(re.match(pattern, url.strip(), flags=re.IGNORECASE))


def get_repo_name(url: str) -> str:
    return url.rstrip("/").split("/")[-1].removesuffix(".git")


@st.cache_resource(show_spinner="Loading local language and embedding models...")
def configure_models():
    """Load reusable local models once per Streamlit server process."""
    Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=180.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    return True


def build_query_engine(github_url: str):
    """Ingest a public GitHub repository and build an in-memory vector index."""
    try:
        summary, tree, content = ingest(github_url)
        if not content or not content.strip():
            raise RepositoryError("GitIngest returned no repository content.")

        if len(content) > MAX_REPO_CHARS:
            content = content[:MAX_REPO_CHARS]
            st.warning("The repository text was large, so indexing was limited to the first 2 million characters.")

        repo_document = Document(
            text=f"Repository summary:\n{summary}\n\nRepository file tree:\n{tree}\n\nRepository content:\n{content}",
            metadata={"source": github_url, "repository": get_repo_name(github_url)},
        )
        parser = MarkdownNodeParser()
        index = VectorStoreIndex.from_documents(
            [repo_document],
            transformations=[parser],
            show_progress=True,
        )
        return index.as_query_engine(
            streaming=True,
            similarity_top_k=5,
        )
    except RepositoryError:
        raise
    except Exception as exc:
        logger.exception("Repository ingestion/indexing failed")
        raise RepositoryError(
            "Could not load or index this repository. Check that it is public, "
            "the URL is correct, and your local Ollama service is running."
        ) from exc


def initialise_state():
    defaults = {
        "messages": [],
        "query_engine": None,
        "loaded_url": "",
        "session_id": str(uuid.uuid4()),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_chat():
    st.session_state.messages = []


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
    initialise_state()
    st.title("🔎 GitHub RAG Assistant")
    st.caption("Ask questions about a public GitHub repository using retrieval-augmented generation.")

    with st.sidebar:
        st.header("Repository")
        github_url = st.text_input(
            "Public GitHub repository URL",
            placeholder="https://github.com/owner/repository",
        )
        load_clicked = st.button("Load repository", type="primary", use_container_width=True)

        if load_clicked:
            cleaned_url = github_url.strip()
            if not validate_github_url(cleaned_url):
                st.error("Enter a valid repository URL, such as https://github.com/owner/repository.")
            else:
                try:
                    configure_models()
                    with st.spinner(f"Ingesting {get_repo_name(cleaned_url)} and building its search index..."):
                        engine = build_query_engine(cleaned_url)
                    st.session_state.query_engine = engine
                    st.session_state.loaded_url = cleaned_url
                    st.session_state.messages = []
                    st.success(f"Loaded {get_repo_name(cleaned_url)}.")
                except RepositoryError as exc:
                    st.session_state.query_engine = None
                    st.session_state.loaded_url = ""
                    st.error(str(exc))

        if st.session_state.loaded_url:
            st.caption(f"Loaded: {st.session_state.loaded_url}")
        st.divider()
        st.caption(f"LLM: {OLLAMA_MODEL}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")
        st.button("Clear chat", on_click=clear_chat, use_container_width=True)

    if st.session_state.query_engine is None:
        st.info("Add a public GitHub repository in the sidebar and select **Load repository** to begin.")
        st.markdown(
            "The app ingests repository text, creates a vector index, and retrieves relevant context "
            "to answer your questions."
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something about the repository...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                response = st.session_state.query_engine.query(prompt)
                if hasattr(response, "response_gen"):
                    for chunk in response.response_gen:
                        full_response += str(chunk)
                        response_placeholder.markdown(full_response + "▌")
                else:
                    full_response = str(response)
                response_placeholder.markdown(full_response)
            except Exception:
                logger.exception("Query failed")
                full_response = "I couldn't generate an answer. Check that Ollama is running and try again."
                response_placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
