import streamlit as st

from src.config import APP_TITLE, DEFAULT_REPO_URL
from src.generation.llm import configure_models
from src.pipeline import answer_question, build_repository_index
from src.ingestion.github_loader import RepositoryLoadError, validate_github_url

st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
st.title("🔎 GitHub RAG Assistant")
st.caption("Ask questions about a public GitHub repository and inspect the source files behind each answer.")

@st.cache_resource(show_spinner="Loading local embedding and language models...")
def load_models():
    return configure_models()

st.session_state.setdefault("query_engine", None)
st.session_state.setdefault("repo_url", "")
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Repository")
    repo_url = st.text_input("Public GitHub URL", value=st.session_state.repo_url or DEFAULT_REPO_URL)
    if st.button("Index repository", type="primary", use_container_width=True):
        if not validate_github_url(repo_url):
            st.error("Use a URL like https://github.com/owner/repository.")
        else:
            try:
                load_models()
                with st.spinner("Cloning, parsing, chunking and indexing files..."):
                    engine, stats = build_repository_index(repo_url)
                st.session_state.query_engine = engine
                st.session_state.repo_url = repo_url.rstrip("/")
                st.session_state.messages = []
                st.success(f"Indexed {stats['files_indexed']} files and {stats['chunks_indexed']} chunks.")
                st.caption(f"Skipped {stats['files_skipped']} unsupported or oversized files.")
            except RepositoryLoadError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)
    if st.session_state.repo_url:
        st.caption(f"Loaded: {st.session_state.repo_url}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []

if st.session_state.query_engine is None:
    st.info("Enter a public GitHub repository URL in the sidebar and select Index repository.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved source files"):
                for source in msg["sources"]:
                    st.markdown(f"- [{source['file_path']}]({source['url']})")

question = st.chat_input("Ask about the repository...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = answer_question(st.session_state.query_engine, question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Retrieved source files", expanded=True):
                        for source in result["sources"]:
                            st.markdown(f"- [{source['file_path']}]({source['url']})")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception:
                st.error("Query failed. Check that Ollama is running and try again.")
                st.exception(Exception("See the application logs for details."))
