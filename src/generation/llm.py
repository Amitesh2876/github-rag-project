
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from src.config import (
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_CONTEXT_WINDOW,
)


def configure_models():
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
        context_window=OLLAMA_CONTEXT_WINDOW,
        temperature=0.1,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    return {
        "llm": OLLAMA_MODEL,
        "embedding": EMBEDDING_MODEL,
        "context_window": OLLAMA_CONTEXT_WINDOW,
    }
