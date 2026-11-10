import hashlib
import re

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMA_PERSIST_DIR, CHUNK_OVERLAP, CHUNK_SIZE


def collection_name(repository: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", repository.lower()).strip("_")[:40]
    digest = hashlib.sha1(repository.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def create_repository_index(documents, repository: str, rebuild: bool = True):
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    name = collection_name(repository)
    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    return index, collection.count()
