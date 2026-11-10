
import time

from llama_index.core import PromptTemplate

from src.config import RETRIEVAL_TOP_K
from src.ingestion.github_loader import load_github_repository
from src.retrieval.vector_store import create_repository_index


def build_repository_index(repo_url: str):
    documents, stats = load_github_repository(repo_url)

    index, chunk_count = create_repository_index(
        documents,
        repository=stats["repository"],
        rebuild=True,
    )

    stats["chunks_indexed"] = chunk_count

    qa_prompt = PromptTemplate(
        """You are a technical assistant that explains GitHub repositories.

Use ONLY the provided source context.

Rules:
- Answer using exactly 5 concise bullet points when asked for an architecture overview.
- Put every bullet on a separate line.
- Use Markdown bullet syntax (- ).
- Mention actual files, functions, models and technologies.
- Do not invent features or claim scalability without evidence.
- If information is missing, explicitly say so.
- Avoid repeating information.

Context:
{context_str}

Question:
{query_str}

Answer:
"""
    )

    query_engine = index.as_query_engine(
        similarity_top_k=RETRIEVAL_TOP_K,
        text_qa_template=qa_prompt,
    )

    return query_engine, stats


def answer_question(query_engine, question: str):
    start_time = time.perf_counter()

    response = query_engine.query(question)

    latency_ms = (time.perf_counter() - start_time) * 1000

    sources, seen = [], set()

    for item in getattr(response, "source_nodes", []):
        metadata = item.node.metadata or {}
        file_path = metadata.get("file_path", "Unknown file")

        if file_path in seen:
            continue

        seen.add(file_path)

        score = getattr(item, "score", None)

        sources.append({
            "file_path": file_path,
            "url": metadata.get("source_url", ""),
            "score": round(float(score), 4)
            if score is not None else None,
        })

    return {
        "answer": str(response),
        "sources": sources,
        "latency_ms": round(latency_ms, 2),
    }
