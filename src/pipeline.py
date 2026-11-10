from src.config import RETRIEVAL_TOP_K
from src.ingestion.github_loader import load_github_repository
from src.retrieval.vector_store import create_repository_index


def build_repository_index(repo_url: str):
    documents, stats = load_github_repository(repo_url)
    index, chunk_count = create_repository_index(
        documents, repository=stats["repository"], rebuild=True
    )
    stats["chunks_indexed"] = chunk_count
    return index.as_query_engine(similarity_top_k=RETRIEVAL_TOP_K), stats


def answer_question(query_engine, question: str):
    response = query_engine.query(question)
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
            "score": round(float(score), 4) if score is not None else None,
        })
    return {"answer": str(response), "sources": sources}
