
from src.config import DEFAULT_REPO_URL, RETRIEVAL_TOP_K
from src.generation.llm import configure_models
from src.ingestion.github_loader import load_github_repository
from src.retrieval.vector_store import create_repository_index

from tests.test_rag_evaluation import (
    TEST_QUESTIONS,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def main():
    print("Initializing RAG retrieval benchmark...")

    configure_models()

    documents, stats = load_github_repository(DEFAULT_REPO_URL)
    print(f"Loaded {stats['files_indexed']} files.")

    index, chunks = create_repository_index(
        documents,
        repository=stats["repository"],
        rebuild=True,
    )

    print(f"Indexed {chunks} chunks.")

    # IMPORTANT: Initialize retriever before the evaluation loop.
    retriever = index.as_retriever(
        similarity_top_k=RETRIEVAL_TOP_K
    )

    precision_scores = []
    recall_scores = []
    mrr_scores = []

    print("\nRunning retrieval evaluation...\n")

    for test in TEST_QUESTIONS:
        results = retriever.retrieve(test["question"])

        retrieved_files = []

        for rank, node in enumerate(results, start=1):
            file_path = node.node.metadata.get("file_path", "")
            retrieved_files.append(file_path)

            print(
                f"Rank {rank}: {file_path} "
                f"| Score: {node.score}"
            )

        expected = test["expected_file"]

        p = precision_at_k(
            retrieved_files, expected, RETRIEVAL_TOP_K
        )
        r = recall_at_k(retrieved_files, expected)
        m = reciprocal_rank(retrieved_files, expected)

        precision_scores.append(p)
        recall_scores.append(r)
        mrr_scores.append(m)

        print(f"Question: {test['question']}")
        print(f"Expected file: {expected}")
        print(f"Retrieved files: {retrieved_files}")
        print(f"Precision@K: {p:.3f}")
        print(f"Recall@K: {r:.3f}")
        print(f"Reciprocal Rank: {m:.3f}")
        print("-" * 60)

    total = len(TEST_QUESTIONS)

    print("\n========== BENCHMARK RESULTS ==========")
    print(
        f"Precision@{RETRIEVAL_TOP_K}: "
        f"{sum(precision_scores) / total:.3f}"
    )
    print(
        f"Recall@{RETRIEVAL_TOP_K}: "
        f"{sum(recall_scores) / total:.3f}"
    )
    print(f"MRR: {sum(mrr_scores) / total:.3f}")
    print("=======================================")


if __name__ == "__main__":
    main()
