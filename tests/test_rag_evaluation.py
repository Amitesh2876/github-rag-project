
TEST_QUESTIONS = [
    {
        "question": "Which function builds the repository index?",
        "expected_file": "src/pipeline.py",
        "expected_fact": "build_repository_index",
    },
    {
        "question": "Which function processes a user's question?",
        "expected_file": "src/pipeline.py",
        "expected_fact": "answer_question",
    },
    {
        "question": "What is the default EMBEDDING_MODEL value in src/config.py?",
        "expected_file": "src/config.py",
        "expected_fact": "BAAI/bge-small-en-v1.5",
    },
    {
        "question": "What is the default value of OLLAMA_MODEL in src/config.py?",
        "expected_file": "src/config.py",
        "expected_fact": "llama3.2",
    },
    {
        "question": "Which LlamaIndex class connects the application to Ollama?",
        "expected_file": "src/generation/llm.py",
        "expected_fact": "Ollama",
    },
    {
        "question": "How are documents split into chunks?",
        "expected_file": "src/retrieval/vector_store.py",
        "expected_fact": "SentenceSplitter",
    },
    {
        "question": "Which database stores document embeddings?",
        "expected_file": "src/retrieval/vector_store.py",
        "expected_fact": "ChromaDB",
    },
]



def precision_at_k(retrieved_files, expected_file, k):
    top_k = retrieved_files[:k]
    if not top_k or k <= 0:
        return 0.0

    relevant = sum(
        1 for file in top_k if file == expected_file
    )
    return relevant / k


def recall_at_k(retrieved_files, expected_file):
    return float(expected_file in retrieved_files)


def reciprocal_rank(retrieved_files, expected_file):
    for rank, file in enumerate(retrieved_files, start=1):
        if file == expected_file:
            return 1.0 / rank

    return 0.0


# Unit tests for evaluation metrics

def test_precision_at_k():
    files = ["src/pipeline.py", "README.md", "src/config.py"]
    assert precision_at_k(files, "src/pipeline.py", 3) == 1 / 3


def test_recall_at_k():
    files = ["README.md", "src/config.py"]
    assert recall_at_k(files, "src/config.py") == 1.0


def test_recall_when_file_missing():
    files = ["README.md", "app.py"]
    assert recall_at_k(files, "src/config.py") == 0.0


def test_reciprocal_rank():
    files = ["README.md", "src/config.py", "app.py"]
    assert reciprocal_rank(files, "src/config.py") == 0.5


def test_reciprocal_rank_when_missing():
    assert reciprocal_rank(["app.py"], "src/config.py") == 0.0
