# GitHub RAG Assistant

An AI-powered assistant that indexes public GitHub repositories and answers questions using retrieval-augmented generation.

## Architecture

- **Ingestion:** shallow Git clone of a public repository.
- **File-aware documents:** each supported source file retains its path and GitHub URL metadata.
- **Chunking:** LlamaIndex `SentenceSplitter`.
- **Embeddings:** `BAAI/bge-small-en-v1.5`.
- **Vector store:** persistent ChromaDB.
- **Generation:** local Ollama model (`llama3.2` by default).
- **Interface:** Streamlit.
- **Tests:** pytest.

## Requirements

Python 3.11, Git, Ollama, and a local Ollama model such as `llama3.2`.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv-rag
.\.venv-rag\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull llama3.2
```

## Run

```powershell
streamlit run app.py
```

Enter a public repository URL and select **Index repository**.

## Tests

```powershell
python -m pytest -q
```

## Initial ingestion limits

The loader indexes up to 500 supported text files, with a 256 KB per-file limit and a 20 MB total text budget. It skips common generated, environment, and dependency directories. Limits are defined in `src/config.py`.

## Roadmap

- Labelled retrieval evaluation dataset and Precision@K / Recall@K.
- Answer faithfulness and citation correctness evaluation.
- Caching and incremental re-indexing.
- Private repository authentication.
- More code-aware chunking and repository filters.

## Limitations

This is an initial local prototype. It supports public repositories, requires a working local Ollama service, and does not yet perform incremental indexing. Do not claim benchmark performance until a repeatable evaluation has been run.
