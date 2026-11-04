
# GitHub RAG Assistant

A Streamlit application that lets users explore public GitHub repositories using Retrieval-Augmented Generation (RAG).

This is an adapted learning project based on an existing GitHub-RAG implementation. It focuses on understanding and configuring repository ingestion, document indexing, semantic retrieval, and local LLM inference.

## How it works

1. Enter a public GitHub repository URL.
2. GitIngest extracts repository content and file structure.
3. LlamaIndex processes the content and builds a vector index.
4. Relevant repository context is retrieved based on the user's question.
5. Ollama generates an answer using the retrieved context.

## Technology stack

- Python
- Streamlit
- GitIngest
- LlamaIndex
- Ollama (`llama3.2`)
- Hugging Face sentence embeddings

## Run locally

### Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- Sufficient memory to run the selected local model

### 1. Clone the repository

```bash
git clone https://github.com/Amitesh2876/github-rag-project.git
cd github-rag-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements-local.txt
```

### 4. Download the model

```bash
ollama pull llama3.2
```

Ensure Ollama is running before starting the application.

### 5. Run the application

```bash
streamlit run app_local.py
```

Open the local URL displayed in the terminal, enter a public GitHub repository URL, and select **Load repository** to begin.

## Project structure

```text
github-rag-project/
├── app_local.py
├── app.py
├── requirements-local.txt
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`app_local.py` is the recommended entry point for local execution. `app.py` is the alternate API-oriented implementation retained from the starter project.

## Current limitations

- Supports public GitHub repositories only.
- Large repositories may require additional ingestion and indexing time.
- Generated answers may be incomplete or inaccurate and should be verified against the source code.
- The vector index is stored in memory and must be rebuilt when a new session starts.

## Future improvements

- Persistent vector storage.
- Better file filtering and code-aware chunking.
- Source references in generated answers.
- Support for private repositories.
- Automated testing and improved retrieval accuracy.

## Credits and attribution

This project adapts an existing GitHub-RAG tutorial/implementation and uses the following open-source technologies:

- [GitIngest](https://github.com/cyclotruc/gitingest)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Ollama](https://github.com/ollama/ollama)
- [Hugging Face](https://github.com/huggingface)

Please preserve applicable upstream licenses and copyright notices when modifying or redistributing source code.
