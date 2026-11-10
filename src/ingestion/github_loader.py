"""Clone public GitHub repositories and retain file-level metadata."""

import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote
import re

from llama_index.core import Document

from src.config import MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES

SUPPORTED_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".html", ".css",
    ".scss", ".json", ".yaml", ".yml", ".toml", ".ini", ".md", ".rst",
    ".txt", ".sql", ".sh", ".ps1", ".xml", ".ipynb", ".vue", ".svelte",
}
IGNORED_DIRS = {
    ".git", ".venv", ".venv-rag", "venv", "__pycache__", "node_modules",
    "dist", "build", ".next", ".idea", ".vscode", "data", "site-packages",
}
IGNORED_NAMES = {".env", ".env.example", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}


class RepositoryLoadError(Exception):
    """Raised when a repository cannot be cloned or parsed."""


def validate_github_url(url: str) -> bool:
    pattern = r"https?://github\.com/[\w.-]+/[\w.-]+(?:\.git)?"
    return bool(
        re.fullmatch(
            pattern,
            url.strip().rstrip("/"),
            flags=re.IGNORECASE,
        )
    )


def parse_owner_repo(url: str):
    if not validate_github_url(url):
        raise RepositoryLoadError("Invalid URL. Expected https://github.com/owner/repository.")
    parts = url.strip().rstrip("/").removesuffix(".git").split("/")
    return parts[-2], parts[-1]


def should_index_file(path: Path) -> bool:
    if path.name.lower() in IGNORED_NAMES:
        return False
    if any(part.lower() in IGNORED_DIRS for part in path.parts):
        return False
    return path.suffix.lower() in SUPPORTED_SUFFIXES or path.name in {"Dockerfile", "Makefile"}


def _read_documents(repo_dir: Path, owner: str, repo: str, branch: str):
    documents, skipped, total_bytes = [], 0, 0
    candidates = [p for p in sorted(repo_dir.rglob("*"))
                  if p.is_file() and should_index_file(p.relative_to(repo_dir))]
    for path in candidates:
        if len(documents) >= MAX_FILES:
            skipped += 1
            continue
        size = path.stat().st_size
        if size == 0 or size > MAX_FILE_BYTES or total_bytes + size > MAX_TOTAL_BYTES:
            skipped += 1
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue
        if not content.strip():
            skipped += 1
            continue

        relative = path.relative_to(repo_dir).as_posix()
        source_url = (
            f"https://github.com/{owner}/{repo}/blob/{quote(branch, safe='')}/"
            f"{quote(relative, safe='/')}"
        )
        documents.append(Document(
            text=content,
            metadata={
                "file_path": relative,
                "repository": f"{owner}/{repo}",
                "source_url": source_url,
                "file_extension": path.suffix.lower() or path.name,
            },
        ))
        total_bytes += size
    skipped += max(0, len(candidates) - len(documents) - skipped)
    return documents, skipped


def load_github_repository(repo_url: str):
    owner, repo = parse_owner_repo(repo_url)
    with tempfile.TemporaryDirectory(prefix="github-rag-") as temp_dir:
        repo_dir = Path(temp_dir) / "repository"
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet",
                 f"https://github.com/{owner}/{repo}.git", str(repo_dir)],
                capture_output=True, text=True, timeout=180, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RepositoryLoadError("Repository cloning timed out after 180 seconds.") from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or "Git clone failed."
            raise RepositoryLoadError(
                f"Could not clone the repository. Confirm it is public and the URL is correct. {detail}"
            )

        branch_result = subprocess.run(
            ["git", "-C", str(repo_dir), "branch", "--show-current"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        branch = branch_result.stdout.strip() or "main"
        documents, skipped = _read_documents(repo_dir, owner, repo, branch)

    if not documents:
        raise RepositoryLoadError("No supported text or code files were found.")

    return documents, {
        "files_indexed": len(documents),
        "files_skipped": skipped,
        "repository": f"{owner}/{repo}",
        "branch": branch,
    }
