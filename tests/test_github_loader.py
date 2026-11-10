from pathlib import Path

from src.ingestion.github_loader import (
    parse_owner_repo, should_index_file, validate_github_url,
)


def test_valid_github_url():
    assert validate_github_url("https://github.com/openai/openai-python")


def test_git_suffix_is_valid():
    assert validate_github_url("https://github.com/openai/openai-python.git")


def test_invalid_github_url_is_rejected():
    assert not validate_github_url("https://github.com/")
    assert not validate_github_url("https://example.com/user/repo")


def test_parse_owner_repo():
    assert parse_owner_repo("https://github.com/openai/openai-python") == ("openai", "openai-python")


def test_supported_file_is_indexed():
    assert should_index_file(Path("src/main.py"))


def test_venv_and_lock_files_are_skipped():
    assert not should_index_file(Path(".venv/Lib/site-packages/demo.py"))
    assert not should_index_file(Path("package-lock.json"))
