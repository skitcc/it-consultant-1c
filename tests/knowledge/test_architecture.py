from __future__ import annotations

import ast
from pathlib import Path

import pytest

from knowledge.adapters.outbound.ollama_reranker import (
    parse_relevance_score,
    score_from_logprobs,
)

ROOT = Path(__file__).parents[2]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_core_has_no_infrastructure_imports():
    forbidden = {"fastapi", "open_webui", "ews", "qdrant_client", "docling", "httpx", "sqlite3"}
    violations = {
        path.relative_to(ROOT).as_posix(): sorted(imported_roots(path) & forbidden)
        for path in (ROOT / "knowledge" / "core").rglob("*.py")
        if imported_roots(path) & forbidden
    }
    assert violations == {}


def test_new_package_does_not_depend_on_legacy_packages():
    forbidden = {"mail_gateway", "reindex"}
    violations = {
        path.relative_to(ROOT).as_posix(): sorted(imported_roots(path) & forbidden)
        for path in (ROOT / "knowledge").rglob("*.py")
        if imported_roots(path) & forbidden
    }
    assert violations == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", 1.0),
        ("<think>reasoning</think>\nнет", 0.0),
        ('{"score": 0.75}', 0.75),
    ],
)
def test_qwen_text_scoring_helpers_are_preserved(text: str, expected: float):
    assert parse_relevance_score(text) == expected


def test_qwen_logprob_helper_softmaxes_yes_and_no():
    score = score_from_logprobs(
        {
            "logprobs": [
                {
                    "token": "yes",
                    "logprob": -0.1,
                    "top_logprobs": [{"token": "no", "logprob": -2.1}],
                }
            ]
        }
    )
    assert score == pytest.approx(0.880797, rel=1e-5)
