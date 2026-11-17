"""Offline tests: chunking, retrieval and metrics. No API key needed."""

import json
from pathlib import Path

from evals.retrieval_metrics import hit_at_k, recall_at_k, reciprocal_rank
from rag.chunker import load_corpus
from rag.pipeline import DOCS_DIR, get_retriever
from rag.retriever import tokenize

ROOT = Path(__file__).resolve().parent.parent


def test_every_clause_becomes_a_chunk_with_stable_id():
    ids = {c.id for c in load_corpus(DOCS_DIR)}
    assert {"msa:6", "dpa:4", "sla:2", "aup:3"} <= ids
    assert len(ids) == 25


def test_golden_set_only_references_real_chunks():
    ids = {c.id for c in load_corpus(DOCS_DIR)}
    for line in (ROOT / "data" / "golden.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row["gold_chunks"]) <= ids, row["id"]


def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("The breaches and credits") == ["breach", "credit"]


def test_retriever_ranks_exact_clause_first():
    hits = get_retriever().search("personal data breach notification", k=3)
    assert hits[0].chunk.id == "dpa:4"


def test_off_topic_query_retrieves_nothing():
    assert get_retriever().search("recipe for banana bread") == []


def test_metrics():
    assert hit_at_k(["a", "b"], ["b"]) == 1.0
    assert recall_at_k(["a", "b"], ["b", "c"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["z"]) == 0.0
