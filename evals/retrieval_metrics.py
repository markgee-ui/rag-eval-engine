"""Deterministic retrieval metrics against hand-labelled gold clause IDs."""


def hit_at_k(retrieved: list[str], gold: list[str]) -> float:
    """1.0 if any gold clause was retrieved at all."""
    return float(any(g in retrieved for g in gold))


def recall_at_k(retrieved: list[str], gold: list[str]) -> float:
    """Fraction of gold clauses retrieved (matters for multi-clause questions)."""
    return sum(g in retrieved for g in gold) / len(gold)


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    """1/rank of the first gold clause (1.0 = ranked first, 0 = missing)."""
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0
