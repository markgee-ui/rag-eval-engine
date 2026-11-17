"""BM25 retriever in ~50 lines of plain Python (no vector DB, no embedding API).

Why lexical search for contracts: legal questions hinge on exact terms
("indemnification", "subprocessor", "99.9%"), the corpus is small, and BM25 is
deterministic, so retrieval evals run offline, for free, and give the same
result on every CI run. The weakness (synonyms: "raise prices" vs
"increase fees") is measured by the retrieval eval, not hidden.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from rag.chunker import Chunk

STOPWORDS = set("""
a an and are as at be by can do does for from has have how if in is it its
may of on or our than that the their then there these this to under us we
what when where which who will with within would you your
""".split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    # Crude plural folding ("breaches" -> "breach", "credits" -> "credit").
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("es") and t[-3] in "sxh":
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index heading + title too, so "liability cap" matches the clause name.
        self.docs = [Counter(tokenize(c.render())) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4, min_score: float = 1.0) -> list[Hit]:
        """Top-k chunks; chunks scoring below `min_score` are dropped so that an
        off-topic question gets no context instead of k irrelevant clauses."""
        q = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self.docs, self.lengths):
            s = 0.0
            for term in q:
                if term in tf:
                    f = tf[term]
                    s += self.idf[term] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if s >= min_score:
                scored.append(Hit(chunk, s))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
