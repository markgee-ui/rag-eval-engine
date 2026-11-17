"""Split markdown contracts into one chunk per numbered clause.

Contracts are already structured by clause ("## 6. Limitation of Liability"),
so the clause is the natural retrieval unit: it is self-contained, it is what a
lawyer would cite, and it gives every chunk a stable, human-readable ID
(e.g. "msa:6") that the golden eval set can reference.
"""

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc>:<clause>", e.g. "msa:6"
    doc_title: str  # "Master Services Agreement"
    heading: str  # "6. Limitation of Liability"
    text: str

    def render(self) -> str:
        return f"{self.doc_title} - {self.heading}\n{self.text}"


def chunk_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else path.stem
    matches = list(HEADING.finditer(raw))
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[m.end():end].strip()
        if body:
            chunks.append(Chunk(
                id=f"{path.stem}:{m.group(1)}",
                doc_title=doc_title,
                heading=f"{m.group(1)}. {m.group(2).strip()}",
                text=body,
            ))
    return chunks


def load_corpus(docs_dir: Path) -> list[Chunk]:
    chunks = [c for p in sorted(docs_dir.glob("*.md")) for c in chunk_document(p)]
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs - check for repeated clause numbers")
    return chunks
