"""Ask the contracts a question from the command line.

    python ask.py "How quickly must a data breach be reported?"
    python ask.py            # interactive mode
"""

import sys

from rich.console import Console
from rich.panel import Panel

from rag.llm import LLMError
from rag.pipeline import answer

console = Console()


def ask(question: str) -> None:
    try:
        result = answer(question)
    except LLMError as e:
        console.print(f"[red]{e}[/]")
        return
    style = "green" if result.answerable else "yellow"
    sources = ", ".join(result.citations) or "none"
    console.print(Panel(result.answer, title="Answer", border_style=style,
                        subtitle=f"cited: {sources}"))
    for h in result.hits:
        mark = "[bold]*[/]" if h.chunk.id in result.citations else " "
        console.print(f" {mark} {h.chunk.id:<6} {h.score:5.1f}  {h.chunk.doc_title} - {h.chunk.heading}")
    if result.invalid_citations:
        console.print(f"[red]Dropped citations to clauses that were never retrieved: "
                      f"{result.invalid_citations}[/]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        while (q := console.input("[bold]Question[/] (blank to quit): ").strip()):
            ask(q)
