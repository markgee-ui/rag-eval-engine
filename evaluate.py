"""Run the eval suite: retrieval metrics (offline) + LLM-judged generation metrics.

    python evaluate.py --retrieval-only     # free, no API key, deterministic
    python evaluate.py                      # full run (calls Claude)
    python evaluate.py --svg docs/eval.svg  # also save a screenshot of the report

Exits non-zero when any metric misses its threshold, so it can gate CI.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.judge import judge
from evals.retrieval_metrics import hit_at_k, recall_at_k, reciprocal_rank
from rag.pipeline import answer, format_context, get_retriever

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "data" / "golden.jsonl"
REPORT = ROOT / "reports" / "latest.json"

# metric -> (threshold, "min" if higher is better / "max" if lower is better)
THRESHOLDS = {
    "hit@k": (0.85, "min"),
    "mrr": (0.75, "min"),
    "answer_correctness": (0.80, "min"),
    "faithfulness": (0.90, "min"),
    "hallucination_rate": (0.10, "max"),
    "abstention_accuracy": (0.75, "min"),
}


def load_golden(limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def eval_retrieval(row: dict, k: int) -> dict:
    retrieved = [h.chunk.id for h in get_retriever().search(row["question"], k=k)]
    rec = {"id": row["id"], "question": row["question"], "retrieved": retrieved,
           "answerable_gold": bool(row["gold_chunks"])}
    if rec["answerable_gold"]:
        gold = row["gold_chunks"]
        rec.update(hit=hit_at_k(retrieved, gold), recall=recall_at_k(retrieved, gold),
                   rr=reciprocal_rank(retrieved, gold))
    return rec


def eval_generation(row: dict, k: int) -> dict:
    rec = eval_retrieval(row, k)
    try:
        result = answer(row["question"], k=k)
        rec.update(answer=result.answer, abstained=not result.answerable,
                   citations=result.citations, invalid_citations=result.invalid_citations)
        if result.answerable:
            verdict = judge(row["question"], format_context(result.hits),
                            result.answer, row["reference"])
            unsupported = [c["claim"] for c in verdict["claims"] if not c["supported"]]
            rec.update(claims=len(verdict["claims"]), unsupported_claims=unsupported,
                       correctness=verdict["correctness"], judge_reasoning=verdict["reasoning"])
            rec["hallucinated"] = bool(unsupported or result.invalid_citations)
        elif rec["answerable_gold"]:
            rec["correctness"] = "incorrect"  # wrongly abstained
    except Exception as e:  # one bad question must not kill the whole run
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def summarize(records: list[dict], retrieval_only: bool) -> dict:
    ans = [r for r in records if r["answerable_gold"]]
    m = {"hit@k": mean([r["hit"] for r in ans]),
         "recall@k": mean([r["recall"] for r in ans]),
         "mrr": mean([r["rr"] for r in ans])}
    if retrieval_only:
        return m
    ok = [r for r in records if "error" not in r]
    answered = [r for r in ok if not r["abstained"]]
    unanswerable = [r for r in ok if not r["answerable_gold"]]
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    m.update({
        "answer_correctness": mean([score[r["correctness"]] for r in ok if r["answerable_gold"]]),
        "faithfulness": (sum(r["claims"] - len(r["unsupported_claims"]) for r in answered)
                         / max(1, sum(r["claims"] for r in answered))),
        "hallucination_rate": mean([float(r["hallucinated"]) for r in answered]),
        "abstention_accuracy": mean([float(r["abstained"]) for r in unanswerable]),
        "false_abstention_rate": mean([float(r["abstained"]) for r in ok if r["answerable_gold"]]),
        "errors": len(records) - len(ok),
    })
    return m


def render(console: Console, records: list[dict], metrics: dict, k: int,
           retrieval_only: bool) -> bool:
    t = Table(title=f"Per-question results (k={k})", title_justify="left")
    for col in ["ID", "Question", "Retrieved (top 3)", "Retrieval"]:
        t.add_column(col, overflow="fold")
    if not retrieval_only:
        t.add_column("Answer")
        t.add_column("Judge")
    for r in records:
        if r["answerable_gold"]:
            ret = "[green]hit[/]" if r["hit"] else "[red]MISS[/]"
            ret += f" rr={r['rr']:.2f}"
        else:
            ret = "[dim]n/a (no answer exists)[/]"
        row = [r["id"], r["question"], ", ".join(r["retrieved"][:3]) or "-", ret]
        if not retrieval_only:
            if "error" in r:
                row += ["[red]ERROR[/]", r["error"][:40]]
            elif r["abstained"]:
                good = not r["answerable_gold"]
                row += ["[green]abstained[/]" if good else "[red]abstained[/]",
                        "correct" if good else "[red]wrongly abstained[/]"]
            else:
                colour = {"correct": "green", "partial": "yellow", "incorrect": "red"}[r["correctness"]]
                judge_txt = f"[{colour}]{r['correctness']}[/]"
                if r["hallucinated"]:
                    judge_txt += f" [red]+{len(r['unsupported_claims'])} unsupported[/]"
                if not r["answerable_gold"]:
                    judge_txt = "[red]should have abstained[/]" + (
                        " [red](hallucinated)[/]" if r["hallucinated"] else "")
                row += ["answered", judge_txt]
        t.add_row(*row)
    console.print(t)

    s = Table(title="Summary", title_justify="left")
    for col in ["Metric", "Value", "Threshold", "Status"]:
        s.add_column(col)
    passed = True
    for name, value in metrics.items():
        if name == "errors":
            continue
        th = THRESHOLDS.get(name)
        val = "-" if value is None else f"{value:.2f}"
        if th is None or value is None:
            s.add_row(name, val, "-", "[dim]info[/]")
            continue
        limit, direction = th
        ok = value >= limit if direction == "min" else value <= limit
        passed &= ok
        s.add_row(name, val, f"{'>=' if direction == 'min' else '<='} {limit:.2f}",
                  "[green]PASS[/]" if ok else "[red]FAIL[/]")
    if metrics.get("errors"):
        passed = False
        s.add_row("errors", str(metrics["errors"]), "0", "[red]FAIL[/]")
    console.print(s)
    console.print("[bold green]All thresholds met.[/]" if passed
                  else "[bold red]Some thresholds missed. See reports/latest.json for details.[/]")
    return passed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=4, help="clauses to retrieve per question")
    p.add_argument("--retrieval-only", action="store_true", help="skip the LLM (no API key needed)")
    p.add_argument("--limit", type=int, help="only run the first N questions")
    p.add_argument("--workers", type=int, default=4, help="parallel questions (full run)")
    p.add_argument("--svg", type=Path, help="save the console report as an SVG screenshot")
    args = p.parse_args()

    rows = load_golden(args.limit)
    console = Console(record=bool(args.svg), width=140 if args.svg else None)
    if args.retrieval_only:
        records = [eval_retrieval(r, args.k) for r in rows]
    else:
        with console.status(f"Running {len(rows)} questions through RAG + judge..."):
            with ThreadPoolExecutor(args.workers) as pool:
                records = list(pool.map(lambda r: eval_generation(r, args.k), rows))

    metrics = summarize(records, args.retrieval_only)
    passed = render(console, records, metrics, args.k, args.retrieval_only)

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({"k": args.k, "retrieval_only": args.retrieval_only,
                                  "metrics": metrics, "records": records}, indent=2),
                      encoding="utf-8")
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        console.save_svg(str(args.svg), title="python evaluate.py")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
