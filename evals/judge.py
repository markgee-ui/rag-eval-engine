"""LLM-as-judge for generation quality.

Implements the same ideas as RAGAS "faithfulness" and "answer correctness":
1. Break the answer into atomic claims.
2. Check each claim against the retrieved clauses ONLY -> unsupported = hallucination.
3. Compare the answer to a human-written reference answer -> correctness.

The judge sees the retrieved context (not the whole corpus), so a claim that is true
in some other clause but was not retrieved still counts as unsupported: the answer
was not grounded in what the model was given.
"""

import os

from rag.llm import MODEL, structured_call

# Set RAG_JUDGE_MODEL to a different model to reduce self-preference bias.
JUDGE_MODEL = os.environ.get("RAG_JUDGE_MODEL", MODEL)

JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a contract \
question-answering system. You will receive the question, the contract clauses the \
system retrieved (CONTEXT), the system's ANSWER, and a REFERENCE answer written by a human.

Task 1 - faithfulness. Split ANSWER into atomic factual claims (one fact each; skip \
filler such as "according to the contract"). For each claim set `supported` to true \
only if the CONTEXT states it or it follows directly from the CONTEXT. Claims based on \
outside knowledge or assumptions are unsupported, even if plausibly true. A statement \
that the documents do not cover something counts as supported if the CONTEXT indeed \
does not cover it.

Task 2 - correctness. Compare ANSWER to REFERENCE:
- "correct": contains all key facts of REFERENCE (numbers, periods, conditions) and \
contradicts none.
- "partial": some key facts are right but at least one is missing.
- "incorrect": contradicts REFERENCE, or misses its main point.
Wording differences do not matter; extra correct detail is fine."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        },
        "correctness": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "reasoning": {"type": "string"},
    },
    "required": ["claims", "correctness", "reasoning"],
    "additionalProperties": False,
}


def judge(question: str, context: str, answer: str, reference: str) -> dict:
    prompt = (f"QUESTION:\n{question}\n\nCONTEXT:\n{context or '(no clauses retrieved)'}\n\n"
              f"ANSWER:\n{answer}\n\nREFERENCE:\n{reference}")
    return structured_call(JUDGE_SYSTEM, prompt, JUDGE_SCHEMA, model=JUDGE_MODEL, effort="high")
