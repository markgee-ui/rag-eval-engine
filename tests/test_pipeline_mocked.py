"""Pipeline + eval logic with the LLM mocked out (no API key, no cost)."""

import evaluate
import rag.pipeline as pipeline


def test_off_topic_question_abstains_without_calling_llm(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("LLM should not be called when nothing is retrieved")
    monkeypatch.setattr(pipeline, "structured_call", boom)
    result = pipeline.answer("recipe for banana bread")
    assert not result.answerable and result.hits == []


def test_citations_to_unretrieved_clauses_are_dropped(monkeypatch):
    monkeypatch.setattr(pipeline, "structured_call", lambda *a, **kw: {
        "answerable": True, "answer": "Within 48 hours.",
        "citations": ["dpa:4", "msa:99", "dpa:4"]})
    result = pipeline.answer("How quickly must a personal data breach be notified?")
    assert result.citations == ["dpa:4"]
    assert result.invalid_citations == ["msa:99"]


def test_eval_flags_hallucination_and_scores_abstention(monkeypatch):
    def fake_answer(question, k=4):
        r = pipeline.RagResult(question, [], "x", answerable="trial" not in question)
        return r
    monkeypatch.setattr(evaluate, "answer", fake_answer)
    monkeypatch.setattr(evaluate, "judge", lambda *a: {
        "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
        "correctness": "partial", "reasoning": ""})

    rows = [r for r in evaluate.load_golden(None) if r["id"] in ("q06", "q19")]
    records = [evaluate.eval_generation(r, 4) for r in rows]
    m = evaluate.summarize(records, retrieval_only=False)
    assert m["hallucination_rate"] == 1.0      # q06 answered with an unsupported claim
    assert m["faithfulness"] == 0.5
    assert m["answer_correctness"] == 0.5      # partial
    assert m["abstention_accuracy"] == 1.0     # q19 (free trial) correctly abstained
