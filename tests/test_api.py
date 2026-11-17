"""Web API tests with the LLM mocked out."""

from fastapi.testclient import TestClient

import app as web
import rag.pipeline as pipeline

client = TestClient(web.app)


def test_index_and_health():
    assert "Contract RAG" in client.get("/").text
    assert client.get("/api/health").json()["chunks"] == 25


def test_question_length_is_validated():
    assert client.post("/api/ask", json={"question": "x" * 301}).status_code == 422


def test_missing_api_key_gives_clear_503(monkeypatch):
    web._requests.clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client.post("/api/ask", json={"question": "What is the liability cap?"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_ask_returns_answer_with_clauses(monkeypatch):
    web._requests.clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(pipeline, "structured_call", lambda *a, **kw: {
        "answerable": True, "answer": "Within 48 hours.", "citations": ["dpa:4"]})
    data = client.post("/api/ask", json={"question": "personal data breach notification"}).json()
    assert data["answer"] == "Within 48 hours." and data["citations"] == ["dpa:4"]
    assert data["clauses"][0]["id"] == "dpa:4"


def test_rate_limit(monkeypatch):
    web._requests.clear()
    monkeypatch.setattr(web, "RATE_LIMIT", 2)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    codes = [client.post("/api/ask", json={"question": "banana bread recipe"}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]
