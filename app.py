"""Web API + single-page UI. Vercel detects `app` in app.py automatically.

    uvicorn app:app --reload     # local dev at http://127.0.0.1:8000
"""

import os
import time
from collections import defaultdict, deque
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag.llm import LLMError
from rag.pipeline import answer, get_retriever

INDEX_HTML = (Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")

# A public demo spends your API credit, so cap usage. This limiter lives in memory, so
# it is per server instance: it slows down casual abuse but is not a hard budget.
# Set a monthly spend limit in the Anthropic Console as the real backstop.
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "5"))
_requests: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="Contract RAG", docs_url=None, redoc_url=None)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=300)


def check_rate_limit(ip: str) -> None:
    now, window = time.monotonic(), _requests[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, "Too many questions - please wait a minute.")
    window.append(now)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "chunks": len(get_retriever().chunks),
            "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> dict:
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    check_rate_limit(ip.split(",")[0].strip())
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "Server is missing ANTHROPIC_API_KEY - set it in the "
                                 "Vercel project's Environment Variables and redeploy.")
    try:
        result = answer(body.question.strip())
    except LLMError as e:
        raise HTTPException(502, str(e))
    except anthropic.AuthenticationError:
        raise HTTPException(503, "Server is missing a valid ANTHROPIC_API_KEY.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "The model is busy - try again shortly.")
    except anthropic.APIError as e:
        raise HTTPException(502, f"Model API error: {type(e).__name__}")
    return {
        "answer": result.answer,
        "answerable": result.answerable,
        "citations": result.citations,
        "invalid_citations": result.invalid_citations,
        "clauses": [{"id": h.chunk.id, "title": f"{h.chunk.doc_title} - {h.chunk.heading}",
                     "score": round(h.score, 2), "text": h.chunk.text} for h in result.hits],
    }
