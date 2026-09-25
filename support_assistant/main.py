from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .rag_engine import AnswerResponse, RAGEngine
except ImportError:
    from rag_engine import AnswerResponse, RAGEngine

app = FastAPI(title="Zepto Support Assistant", version="1.0.0")
engine: RAGEngine | None = None


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


@app.on_event("startup")
def startup_event() -> None:
    global engine
    engine = RAGEngine()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized")
    try:
        return engine.ask(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

