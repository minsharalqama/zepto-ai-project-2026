from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

PROMPT_TEMPLATE = """
ROLE:
You are Zepto's policy support assistant. You answer customer questions using only the provided Zepto policy context.

CONTEXT:
{context}

TASK:
Answer the user's question directly and only from the supplied context. If the context does not contain the answer, say that the provided policy context does not specify it.

FORMAT:
Return valid JSON with exactly these fields: answer (string), sources (list of document/chunk IDs), confidence (float from 0 to 1).

LENGTH:
Keep the answer concise, normally 1-3 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context. Do not invent policy rules, prices, dates, or support procedures.

FEW-SHOT EXAMPLE:
User: What is the delivery fee for an order below INR 149?
Context: Orders below INR 149 incur a flat INR 25 delivery fee.
Assistant: {{"answer":"Orders below INR 149 incur a flat INR 25 delivery fee.","sources":["doc_01"],"confidence":1.0}}

USER QUESTION:
{question}
""".strip()


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AppState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_ids: list[str]
    retrieved_docs: list[str]
    retrieved_distances: list[float]
    response: dict


def is_mock() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY in the environment.")
    return Groq(api_key=api_key)


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class RAGEngine:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                "Chroma collection not found. Run `python support_assistant/ingest.py` first."
            ) from exc
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.graph = self._build_graph()

    def _embed(self, text: str) -> list[float]:
        return self.embedding_model.encode([text], normalize_embeddings=True)[0].tolist()

    def _classify_with_real_llm(self, query: str) -> str:
        prompt = (
            "Classify the query as exactly one of policy_question or general_question. "
            "Use policy_question only if the query asks about Zepto policies in delivery, returns, refunds, "
            "membership, tracking, cancellation, gift cards, or support. Return only the label.\n\n"
            f"Query: {query}"
        )
        client = get_groq_client()
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()
        return "policy_question" if "policy_question" in label else "general_question"

    def classify_intent(self, state: AppState) -> dict:
        query = state["query"]
        if is_mock():
            lower = query.lower()
            intent = "policy_question" if any(k in lower for k in POLICY_KEYWORDS) else "general_question"
        else:
            intent = self._classify_with_real_llm(query)
        return {"intent": intent}

    def retrieve(self, query: str) -> tuple[list[str], list[str], list[float]]:
        query_embedding = self._embed(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        distances = result["distances"][0]
        return ids, docs, distances

    def _generate_real_llm(self, question: str, context_ids: list[str], context_docs: list[str]) -> AnswerResponse:
        context = "\n\n".join(
            f"[{doc_id}] {doc}" for doc_id, doc in zip(context_ids, context_docs)
        )
        base_prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        last_error = ""

        for attempt in range(3):
            if attempt == 0:
                prompt = base_prompt
            else:
                prompt = (
                    base_prompt
                    + "\n\nCORRECTIVE INSTRUCTION:\n"
                    + f"The previous output failed schema validation with: {last_error}. "
                      "Return JSON only and make sure answer is a string, sources is a list of IDs from the supplied context, "
                      "and confidence is a number between 0 and 1."
                )

            client = get_groq_client()
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = clean_json_text(response.choices[0].message.content or "")
            try:
                return AnswerResponse.model_validate_json(raw)
            except ValidationError as exc:
                last_error = str(exc)

        return AnswerResponse(
            answer=f"ERROR: The real-LLM response did not validate after 3 attempts. {last_error}",
            sources=context_ids,
            confidence=0.0,
        )

    def retrieve_and_answer(self, state: AppState) -> dict:
        ids, docs, distances = self.retrieve(state["query"])
        if is_mock():
            snippet = docs[0][:200]
            response = AnswerResponse(
                answer=f"Based on the retrieved context: {snippet}",
                sources=ids,
                confidence=1.0,
            )
        else:
            response = self._generate_real_llm(state["query"], ids, docs)
        return {
            "retrieved_ids": ids,
            "retrieved_docs": docs,
            "retrieved_distances": distances,
            "response": response.model_dump(),
        }

    def direct_answer(self, state: AppState) -> dict:
        if is_mock():
            response = AnswerResponse(
                answer="I can only answer questions about Zepto policies right now.",
                sources=[],
                confidence=1.0,
            )
        else:
            context = "No retrieval context is provided for general questions."
            prompt = PROMPT_TEMPLATE.format(context=context, question=state["query"])
            base_prompt = (
                prompt
                + "\n\nIMPORTANT: This is a general-question route. Do not invent Zepto policy facts; answer the general question only if the LLM can do so without policy context."
            )
            last_error = ""
            for attempt in range(3):
                prompt_to_send = base_prompt
                if attempt > 0:
                    prompt_to_send += (
                        "\nCORRECTIVE INSTRUCTION: Previous output failed JSON validation. "
                        f"Validation error: {last_error}. Return JSON only."
                    )
                client = get_groq_client()
                result = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    messages=[{"role": "user", "content": prompt_to_send}],
                    temperature=0,
                )
                raw = clean_json_text(result.choices[0].message.content or "")
                try:
                    response = AnswerResponse.model_validate_json(raw)
                    response = AnswerResponse(
                        answer=response.answer,
                        sources=[],
                        confidence=response.confidence,
                    )
                    break
                except ValidationError as exc:
                    last_error = str(exc)
            else:
                response = AnswerResponse(
                    answer=f"ERROR: The real-LLM response did not validate after 3 attempts. {last_error}",
                    sources=[],
                    confidence=0.0,
                )
        return {"response": response.model_dump()}

    @staticmethod
    def route(state: AppState) -> Literal["retrieve_and_answer", "direct_answer"]:
        return "retrieve_and_answer" if state.get("intent") == "policy_question" else "direct_answer"

    def _build_graph(self):
        builder = StateGraph(AppState)
        builder.add_node("classify_intent", self.classify_intent)
        builder.add_node("retrieve_and_answer", self.retrieve_and_answer)
        builder.add_node("direct_answer", self.direct_answer)
        builder.add_edge(START, "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self.route,
            {
                "retrieve_and_answer": "retrieve_and_answer",
                "direct_answer": "direct_answer",
            },
        )
        builder.add_edge("retrieve_and_answer", END)
        builder.add_edge("direct_answer", END)
        return builder.compile()

    def ask(self, query: str) -> AnswerResponse:
        state = self.graph.invoke({"query": query})
        return AnswerResponse.model_validate(state["response"])
