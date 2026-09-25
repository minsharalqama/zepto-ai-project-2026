from __future__ import annotations

import json
from pathlib import Path

from rag_engine import RAGEngine

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs" / "example_calls.md"


def main() -> None:
    engine = RAGEngine()
    examples = [
        "What delivery fee applies to orders below INR 149?",
        "What is the capital of France?",
    ]
    parts = ["# Example Calls (MOCK_LLM default)", ""]
    for query in examples:
        response = engine.ask(query)
        payload = response.model_dump()
        parts.append(f"## Query\n\n`{query}`\n")
        parts.append("### Raw JSON\n")
        parts.append("```json")
        parts.append(json.dumps(payload, ensure_ascii=False))
        parts.append("```\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
