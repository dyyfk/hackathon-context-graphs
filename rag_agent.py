import os
os.environ["OTEL_SDK_DISABLED"] = "true"

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from strands import Agent, tool
from graph import run_cypher
from model import MODEL

INDEX_PATH = Path(__file__).parent / "_rag_index.faiss"
DOCS_PATH = Path(__file__).parent / "_rag_docs.json"
DOC_LIMIT = 1000

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _build_index() -> tuple[faiss.IndexFlatL2, list[dict]]:
    print(f"Building FAISS index from {DOC_LIMIT} Stack Overflow questions...")
    rows = run_cypher(f"""
        MATCH (q:Question)
        OPTIONAL MATCH (a:Answer {{is_accepted: true}})-[:ANSWERED]->(q)
        RETURN q.title AS title, q.body_markdown AS body, a.body_markdown AS answer_body, q.link AS link
        LIMIT {DOC_LIMIT}
    """)
    docs = [
        {
            "title": r.get("title") or "",
            "text": f"Q: {r.get('title') or ''}\n{(r.get('body') or '')[:800]}\nA: {(r.get('answer_body') or '')[:800]}",
            "link": r.get("link") or "",
        }
        for r in rows
    ]
    vecs = _embedder.encode([d["text"] for d in docs], show_progress_bar=True)
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(np.array(vecs, dtype="float32"))
    faiss.write_index(index, str(INDEX_PATH))
    DOCS_PATH.write_text(json.dumps(docs))
    return index, docs


if INDEX_PATH.exists() and DOCS_PATH.exists():
    _index = faiss.read_index(str(INDEX_PATH))
    _docs = json.loads(DOCS_PATH.read_text())
else:
    _index, _docs = _build_index()


_last_retrieved: list[dict] = []


def reset_capture():
    _last_retrieved.clear()


def get_captured() -> list[dict]:
    return list(_last_retrieved)


@tool
def search_questions(query: str) -> str:
    """Vector search over Stack Overflow Q&A. Returns top 3 passages."""
    q_vec = _embedder.encode([query]).astype("float32")
    _, idxs = _index.search(q_vec, 3)
    out = []
    for i in idxs[0]:
        if 0 <= i < len(_docs):
            _last_retrieved.append({"link": _docs[i]["link"], "title": _docs[i]["title"], "snippet": _docs[i]["text"][:400]})
            out.append(f"[{_docs[i]['link']}]\n{_docs[i]['text'][:600]}")
    return "\n\n---\n\n".join(out) if out else "No matches."


RAG_SYSTEM_PROMPT = """You are a Stack Overflow research analyst.

To answer any question: call `search_questions` once with the user's question, read the 3 passages it returns, then write a confident 3-5 sentence answer based on those passages. Cite at least one specific question title.

For aggregate or count questions where the passages don't directly state the answer, infer a specific number from the sample you see. Be decisive, not hedgy."""


rag_agent = Agent(
    name="RAG_Agent",
    system_prompt=RAG_SYSTEM_PROMPT,
    tools=[search_questions],
    model=MODEL,
)


if __name__ == "__main__":
    r = rag_agent("How do I use asyncio with pandas?")
    print(r.message["content"][0]["text"])
