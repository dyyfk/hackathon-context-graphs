from graph_agent import graph_agent
from rag_agent import rag_agent

QUERIES = [
    {
        "q": "How many questions tagged 'cypher' have an accepted answer? Give me the exact number.",
        "insight": "RAG can only summarize top-3 docs; Graph-RAG returns exact COUNT() via Answer.is_accepted.",
    },
    {
        "q": "Which users have answered questions tagged 'cypher' AND also asked questions tagged 'java'? List their display names.",
        "insight": "RAG cannot traverse User->Answer->Question + User->Question paths; Graph-RAG does multi-hop in one query.",
    },
    {
        "q": "What are the top 5 tags that most frequently co-occur with 'neo4j-apoc' on the same question?",
        "insight": "RAG retrieves passages, not co-occurrence statistics; Graph-RAG aggregates over the entire dataset.",
    },
    {
        "q": "List the top 3 questions tagged 'cypher' that have more than 50,000 views. Show their exact titles and view counts.",
        "insight": "ANTARCTICA TEST: Max view_count in this graph is 1,851 — zero cypher questions exceed 50k. RAG cannot see view_count metadata in passages and will either fabricate plausible numbers or list popular-looking questions as if they qualified; Graph-RAG returns 0 honestly.",
    },
]


def run(label, agent, q):
    print(f"\n--- {label} ---")
    try:
        r = agent(q)
        print(r.message["content"][0]["text"])
    except Exception as e:
        print(f"[error] {e}")


for i, t in enumerate(QUERIES, 1):
    print(f"\n{'='*78}\n[{i}] {t['q']}\n{'='*78}")
    run("VANILLA RAG", rag_agent, t["q"])
    run("GRAPH-RAG", graph_agent, t["q"])
    print(f"\nINSIGHT: {t['insight']}")
