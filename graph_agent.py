import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from strands import Agent, tool
from graph import run_cypher
from model import MODEL

SYSTEM_PROMPT = """You are a research agent over a Stack Overflow knowledge graph (1,589 questions, 1,367 answers, 1,365 users, 476 tags).

Schema:
  Nodes:
    User(uuid, display_name)
    Question(uuid, title, link, view_count, answer_count, creation_date, body_markdown, accepted_answer_id)
    Answer(uuid, title, link, score, is_accepted, body_markdown)
    Tag(name, link)
  Relationships:
    (User)-[:ASKED]->(Question)
    (User)-[:PROVIDED]->(Answer)
    (Answer)-[:ANSWERED]->(Question)
    (Question)-[:TAGGED]->(Tag)

Accepted-answer logic: use Answer.is_accepted = true. (There is no ACCEPTED_ANSWER edge.)

Rules:
  - Use ONLY read queries: MATCH/WHERE/RETURN. Never CREATE/MERGE/DELETE/SET.
  - If a query returns zero rows, say "No results in the graph." Do NOT invent.
  - You may run multiple queries to explore before answering.
  - Always cite specific node properties (titles, display_names) in your final answer.

Example Cypher patterns:
  // Count questions tagged X that have an accepted answer
  MATCH (q:Question)-[:TAGGED]->(:Tag {name:'neo4j'})
  MATCH (a:Answer {is_accepted: true})-[:ANSWERED]->(q)
  RETURN count(DISTINCT q) AS n;

  // Tag co-occurrence
  MATCH (:Tag {name:'cypher'})<-[:TAGGED]-(q)-[:TAGGED]->(t:Tag)
  WHERE t.name <> 'cypher' RETURN t.name, count(*) ORDER BY count(*) DESC LIMIT 5;

  // Multi-hop user
  MATCH (u:User)-[:PROVIDED]->(:Answer)-[:ANSWERED]->(:Question)-[:TAGGED]->(:Tag {name:'X'})
  MATCH (u)-[:ASKED]->(:Question)-[:TAGGED]->(:Tag {name:'Y'})
  RETURN DISTINCT u.display_name LIMIT 10;
"""


_last_cypher: list[dict] = []


def reset_capture():
    _last_cypher.clear()


def get_captured() -> list[dict]:
    return list(_last_cypher)


@tool
def query_knowledge_graph(cypher: str) -> str:
    """Execute a read-only Cypher query against the Stack Overflow Neo4j graph.

    Returns the rows as a string, or 'No results.' if empty.
    """
    try:
        rows = run_cypher(cypher)
        _last_cypher.append({"cypher": cypher, "row_count": len(rows), "sample": rows[:5]})
        if not rows:
            return "No results."
        out = f"Found {len(rows)} rows:\n"
        for r in rows[:20]:
            out += f"  {r}\n"
        return out
    except Exception as e:
        _last_cypher.append({"cypher": cypher, "row_count": 0, "sample": [], "error": str(e)})
        return f"Cypher error: {e}"


graph_agent = Agent(
    name="GraphRAG_Agent",
    system_prompt=SYSTEM_PROMPT,
    tools=[query_knowledge_graph],
    model=MODEL,
)


if __name__ == "__main__":
    r = graph_agent("How many questions are tagged 'python'?")
    print(r.message["content"][0]["text"])
