"""Run this FIRST. Prints the actual schema of your sandbox so you can fix the
prompt in graph_agent.py if my placeholders are wrong."""
from graph import run_cypher

print("=== Node labels and counts ===")
for row in run_cypher("MATCH (n) RETURN labels(n) AS label, count(*) AS n ORDER BY n DESC"):
    print(f"  {row['label']}: {row['n']:,}")

print("\n=== Relationship types and counts ===")
for row in run_cypher("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC"):
    print(f"  {row['type']}: {row['n']:,}")

print("\n=== Sample property keys per label ===")
for row in run_cypher("""
    MATCH (n)
    WITH labels(n)[0] AS label, keys(n) AS k LIMIT 200
    UNWIND k AS key
    RETURN label, collect(DISTINCT key) AS keys
"""):
    print(f"  {row['label']}: {row['keys']}")

print("\n=== Smoke test: 3 example questions ===")
for row in run_cypher("MATCH (q:Question) RETURN q.title AS title LIMIT 3"):
    print(f"  - {row['title']}")
