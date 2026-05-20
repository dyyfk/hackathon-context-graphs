import os
os.environ["OTEL_SDK_DISABLED"] = "true"

import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import streamlit.components.v1 as components
from strands import Agent

from graph import run_cypher
from model import MODEL
import rag_agent as rag_mod
import graph_agent as graph_mod
from rag_agent import search_questions, RAG_SYSTEM_PROMPT
from graph_agent import query_knowledge_graph, SYSTEM_PROMPT as GRAPH_SYSTEM_PROMPT


st.set_page_config(
    page_title="Context Graphs vs Vector RAG",
    page_icon="🧠",
    layout="wide",
)


QUERIES = [
    {
        "label": "Q1 · Aggregate count",
        "q": "How many questions tagged 'cypher' have an accepted answer? Give me the exact number.",
        "insight": "RAG can only summarize top-3 docs and will fabricate a count. Graph-RAG returns the exact `COUNT()` over `Answer.is_accepted`.",
    },
    {
        "label": "Q2 · Multi-hop join",
        "q": "Which users have answered questions tagged 'cypher' AND also asked questions tagged 'java'? List their display names.",
        "insight": "RAG cannot traverse User→Answer→Question + User→Question paths. Graph-RAG does the multi-hop join in one Cypher query.",
    },
    {
        "label": "Q3 · Co-occurrence",
        "q": "What are the top 5 tags that most frequently co-occur with 'neo4j-apoc' on the same question?",
        "insight": "RAG retrieves passages, not co-occurrence statistics. Graph-RAG aggregates over the entire 1,589-question dataset.",
    },
    {
        "label": "Q4 · The Antarctica Test",
        "q": "List the top 3 questions tagged 'cypher' that have more than 50,000 views. Show their exact titles and view counts.",
        "insight": "Max view_count in this graph is 1,851 — **zero** cypher questions exceed 50k. RAG will confidently fabricate plausible-looking results; Graph-RAG returns the empty set honestly and can prove it with the real ceiling.",
    },
]


def run_rag(q: str) -> dict:
    rag_mod.reset_capture()
    fresh = Agent(name="RAG_Agent", system_prompt=RAG_SYSTEM_PROMPT, tools=[search_questions], model=MODEL)
    t0 = time.monotonic()
    try:
        r = fresh(q)
        text = r.message["content"][0]["text"]
    except Exception as e:
        text = f"_Error: {e}_"
    return {"text": text, "latency": time.monotonic() - t0, "retrieved": rag_mod.get_captured()}


def canned_graph_query(q: str) -> tuple[str, str] | None:
    lower = q.lower()
    if "accepted answer" in lower and "cypher" in lower and "how many" in lower:
        return (
            "accepted_cypher_questions",
            """
            MATCH (q:Question)-[:TAGGED]->(:Tag {name:'cypher'})
            MATCH (a:Answer {is_accepted: true})-[:ANSWERED]->(q)
            RETURN count(DISTINCT q) AS accepted_cypher_questions
            """,
        )
    if "answered questions tagged 'cypher'" in lower and "asked questions tagged 'java'" in lower:
        return (
            "cypher_answerers_who_asked_java",
            """
            MATCH (u:User)-[:PROVIDED]->(:Answer)-[:ANSWERED]->(:Question)-[:TAGGED]->(:Tag {name:'cypher'})
            MATCH (u)-[:ASKED]->(:Question)-[:TAGGED]->(:Tag {name:'java'})
            RETURN DISTINCT u.display_name AS user
            ORDER BY user
            LIMIT 20
            """,
        )
    if "neo4j-apoc" in lower and ("co-occur" in lower or "cooccur" in lower):
        return (
            "neo4j_apoc_cooccurring_tags",
            """
            MATCH (:Tag {name:'neo4j-apoc'})<-[:TAGGED]-(q:Question)-[:TAGGED]->(t:Tag)
            WHERE t.name <> 'neo4j-apoc'
            RETURN t.name AS tag, count(*) AS questions
            ORDER BY questions DESC, tag
            LIMIT 5
            """,
        )
    if "50,000" in lower and "cypher" in lower and "view" in lower:
        return (
            "cypher_questions_over_50000_views",
            """
            MATCH (q:Question)-[:TAGGED]->(:Tag {name:'cypher'})
            WHERE q.view_count > 50000
            RETURN q.title AS title, q.view_count AS views
            ORDER BY views DESC
            LIMIT 3
            """,
        )
    if "answered the most" in lower and "cypher" in lower:
        return (
            "top_cypher_answerers",
            """
            MATCH (u:User)-[:PROVIDED]->(:Answer)-[:ANSWERED]->(q:Question)-[:TAGGED]->(:Tag {name:'cypher'})
            RETURN u.display_name AS user, count(DISTINCT q) AS answers
            ORDER BY answers DESC, user
            LIMIT 5
            """,
        )
    return None


def run_canned_graph(q: str, error: Exception) -> dict:
    canned = canned_graph_query(q)
    if not canned:
        return {
            "text": f"_Error: {error}_",
            "cypher": [],
            "fallback": False,
        }

    name, cypher = canned
    rows = run_cypher(cypher)
    cypher_runs = [{"cypher": cypher.strip(), "row_count": len(rows), "sample": rows[:5]}]

    if name == "cypher_questions_over_50000_views" and not rows:
        proof_cypher = """
        MATCH (q:Question)-[:TAGGED]->(:Tag {name:'cypher'})
        RETURN q.title AS title, q.view_count AS views
        ORDER BY views DESC
        LIMIT 1
        """
        proof_rows = run_cypher(proof_cypher)
        cypher_runs.append({"cypher": proof_cypher.strip(), "row_count": len(proof_rows), "sample": proof_rows[:5]})
        max_row = proof_rows[0] if proof_rows else {"title": "(none)", "views": 0}
        text = (
            "**Demo fallback: Bedrock is not authorized, so this ran the audited Neo4j Cypher directly.**\n\n"
            "No cypher-tagged questions have more than 50,000 views. "
            f"The highest cypher-tagged question is **{max_row['title']}** with **{max_row['views']}** views."
        )
        return {"text": text, "cypher": cypher_runs, "fallback": True}

    if not rows:
        text = "**Demo fallback: Bedrock is not authorized, so this ran the audited Neo4j Cypher directly.**\n\nNo results in the graph."
    else:
        text = (
            "**Demo fallback: Bedrock is not authorized, so this ran the audited Neo4j Cypher directly.**\n\n"
            f"Found **{len(rows)}** row(s):\n\n"
            + "\n".join(f"- `{row}`" for row in rows[:10])
        )

    return {"text": text, "cypher": cypher_runs, "fallback": True}


def run_graph(q: str) -> dict:
    graph_mod.reset_capture()
    fresh = Agent(name="GraphRAG_Agent", system_prompt=GRAPH_SYSTEM_PROMPT, tools=[query_knowledge_graph], model=MODEL)
    t0 = time.monotonic()
    try:
        r = fresh(q)
        text = r.message["content"][0]["text"]
    except Exception as e:
        fallback = run_canned_graph(q, e)
        fallback["latency"] = time.monotonic() - t0
        return fallback
    return {"text": text, "latency": time.monotonic() - t0, "cypher": graph_mod.get_captured(), "fallback": False}



KNOWN_TAGS = ("neo4j-apoc", "graph-data-science", "graph-databases", "cypher", "neo4j", "java", "python", "csv")


def infer_trace_tags(question: str, cypher_runs: list[dict]) -> list[str]:
    text = "\n".join([question, *[c.get("cypher", "") for c in cypher_runs]]).lower()
    tags = []
    for tag in KNOWN_TAGS:
        if tag in text and tag not in tags:
            tags.append(tag)

    for match in re.findall(r"name\s*:\s*['\"]([^'\"]+)['\"]", text):
        if match not in tags:
            tags.append(match)
    return tags[:3]


def build_neo4j_trace(question: str, cypher_runs: list[dict]) -> dict:
    tags = infer_trace_tags(question, cypher_runs)
    params = {"tags": tags}
    if tags:
        trace_cypher = """
        MATCH (q:Question)-[:TAGGED]->(anchor:Tag)
        WHERE anchor.name IN $tags
        WITH DISTINCT q
        ORDER BY q.view_count DESC
        LIMIT 10
        OPTIONAL MATCH (q)-[:TAGGED]->(tag:Tag)
        OPTIONAL MATCH (asker:User)-[:ASKED]->(q)
        OPTIONAL MATCH (answer:Answer)-[:ANSWERED]->(q)
        OPTIONAL MATCH (answerer:User)-[:PROVIDED]->(answer)
        RETURN q.uuid AS q_id,
               q.title AS q_title,
               q.view_count AS views,
               collect(DISTINCT tag.name) AS tags,
               asker.uuid AS asker_id,
               asker.display_name AS asker,
               collect(DISTINCT {
                   id: answer.uuid,
                   accepted: answer.is_accepted,
                   answerer_id: answerer.uuid,
                   answerer: answerer.display_name
               }) AS answers
        """
    else:
        trace_cypher = """
        MATCH (q:Question)
        WITH q
        ORDER BY q.view_count DESC
        LIMIT 8
        OPTIONAL MATCH (q)-[:TAGGED]->(tag:Tag)
        OPTIONAL MATCH (asker:User)-[:ASKED]->(q)
        OPTIONAL MATCH (answer:Answer)-[:ANSWERED]->(q)
        OPTIONAL MATCH (answerer:User)-[:PROVIDED]->(answer)
        RETURN q.uuid AS q_id,
               q.title AS q_title,
               q.view_count AS views,
               collect(DISTINCT tag.name) AS tags,
               asker.uuid AS asker_id,
               asker.display_name AS asker,
               collect(DISTINCT {
                   id: answer.uuid,
                   accepted: answer.is_accepted,
                   answerer_id: answerer.uuid,
                   answerer: answerer.display_name
               }) AS answers
        """

    try:
        rows = run_cypher(trace_cypher, params)
    except Exception as e:
        return {"nodes": [], "edges": [], "cypher": trace_cypher, "tags": tags, "error": str(e)}

    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    def add_node(node_id: str | None, label: str | None, group: str, title: str = ""):
        if not node_id:
            return
        nodes[node_id] = {"id": node_id, "label": label or node_id, "group": group, "title": title}

    def add_edge(from_id: str | None, to_id: str | None, label: str):
        if not from_id or not to_id:
            return
        edge_id = f"{from_id}->{label}->{to_id}"
        edges[edge_id] = {"id": edge_id, "from": from_id, "to": to_id, "label": label}

    for row in rows:
        q_id = f"q:{row.get('q_id')}"
        add_node(q_id, row.get("q_title"), "Question", f"views: {row.get('views', 0)}")

        asker_id = row.get("asker_id")
        if asker_id:
            asker_node = f"u:{asker_id}"
            add_node(asker_node, row.get("asker"), "User")
            add_edge(asker_node, q_id, "ASKED")

        for tag in row.get("tags") or []:
            tag_node = f"t:{tag}"
            add_node(tag_node, f"#{tag}", "Tag")
            add_edge(q_id, tag_node, "TAGGED")

        answers = [a for a in (row.get("answers") or []) if a.get("id")][:3]
        for answer in answers:
            answer_node = f"a:{answer['id']}"
            add_node(answer_node, "accepted answer" if answer.get("accepted") else "answer", "Answer")
            add_edge(answer_node, q_id, "ANSWERED")
            answerer_id = answer.get("answerer_id")
            if answerer_id:
                answerer_node = f"u:{answerer_id}"
                add_node(answerer_node, answer.get("answerer"), "User")
                add_edge(answerer_node, answer_node, "PROVIDED")

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "cypher": trace_cypher.strip(),
        "tags": tags,
        "error": None,
    }


def render_trace_graph(trace: dict):
    if trace.get("error"):
        st.error(f"Could not load Neo4j trace: {trace['error']}")
        return
    if not trace["nodes"]:
        st.info("No graph nodes found for this trace.")
        return

    data = html.escape(json.dumps({"nodes": trace["nodes"], "edges": trace["edges"]}))
    components.html(
        f"""
        <div id="neo4j-trace" data-graph="{data}"></div>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <script>
        const graph = JSON.parse(document.getElementById("neo4j-trace").dataset.graph);
        const colors = {{
          User: {{ background: "#6ea8ff", border: "#376ad8" }},
          Question: {{ background: "#ffb86b", border: "#bf6d1f" }},
          Answer: {{ background: "#7be495", border: "#2b9a52" }},
          Tag: {{ background: "#c792ea", border: "#7b42ad" }}
        }};
        const nodes = new vis.DataSet(graph.nodes.map(n => ({{
          ...n,
          shape: n.group === "Question" ? "box" : "dot",
          color: colors[n.group],
          font: {{ color: "#e6e8ee", size: 13, face: "Inter, system-ui, sans-serif" }},
          margin: 8
        }})));
        const edges = new vis.DataSet(graph.edges.map(e => ({{
          ...e,
          arrows: "to",
          color: {{ color: "#7f8798", highlight: "#6ee7ff" }},
          font: {{ color: "#aeb6c8", size: 10, strokeWidth: 3, strokeColor: "#0b0e14" }},
          smooth: {{ type: "dynamic" }}
        }})));
        new vis.Network(
          document.getElementById("neo4j-trace"),
          {{ nodes, edges }},
          {{
            height: "460px",
            autoResize: true,
            physics: {{ solver: "forceAtlas2Based", stabilization: {{ iterations: 120 }} }},
            interaction: {{ hover: true, tooltipDelay: 120 }},
            groups: {{ User: {{}}, Question: {{}}, Answer: {{}}, Tag: {{}} }}
          }}
        );
        </script>
        """,
        height=480,
    )


st.title("🧠 Context Graphs vs Vector RAG")
st.markdown(
    "**You're prepping a board deck on Neo4j community traction.** "
    "Two research tools, same question. Watch one confidently fabricate numbers — "
    "and the other answer correctly with a query you can audit."
)
st.caption(
    "Corpus: 1,589 Stack Overflow questions · 1,367 answers · 1,365 users · 476 tags. "
    "Both agents wrap Claude Sonnet 4.5."
)

st.divider()
st.markdown("### Pick a question")

cols = st.columns(4)
if "pending" not in st.session_state:
    st.session_state.pending = None

for i, t in enumerate(QUERIES):
    with cols[i]:
        if st.button(t["label"], use_container_width=True, key=f"btn_{i}"):
            st.session_state.pending = (t["q"], t["insight"])

with st.form("custom_form", clear_on_submit=False):
    custom_q = st.text_input(
        "…or ask your own:",
        placeholder="e.g. Which user has answered the most cypher questions?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Run custom →")
    if submitted and custom_q.strip():
        st.session_state.pending = (custom_q.strip(), "")

st.divider()

if st.session_state.pending:
    q, insight = st.session_state.pending
    st.markdown(f"#### ❓ {q}")

    with st.spinner("Both agents thinking in parallel…"):
        with ThreadPoolExecutor(max_workers=2) as ex:
            rag_future = ex.submit(run_rag, q)
            graph_future = ex.submit(run_graph, q)
            rag_result = rag_future.result()
            graph_result = graph_future.result()

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### 🟥 Vanilla RAG")
        st.caption("FAISS · top-3 passages · MiniLM-L6")
        m1, m2 = st.columns(2)
        m1.metric("Latency", f"{rag_result['latency']:.1f}s")
        m2.metric("Retrieved", f"{len(rag_result['retrieved'])} docs")
        st.markdown("---")
        st.markdown(rag_result["text"])
        with st.expander(f"📄 What it retrieved ({len(rag_result['retrieved'])} passages)"):
            if not rag_result["retrieved"]:
                st.info("No passages retrieved.")
            for p in rag_result["retrieved"]:
                st.markdown(f"**{p['title'] or '(untitled)'}**  ·  [link]({p['link']})")
                st.text(p["snippet"])
                st.markdown("")

    with right:
        st.markdown("### 🟩 Graph-RAG")
        st.caption("Claude → Cypher → Neo4j")
        m1, m2 = st.columns(2)
        m1.metric("Latency", f"{graph_result['latency']:.1f}s")
        m2.metric("Queries run", f"{len(graph_result['cypher'])}")
        st.markdown("---")
        st.markdown(graph_result["text"])
        trace = build_neo4j_trace(q, graph_result["cypher"])
        st.markdown("#### Neo4j trace")
        tag_text = ", ".join(trace["tags"]) if trace["tags"] else "top viewed questions"
        st.caption(f"Live subgraph sampled from Neo4j around: {tag_text}")
        render_trace_graph(trace)
        with st.expander("Trace query used for the visualization"):
            st.code(trace["cypher"], language="cypher")
        with st.expander(f"🔍 Cypher generated ({len(graph_result['cypher'])} queries)"):
            if not graph_result["cypher"]:
                st.info("No Cypher executed.")
            for c in graph_result["cypher"]:
                st.code(c["cypher"], language="cypher")
                st.caption(f"→ {c['row_count']} row(s)")
                if c.get("sample"):
                    st.json(c["sample"], expanded=False)
                if c.get("error"):
                    st.error(c["error"])
                st.markdown("")

    if insight:
        st.divider()
        st.info(f"💡 **Why these differ:** {insight}")

else:
    st.info("👆 Pick a pre-canned question or type your own to start.")

st.divider()
st.caption(
    "Both agents are Claude Sonnet 4.5 via Strands. The only difference is the tool: "
    "vector search over passages vs. read-only Cypher against the live graph. "
    "Note the audit trail — every Graph-RAG answer is reproducible by re-running its Cypher."
)
