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


# Pricing for us.anthropic.claude-sonnet-4-5-20250929-v1:0 on Amazon Bedrock (USD per 1M tokens)
PRICE_INPUT_PER_M = 3.00
PRICE_OUTPUT_PER_M = 15.00


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * PRICE_INPUT_PER_M + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_M


QUERIES = [
    {
        "label": "Q1 · Aggregate count",
        "q": "How many questions tagged 'cypher' have an accepted answer? Give me the exact number.",
        "insight": "RAG can only summarize top-3 docs and will decline (or fabricate). Graph-RAG returns the exact `COUNT()` over `Answer.is_accepted`.",
        "truth": "467 (verifiable via Cypher)",
    },
    {
        "label": "Q2 · Multi-hop join",
        "q": "Which users have answered questions tagged 'cypher' AND also asked questions tagged 'java'? List their display names.",
        "insight": "RAG cannot traverse User→Answer→Question + User→Question paths. Graph-RAG does the multi-hop join in one Cypher query.",
        "truth": "9 distinct users",
    },
    {
        "label": "Q3 · Co-occurrence",
        "q": "What are the top 5 tags that most frequently co-occur with 'neo4j-apoc' on the same question?",
        "insight": "RAG retrieves passages, not co-occurrence statistics. Graph-RAG aggregates over the entire 1,589-question dataset.",
        "truth": "neo4j 122, cypher 86, graph-databases 9, graph-data-science 4, csv 4",
    },
    {
        "label": "Q4 · The Antarctica Test",
        "q": "List the top 3 questions tagged 'cypher' that have more than 50,000 views. Show their exact titles and view counts.",
        "insight": "Max view_count in this graph is 1,851 — **zero** cypher questions exceed 50k. RAG can't see view counts; Graph-RAG returns the empty set honestly.",
        "truth": "0 results (max view_count is 1,851)",
    },
]


def run_rag(q: str) -> dict:
    rag_mod.reset_capture()
    fresh = Agent(name="RAG_Agent", system_prompt=RAG_SYSTEM_PROMPT, tools=[search_questions], model=MODEL)
    t0 = time.monotonic()
    try:
        r = fresh(q)
        text = r.message["content"][0]["text"]
        usage = r.metrics.accumulated_usage
    except Exception as e:
        text = f"_Error: {e}_"
        usage = {"inputTokens": 0, "outputTokens": 0}
    in_t = usage.get("inputTokens", 0)
    out_t = usage.get("outputTokens", 0)
    return {
        "text": text,
        "latency": time.monotonic() - t0,
        "retrieved": rag_mod.get_captured(),
        "input_tokens": in_t,
        "output_tokens": out_t,
        "total_tokens": in_t + out_t,
        "cost": _cost(in_t, out_t),
    }


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
        usage = r.metrics.accumulated_usage
    except Exception as e:
        fb = run_canned_graph(q, e)
        fb["latency"] = time.monotonic() - t0
        fb.setdefault("input_tokens", 0)
        fb.setdefault("output_tokens", 0)
        fb.setdefault("total_tokens", 0)
        fb.setdefault("cost", 0.0)
        return fb
    in_t = usage.get("inputTokens", 0)
    out_t = usage.get("outputTokens", 0)
    return {
        "text": text,
        "latency": time.monotonic() - t0,
        "cypher": graph_mod.get_captured(),
        "input_tokens": in_t,
        "output_tokens": out_t,
        "total_tokens": in_t + out_t,
        "cost": _cost(in_t, out_t),
        "fallback": False,
    }


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

    payload = json.dumps({"nodes": trace["nodes"], "edges": trace["edges"]})
    # Guard against </script> appearing in any string value
    payload = payload.replace("</", "<\\/")
    components.html(
        f"""
        <div id="neo4j-trace" style="width:100%;height:460px;background:#fff;border-radius:6px;"></div>
        <pre id="neo4j-trace-error" style="color:#c00;font-size:12px;white-space:pre-wrap;"></pre>
        <script id="neo4j-trace-data" type="application/json">{payload}</script>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <script>
        (function() {{
          const errEl = document.getElementById("neo4j-trace-error");
          function fail(msg) {{ errEl.textContent = "render error: " + msg; }}
          try {{
            if (typeof vis === "undefined") {{ fail("vis-network failed to load from CDN"); return; }}
            const graph = JSON.parse(document.getElementById("neo4j-trace-data").textContent);
            const colors = {{
              User:     {{ background: "#6ea8ff", border: "#376ad8" }},
              Question: {{ background: "#ffb86b", border: "#bf6d1f" }},
              Answer:   {{ background: "#7be495", border: "#2b9a52" }},
              Tag:      {{ background: "#c792ea", border: "#7b42ad" }}
            }};
            const nodes = new vis.DataSet(graph.nodes.map(function(n) {{
              return Object.assign({{}}, n, {{
                shape: n.group === "Question" ? "box" : "dot",
                color: colors[n.group] || {{ background: "#ccc", border: "#888" }},
                font: {{ color: "#1f2330", size: 13, face: "Inter, system-ui, sans-serif" }},
                margin: 8
              }});
            }}));
            const edges = new vis.DataSet(graph.edges.map(function(e) {{
              return Object.assign({{}}, e, {{
                arrows: "to",
                color: {{ color: "#9aa3b2", highlight: "#3478f6" }},
                font: {{ color: "#555", size: 10, strokeWidth: 3, strokeColor: "#fff" }},
                smooth: {{ type: "dynamic" }}
              }});
            }}));
            new vis.Network(
              document.getElementById("neo4j-trace"),
              {{ nodes: nodes, edges: edges }},
              {{
                autoResize: true,
                physics: {{ solver: "forceAtlas2Based", stabilization: {{ iterations: 120 }} }},
                interaction: {{ hover: true, tooltipDelay: 120 }}
              }}
            );
          }} catch (err) {{
            fail((err && err.message) || String(err));
          }}
        }})();
        </script>
        """,
        height=480,
    )


st.title("🧠 Context Graphs vs Vector RAG")
st.markdown(
    "**You're prepping a board deck on Neo4j community traction.** "
    "Two research tools, same question. One declines or guesses from a 3-doc sample. "
    "The other generates Cypher you can audit."
)
st.caption(
    "Corpus: 1,589 Stack Overflow questions · 1,367 answers · 1,365 users · 476 tags. "
    "Both agents wrap Claude Sonnet 4.5 on Amazon Bedrock (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`)."
)

st.divider()
st.markdown("### Pick a question")

cols = st.columns(4)
if "pending" not in st.session_state:
    st.session_state.pending = None

for i, t in enumerate(QUERIES):
    with cols[i]:
        if st.button(t["label"], use_container_width=True, key=f"btn_{i}"):
            st.session_state.pending = (t["q"], t["insight"], t.get("truth"))

with st.form("custom_form", clear_on_submit=False):
    custom_q = st.text_input(
        "…or ask your own:",
        placeholder="e.g. Which user has answered the most cypher questions?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Run custom →")
    if submitted and custom_q.strip():
        st.session_state.pending = (custom_q.strip(), "", None)

st.divider()

if not st.session_state.pending:
    st.info("👆 Pick a pre-canned question or type your own to start.")
else:
    pending = st.session_state.pending
    q, insight, truth = pending if len(pending) == 3 else (*pending, None)
    st.markdown(f"#### ❓ {q}")
    if truth:
        st.caption(f"**Ground truth (verifiable via Cypher):** {truth}")

    with st.spinner("Both agents thinking in parallel…"):
        with ThreadPoolExecutor(max_workers=2) as ex:
            rag_future = ex.submit(run_rag, q)
            graph_future = ex.submit(run_graph, q)
            rag_result = rag_future.result()
            graph_result = graph_future.result()

    # ===== AT-A-GLANCE ANSWER STRIP =====
    st.markdown("#### Side-by-side answers")
    a_left, a_right = st.columns(2, gap="large")
    with a_left:
        st.markdown("##### 🟥 Vanilla RAG  ·  FAISS top-3")
        st.error(rag_result["text"])
    with a_right:
        st.markdown("##### 🟩 Graph-RAG  ·  Cypher → Neo4j")
        st.success(graph_result["text"])

    # ===== COST & LATENCY COMPARISON =====
    st.markdown("#### Cost & latency")
    m_left, m_right = st.columns(2, gap="large")
    for col, res, label in [(m_left, rag_result, "RAG"), (m_right, graph_result, "Graph-RAG")]:
        with col:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latency", f"{res['latency']:.1f}s")
            c2.metric("Input tok", f"{res['input_tokens']:,}")
            c3.metric("Output tok", f"{res['output_tokens']:,}")
            c4.metric("Cost", f"${res['cost']:.4f}")

    # ===== COST DELTA CALLOUT =====
    cost_delta = graph_result["cost"] - rag_result["cost"]
    cost_ratio = graph_result["cost"] / rag_result["cost"] if rag_result["cost"] > 0 else float("inf")
    tok_delta = graph_result["total_tokens"] - rag_result["total_tokens"]
    st.caption(
        f"📊 **Delta:** Graph-RAG used {tok_delta:+,} tokens "
        f"(${cost_delta:+.4f}, {cost_ratio:.1f}× the cost of RAG) — "
        f"and is the only one that gave a verifiable answer."
    )

    # ===== AUDIT TRAIL =====
    st.markdown("#### Audit trail — what each agent actually saw")
    v_left, v_right = st.columns(2, gap="large")

    with v_left:
        st.markdown(f"**📄 RAG retrieved {len(rag_result['retrieved'])} passages:**")
        if not rag_result["retrieved"]:
            st.caption("_No passages retrieved._")
        for p in rag_result["retrieved"]:
            with st.container(border=True):
                title = p.get("title") or "(untitled)"
                link = p.get("link") or ""
                if link:
                    st.markdown(f"**{title}**  ·  [stackoverflow ↗]({link})")
                else:
                    st.markdown(f"**{title}**")
                st.text(p["snippet"])

    with v_right:
        st.markdown(f"**🔍 Graph-RAG ran {len(graph_result['cypher'])} Cypher quer{'y' if len(graph_result['cypher'])==1 else 'ies'}:**")
        if graph_result.get("fallback"):
            st.warning("⚠️ Demo fallback: Bedrock call failed; this ran a canned Cypher template directly against Neo4j.")
        for c in graph_result["cypher"]:
            with st.container(border=True):
                st.code(c["cypher"], language="cypher")
                st.caption(f"→ {c['row_count']} row(s) returned")
                if c.get("sample"):
                    st.json(c["sample"], expanded=False)
                if c.get("error"):
                    st.error(c["error"])
        st.success(
            "✅ **Verify independently:** copy any Cypher above, paste it into "
            "[Neo4j Browser](https://sandbox.neo4j.com) (your Stack Overflow sandbox's interactive console). "
            "Same row count = the agent didn't hallucinate."
        )

    # ===== LIVE SUBGRAPH (visual proof) =====
    st.markdown("#### Live subgraph — what Graph-RAG traversed")
    trace = build_neo4j_trace(q, graph_result["cypher"])
    tag_text = ", ".join(trace["tags"]) if trace["tags"] else "top-viewed questions"
    st.caption(f"Sampled from Neo4j around: **{tag_text}**. Each node is a real row in the graph; hover to inspect.")
    render_trace_graph(trace)
    with st.expander("Trace query used for the visualization"):
        st.code(trace["cypher"], language="cypher")

    if insight:
        st.divider()
        st.info(f"💡 **Why these differ:** {insight}")

st.divider()
st.caption(
    f"Pricing: ${PRICE_INPUT_PER_M:.2f} per 1M input tokens, ${PRICE_OUTPUT_PER_M:.2f} per 1M output tokens "
    "(Claude Sonnet 4.5 on Amazon Bedrock, us-east-1). "
    "Both agents use the identical model — the only variable is the tool: vector search vs read-only Cypher."
)
