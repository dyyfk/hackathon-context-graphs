import os
os.environ["OTEL_SDK_DISABLED"] = "true"

import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from strands import Agent

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


def run_graph(q: str) -> dict:
    graph_mod.reset_capture()
    fresh = Agent(name="GraphRAG_Agent", system_prompt=GRAPH_SYSTEM_PROMPT, tools=[query_knowledge_graph], model=MODEL)
    t0 = time.monotonic()
    try:
        r = fresh(q)
        text = r.message["content"][0]["text"]
    except Exception as e:
        text = f"_Error: {e}_"
    return {"text": text, "latency": time.monotonic() - t0, "cypher": graph_mod.get_captured()}


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
