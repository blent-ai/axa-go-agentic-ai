"""Test interface of the HomeShield multi-agent system.

The system is designed to run in the background, without an interface: this
Streamlit application is only used to **test** it and to **visualize** its
execution during the Masterclass.

Usage
-----
    uv run streamlit run app.py
"""

from __future__ import annotations

import time
import uuid

import pandas as pd
import streamlit as st

from src.knowledge import (
    ATTACHMENTS_DIR,
    ROOT,
    CLAIM_TYPE_LABELS,
    load_claims,
    load_jsonl,
    policy_text,
    process_text,
)
from src.schemas import Claim

st.set_page_config(page_title="HomeShield — multi-agent", page_icon="🏠", layout="wide")

# --------------------------------------------------------------------------- #
# The graph: Graphviz rendering, nodes colored by progress
# --------------------------------------------------------------------------- #
NODES = {
    "__start__": "Claim received",
    "declaration_agent": "🧾 Declaration agent",
    "validation_agent": "📑 Validation agent",
    "expertise_agent": "🔍 Expertise agent",
    "notify_customer": "✉️ Customer notification",
    "__end__": "End",
}

#: Expected graph, displayed as long as `build_graph` is not implemented.
EXPECTED_EDGES = [
    ("__start__", "declaration_agent", False),
    ("declaration_agent", "validation_agent", True),
    ("declaration_agent", "notify_customer", True),
    ("validation_agent", "expertise_agent", True),
    ("validation_agent", "notify_customer", True),
    ("expertise_agent", "notify_customer", False),
    ("notify_customer", "__end__", False),
]

#: Labels of the conditional edges: the business rule leading to each branch.
EDGE_LABELS = {
    ("declaration_agent", "validation_agent"): "complete",
    ("declaration_agent", "notify_customer"): "incomplete /\nout of scope",
    ("validation_agent", "expertise_agent"): "send",
    ("validation_agent", "notify_customer"): "reject /\nadditional info",
}

COLORS = {"waiting": "#e9ecef", "running": "#ffd43b", "done": "#8ce99a", "error": "#ff8787"}

STATUSES = {
    "out_of_scope": ("⚪", "Out of scope"),
    "customer_follow_up": ("🟠", "Customer follow-up: incomplete declaration"),
    "additional_info_requested": ("🟠", "Additional information requested: non-compliant documents"),
    "rejected": ("🔴", "Rejected"),
    "forwarded_to_advisor": ("🟢", "Expertise done, forwarded to the advisor"),
}


@st.cache_resource
def load_graph():
    """Build the graph with a SQLite checkpointer (fast, no model call). Returns (graph, error).

    The graph is cached so that the checkpointer connection is shared by all the
    reruns of the page: an interrupted claim can be resumed from the sidebar.
    """
    try:
        from src.orchestration import build_graph, get_checkpointer

        return build_graph(get_checkpointer(ROOT / "checkpoints" / "app.sqlite")), None
    except NotImplementedError as error:
        return None, str(error)


def graph_edges(graph) -> list[tuple[str, str, bool]]:
    if graph is None:
        return EXPECTED_EDGES
    return [(e.source, e.target, e.conditional) for e in graph.get_graph().edges]


def dot(edges: list[tuple[str, str, bool]], node_states: dict[str, str], expected: bool = False) -> str:
    """Graph in DOT format: one node per agent, conditional edges dashed."""
    lines = [
        "digraph G {",
        '  rankdir=TB; bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=0];',
        '  edge [fontname="Helvetica", fontsize=9, color="#868e96"];',
    ]
    nodes = {n for e in edges for n in e[:2]}
    for node in NODES:
        if node not in nodes:
            continue
        shape = ', shape=ellipse' if node.startswith("__") else ""
        color = COLORS[node_states.get(node, "waiting")]
        lines.append(f'  {node} [label="{NODES[node]}", fillcolor="{color}"{shape}];')
    for source, target, conditional in edges:
        attributes = []
        if conditional:
            attributes.append("style=dashed")
        if (source, target) in EDGE_LABELS:
            attributes.append(f'label="{EDGE_LABELS[(source, target)]}"')
        lines.append(f"  {source} -> {target}" + (f" [{', '.join(attributes)}]" if attributes else "") + ";")
    if expected:
        lines.append('  labelloc=b; label="Expected graph (orchestration to implement: Step 4)"; fontsize=10;')
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rendering of each agent's results
# --------------------------------------------------------------------------- #
def yes_no(value) -> str:
    return "—" if value is None else ("✅ yes" if value else "❌ no")


def euros(value: float) -> str:
    return f"€{value:,.0f}"


def render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander(f"Tool calls ({len(trace)})"):
        for call in trace:
            icon = "❌" if call["error"] else "⚙️"
            arguments = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
            st.markdown(f"{icon} **`{call['tool']}`**`({arguments})`")
            st.code(call["result"][:1200], language="markdown")


def render_declaration(declaration, trace) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Claim type", CLAIM_TYPE_LABELS[declaration.claim_type])
    c2.metric("Loss date", str(declaration.loss_date or "unknown"))
    c3.metric("Complete", "yes" if declaration.complete else "no")
    st.markdown(f"**Location**: {declaration.location or '—'}  \n**Circumstances**: {declaration.circumstances}")
    if declaration.damaged_items:
        st.dataframe(
            pd.DataFrame([i.model_dump() for i in declaration.damaged_items]),
            hide_index=True, width="stretch",
        )
    st.markdown(
        f"Third party involved: {yes_no(declaration.third_party_involved)} · "
        f"Joint report: {yes_no(declaration.joint_report_provided)} · "
        f"Criminal suspicion: {yes_no(declaration.criminal_suspicion)} · "
        f"Police report: {yes_no(declaration.police_report_filed)} · "
        f"Inventory: {yes_no(declaration.inventory_provided)} · "
        f"Fire brigade: {yes_no(declaration.fire_brigade_called)}"
    )
    if declaration.missing_items:
        st.warning("**Completeness guardrail** — missing items: " + ", ".join(
            f"`{i}`" for i in declaration.missing_items))
    render_trace(trace)


def render_validation(validation, trace) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage", validation.coverage or "none")
    c2.metric("Cap", euros(validation.cap) if validation.cap is not None else "—")
    c3.metric("Deductible", euros(validation.deductible) if validation.deductible is not None else "—")
    c4.metric("Deadline", f"{validation.business_days_elapsed} / {validation.reporting_deadline_business_days} business days")
    st.markdown(
        f"Deadline met: {yes_no(validation.deadline_met)} · "
        f"Photos consistent: {yes_no(validation.photos_consistent)} · "
        f"Exclusion: {validation.exclusion or 'none'}"
    )
    st.markdown(f"**Decision**: `{validation.decision}` (LLM: `{validation.llm_decision}`)")
    for correction in validation.corrections:
        st.warning(f"**Consistency guardrail** — {correction}")
    st.caption(validation.rationale)
    render_trace(trace)


def render_expertise(report) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated cost", euros(report.total_cost))
    c2.metric("Deductible", euros(report.deductible))
    c3.metric("Payable amount", euros(report.payable_amount))
    with st.expander("Report for the advisor", expanded=True):
        st.markdown(report.report)


def render_notification(state) -> None:
    icon, label = STATUSES.get(state["status"], ("", state["status"]))
    st.markdown(f"### {icon} {label}")
    st.text_area("Message sent to the policyholder", state["customer_message"], height=220, disabled=True)


TITLES = {
    "declaration_agent": "🧾 Declaration agent",
    "validation_agent": "📑 Validation agent",
    "expertise_agent": "🔍 Expertise agent",
    "notify_customer": "✉️ Customer notification",
}


def render_node(node: str, state: dict, container) -> None:
    """Display the output of a node in its container."""
    traces = {e["agent"]: e["tools"] for e in state.get("journal", [])}
    with container:
        if node == "declaration_agent":
            render_declaration(state["declaration"], traces.get("Declaration", []))
        elif node == "validation_agent":
            render_validation(state["validation"], traces.get("Validation", []))
        elif node == "expertise_agent":
            render_expertise(state["expertise"])
        elif node == "notify_customer":
            render_notification(state)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def run_graph(graph, claim: Claim, graph_area, agents_area, thread_id: str, resume: bool = False) -> dict:
    """Run (or resume) the graph, **streaming** its events, and update the display live.

    Every event produced by `stream_claim` / `resume_claim` is shown as soon as
    it arrives: node started, tool called, tool result, node finished. The state
    is checkpointed after each node, so a crash can be resumed from the sidebar.
    """
    from src.orchestration import resume_claim, stream_claim

    state: dict = {"claim": claim, "journal": []}
    node_states = {"__start__": "done"}
    edges = graph_edges(graph)
    containers: dict = {}
    live: dict = {}
    live_lines: dict = {}
    start = time.time()
    graph_area.graphviz_chart(dot(edges, node_states))
    events = resume_claim(thread_id, graph) if resume else stream_claim(claim, thread_id, graph)
    try:
        for event in events:
            kind, node = event["type"], event.get("node")
            if kind == "node_started":
                node_states[node] = "running"
                containers[node] = agents_area.status(f"{TITLES[node]} — running…", expanded=True)
                live[node], live_lines[node] = containers[node].empty(), []
            elif kind in ("tool_call", "tool_result", "progress"):
                if kind == "tool_call":
                    arguments = ", ".join(f"{k}={v!r}" for k, v in event["arguments"].items())
                    live_lines[node].append(f"⚙️ `{event['tool']}({arguments})`")
                elif kind == "tool_result":
                    live_lines[node].append(f"&nbsp;&nbsp;{'❌' if event['error'] else '✓'} {event['result'][:200]}")
                else:
                    live_lines[node].append(f"… {event['message']}")
                live[node].markdown("  \n".join(live_lines[node]))
            elif kind == "node_finished":
                result = event["update"]
                state.update({k: v for k, v in result.items() if k != "journal"})
                state["journal"] = state["journal"] + result.get("journal", [])
                node_states[node] = "done"
                live[node].empty()
                render_node(node, state, containers[node])
                containers[node].update(
                    label=f"{TITLES[node]} — {time.time() - start:.1f} s", state="complete",
                    expanded=node in ("expertise_agent", "notify_customer"),
                )
            elif kind == "done":
                state = {**event["state"], "journal": event["state"].get("journal", [])}
            graph_area.graphviz_chart(dot(edges, node_states))
        node_states["__end__"] = "done"
        st.session_state.pop("interrupted", None)
    except Exception as error:  # noqa: BLE001 - displayed as is in the interface
        running = [n for n, s in node_states.items() if s == "running"]
        for node in running:
            node_states[node] = "error"
            containers[node].update(label=f"{TITLES[node]} — error", state="error")
        agents_area.error(f"**{type(error).__name__}**: {error}")
        agents_area.info(f"Progress saved (thread `{thread_id}`). Use **Resume interrupted claim** in the sidebar.")
        st.session_state["interrupted"] = (claim.id, thread_id)
    graph_area.graphviz_chart(dot(edges, node_states))
    state["_node_states"] = node_states
    return state


def run_without_graph(claim: Claim, graph_area, agents_area) -> dict:
    """Fallback until the orchestration is written: agents are chained without routing.

    Warning: without orchestration, **no process rule** is applied. An
    incomplete or rejected file still goes to expertise.
    """
    from src.agents.declaration import run_declaration
    from src.agents.expertise import run_expertise
    from src.agents.validation import run_validation

    agents_area.info(
        "Orchestration not implemented (Step 4): the agents are chained **without routing**. "
        "An incomplete or rejected file still goes to expertise."
    )
    state: dict = {"claim": claim, "journal": []}
    node_states = {"__start__": "done"}
    graph_area.graphviz_chart(dot(EXPECTED_EDGES, node_states, expected=True))
    steps = [
        ("declaration_agent", "Declaration", lambda: run_declaration(claim), "declaration"),
        ("validation_agent", "Validation", lambda: run_validation(state["declaration"]), "validation"),
        ("expertise_agent", "Expertise", lambda: (run_expertise(state["declaration"], state["validation"]), []), "expertise"),
    ]
    for node, agent, call, key in steps:
        container = agents_area.status(f"{TITLES[node]} — running…", expanded=True)
        node_states[node] = "running"
        graph_area.graphviz_chart(dot(EXPECTED_EDGES, node_states, expected=True))
        try:
            result, trace = call()
        except Exception as error:  # noqa: BLE001
            node_states[node] = "error"
            container.update(label=f"{TITLES[node]} — {type(error).__name__}", state="error")
            container.error(f"**{type(error).__name__}**: {error}")
            break
        state[key] = result
        state["journal"].append({"agent": agent, "tools": trace, "summary": ""})
        node_states[node] = "done"
        render_node(node, state, container)
        container.update(label=TITLES[node], state="complete", expanded=False)
    graph_area.graphviz_chart(dot(EXPECTED_EDGES, node_states, expected=True))
    state["_node_states"] = node_states
    return state


def redisplay(state: dict, graph, graph_area, agents_area) -> None:
    """Display a processing that was already done again (after a tab change, for example)."""
    expected = graph is None
    graph_area.graphviz_chart(dot(graph_edges(graph), state["_node_states"], expected=expected))
    present = {
        "declaration_agent": "declaration", "validation_agent": "validation",
        "expertise_agent": "expertise", "notify_customer": "status",
    }
    for node in TITLES:
        if present[node] in state:
            container = agents_area.status(TITLES[node], state="complete",
                                           expanded=node in ("expertise_agent", "notify_customer"))
            render_node(node, state, container)


def history_for(thread_id: str | None, graph) -> None:
    """Show the checkpoints of the last processing (the audit trail, step 5)."""
    if not thread_id or graph is None:
        return
    from src.orchestration import claim_history

    with st.expander(f"Checkpoints — thread {thread_id}"):
        st.dataframe(
            pd.DataFrame(claim_history(thread_id, graph)).astype(str), hide_index=True, width="stretch"
        )


# --------------------------------------------------------------------------- #
# Sidebar: the claim to process
# --------------------------------------------------------------------------- #
claims = load_claims()
available_photos = sorted(p.name for p in ATTACHMENTS_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

with st.sidebar:
    st.title("🏠 HomeShield")
    st.caption("Multi-agent claims handling system")
    choice = st.selectbox(
        "Claim",
        list(claims),
        format_func=lambda i: f"{i} — {claims[i].message.replace('Hello,', '').strip()[:45]}…",
    )
    template = claims[choice]
    received_date = st.date_input("Received date", template.received_date, key=f"date_{choice}")
    message = st.text_area("Policyholder's message", template.message, height=220, key=f"msg_{choice}")
    attachments = st.multiselect("Attachments", available_photos, template.attachments, key=f"att_{choice}")
    uploaded = st.file_uploader("Add photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    addendum = st.text_area(
        "Policyholder's addendum (reply to a follow-up)", "", height=90, key=f"add_{choice}",
        placeholder="E.g.: I filled in the joint report with my neighbor, I am attaching it.",
    )
    run = st.button("▶ Process the claim", type="primary", width="stretch")
    interrupted = st.session_state.get("interrupted")
    resume = False
    if interrupted:
        st.warning(f"Claim {interrupted[0]} was interrupted.")
        resume = st.button("↻ Resume interrupted claim", width="stretch")

for file in uploaded or []:
    name = f"upload_{file.name}"
    (ATTACHMENTS_DIR / name).write_bytes(file.getvalue())
    if name not in attachments:
        attachments.append(name)

text = message + (f"\n\nPolicyholder's addendum ({received_date}):\n{addendum}" if addendum.strip() else "")
claim = Claim(id=choice, received_date=received_date, message=text, attachments=attachments)

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
processing_tab, evaluation_tab, docs_tab = st.tabs(["⚙️ Processing", "📊 Evaluation", "📚 Documentation"])
graph, graph_error = load_graph()

with processing_tab:
    graph_col, agents_col = st.columns([1, 2.2])
    with graph_col:
        st.subheader("Workflow")
        graph_area = st.empty()
        if attachments:
            st.subheader("Attachments")
            st.image([str(ATTACHMENTS_DIR / p) for p in attachments], caption=attachments, width=150)
    with agents_col:
        st.subheader(f"Claim {claim.id} — received on {claim.received_date:%m/%d/%Y}")
        with st.expander("Policyholder's message", expanded=not run):
            st.text(claim.message)
        agents_area = st.container()

    if resume and graph is not None:
        state = run_graph(graph, claim, graph_area, agents_area, thread_id=interrupted[1], resume=True)
        st.session_state["last"] = (claim.id, state)
    elif run:
        if graph is not None:
            thread_id = f"{claim.id}-{uuid.uuid4().hex[:8]}"
            state = run_graph(graph, claim, graph_area, agents_area, thread_id=thread_id)
            state["_thread_id"] = thread_id
            history_for(thread_id, graph)
        else:
            state = run_without_graph(claim, graph_area, agents_area)
        st.session_state["last"] = (claim.id, state)
    elif st.session_state.get("last", (None,))[0] == claim.id:
        redisplay(st.session_state["last"][1], graph, graph_area, agents_area)
        history_for(st.session_state["last"][1].get("_thread_id"), graph)
    else:
        edges = graph_edges(graph)
        graph_area.graphviz_chart(dot(edges, {}, expected=graph is None))
        agents_area.info("Pick a claim in the left panel, then **Process the claim**.")

with evaluation_tab:
    st.subheader("Evaluation on the golden datasets")
    st.caption("No threshold is set: these metrics are a basis for discussion with the business.")
    col_d, col_v = st.columns(2)

    def run_evaluation(name: str, function, column) -> None:
        bar = column.progress(0.0, text="Evaluation in progress…")
        try:
            metrics, cases = function(progress=lambda i, n: bar.progress(i / n, text=f"{i}/{n} cases"))
        except NotImplementedError as error:
            column.error(f"To implement: {error}")
            return
        st.session_state[f"eval_{name}"] = (metrics, cases)

    with col_d:
        st.markdown("#### 🧾 Declaration agent — completeness")
        if st.button("Evaluate the Declaration agent", width="stretch"):
            from src.evaluation import evaluate_declaration

            run_evaluation("declaration", evaluate_declaration, col_d)
    with col_v:
        st.markdown("#### 📑 Validation agent — policy facts")
        if st.button("Evaluate the Validation agent", width="stretch"):
            from src.evaluation import evaluate_validation

            run_evaluation("validation", evaluate_validation, col_v)

    for name, column, main_metrics in [
        ("declaration", col_d, ["missing_items_recall", "missing_items_precision", "type_accuracy", "date_accuracy"]),
        ("validation", col_v, ["policy_fact_accuracy", "decision_accuracy", "llm_decision_accuracy", "photo_accuracy"]),
    ]:
        if f"eval_{name}" not in st.session_state:
            continue
        metrics, cases = st.session_state[f"eval_{name}"]
        with column:
            cols = st.columns(len(main_metrics))
            for c, key in zip(cols, main_metrics):
                c.metric(key.replace("_", " "), f"{metrics[key]:.0%}")
            st.dataframe(pd.Series(metrics, name="value").to_frame(), width="stretch")
            st.dataframe(pd.DataFrame(cases), hide_index=True, width="stretch")

with docs_tab:
    c1, c2 = st.columns(2)
    with c1, st.container(height=600):
        st.markdown(process_text())
    with c2, st.container(height=600):
        st.markdown(policy_text())
    with st.expander("Golden dataset — Declaration"):
        st.dataframe(pd.DataFrame(load_jsonl("declaration.jsonl")), hide_index=True, width="stretch")
    with st.expander("Golden dataset — Validation"):
        golden = [{"id": c["id"], "comment": c["comment"], **c["expected"]} for c in load_jsonl("validation.jsonl")]
        st.dataframe(pd.DataFrame(golden), hide_index=True, width="stretch")

if graph_error:
    st.sidebar.warning(f"Orchestration: {graph_error}")
