"""Agent orchestration: the claim processing workflow.

The three agents are orchestrated by a **LangGraph graph**: each agent is a
node, and **conditional edges** encode the business process.

```
START → declaration_agent ─┬─ complete ─→ validation_agent ─┬─ send ─→ expertise_agent ─┐
                           │                                 │                           │
                           └─ incomplete / out of scope ─→ notify_customer ←─ reject / additional info ┘
                                                                  ↓
                                                                 END
```

Why a workflow rather than an LLM supervisor that would freely choose the next
agent? In a regulated process, the sequence of steps is **known in advance**: a
rejected file must never go to expertise. We keep LLMs where they add value
(reading a free-text message, a policy, a photo), and the code decides the
routing: it is deterministic, testable and auditable.

The shared state (`ClaimState`) is what lets each agent have all the information
produced by the previous agents.

Step 5 adds two production concerns on top of the workflow:

- **Streaming**: `stream_claim` yields typed events *while* the claim is being
  processed (node started / tool called / node finished), so a UI or a log can
  follow the progress instead of waiting for the final result.
- **Checkpointing**: the graph state is saved after every node. A claim is
  identified by a ``thread_id``; if a node crashes (gateway timeout, rate
  limit...), `resume_claim` restarts from the last checkpoint and replays only
  the failed node, without paying again for the agents that already succeeded.
  With a SQLite checkpointer, this survives a process restart.
"""

from __future__ import annotations

import operator
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .agents.declaration import run_declaration
from .agents.expertise import run_expertise
from .agents.validation import run_validation
from .schemas import Claim, ClaimDeclaration, ExpertiseReport, Status, ValidationDecision


class ClaimState(TypedDict, total=False):
    """The claim state, shared by all the nodes of the graph.

    Each node returns a dictionary with only the keys it updates. ``journal``
    has a *reducer* (`operator.add`): the entries returned by successive nodes
    are **appended** instead of overwriting each other.
    """

    claim: Claim
    declaration: ClaimDeclaration
    validation: ValidationDecision
    expertise: ExpertiseReport
    status: Status
    customer_message: str
    journal: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------- #
# The nodes: one per agent, plus the policyholder notification
# --------------------------------------------------------------------------- #
def _emitter(node: str):
    """Return a callback that sends an agent's progress events to the *custom* stream.

    ``get_stream_writer`` is the LangGraph way for a node to publish events
    while it is still running. When the graph is run with ``invoke`` (no custom
    stream requested), the writer silently does nothing.
    """
    writer = get_stream_writer()
    return lambda event: writer({**event, "node": node})


def declaration_agent(state: ClaimState) -> dict:
    """Declaration agent node: structure the claim and check its completeness."""
    declaration, trace = run_declaration(state["claim"], on_event=_emitter("declaration_agent"))
    return {
        "declaration": declaration,
        "journal": [{"agent": "Declaration", "tools": trace, "summary": _declaration_summary(declaration)}],
    }


def validation_agent(state: ClaimState) -> dict:
    """Validation agent node: compare the declaration with the policy."""
    validation, trace = run_validation(state["declaration"], on_event=_emitter("validation_agent"))
    return {
        "validation": validation,
        "journal": [{"agent": "Validation", "tools": trace, "summary": f"Decision: {validation.decision}"}],
    }


def expertise_agent(state: ClaimState) -> dict:
    """Expertise agent node: estimate the damage and the payable amount."""
    report = run_expertise(state["declaration"], state["validation"], on_event=_emitter("expertise_agent"))
    return {
        "expertise": report,
        "journal": [
            {
                "agent": "Expertise",
                "tools": [],
                "summary": f"Estimated cost €{report.total_cost:.0f}, payable €{report.payable_amount:.0f}",
            }
        ],
    }


def notify_customer(state: ClaimState) -> dict:
    """Last node: determine the file status and the message for the policyholder.

    The message is a template: no decision is worded by an LLM for the
    policyholder.
    """
    declaration = state["declaration"]
    validation = state.get("validation")

    if declaration.claim_type == "out_of_scope":
        status = "out_of_scope"
        message = (
            "Your declaration is not a water damage, a fire or a theft claim. "
            "A HomeShield advisor will contact you."
        )
    elif not declaration.complete:
        status, message = "customer_follow_up", declaration.follow_up_message
    elif validation.decision == "reject":
        status = "rejected"
        message = f"Unfortunately, we are unable to process your declaration. Reason: {validation.rationale}"
    elif validation.decision == "request_additional_info":
        status = "additional_info_requested"
        message = f"Additional documents are needed to process your file. {validation.rationale}"
    else:
        status = "forwarded_to_advisor"
        message = (
            "Your file is complete and covered by your policy. A first estimate "
            "has been sent to your advisor, who will get back to you within 48 hours."
        )
    return {
        "status": status,
        "customer_message": message,
        "journal": [{"agent": "Notification", "tools": [], "summary": f"Status: {status}"}],
    }


def _declaration_summary(declaration: ClaimDeclaration) -> str:
    missing = ", ".join(declaration.missing_items) or "none"
    return f"{declaration.claim_type}, on {declaration.loss_date} — missing items: {missing}"


# --------------------------------------------------------------------------- #
# The routing: the business process
# --------------------------------------------------------------------------- #
def after_declaration(state: ClaimState) -> Literal["validation_agent", "notify_customer"]:
    """After the declaration: only a complete, in-scope file is validated."""
    declaration = state["declaration"]
    if declaration.claim_type == "out_of_scope" or not declaration.complete:
        return "notify_customer"
    return "validation_agent"


def after_validation(state: ClaimState) -> Literal["expertise_agent", "notify_customer"]:
    """After the validation: only an accepted file goes to expertise."""
    if state["validation"].decision == "send_to_expertise":
        return "expertise_agent"
    return "notify_customer"


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
def build_graph(checkpointer=None):
    """Assemble the multi-agent workflow.

    Parameters
    ----------
    checkpointer : BaseCheckpointSaver, optional
        Persistence of the graph state (not used by default).

    Returns
    -------
    CompiledStateGraph
        The compiled graph, invocable via ``invoke({"claim": ...})`` or ``stream``.
    """
    graph = StateGraph(ClaimState)

    graph.add_node("declaration_agent", declaration_agent)
    graph.add_node("validation_agent", validation_agent)
    graph.add_node("expertise_agent", expertise_agent)
    graph.add_node("notify_customer", notify_customer)

    graph.add_edge(START, "declaration_agent")
    graph.add_conditional_edges("declaration_agent", after_declaration, ["validation_agent", "notify_customer"])
    graph.add_conditional_edges("validation_agent", after_validation, ["expertise_agent", "notify_customer"])
    graph.add_edge("expertise_agent", "notify_customer")
    graph.add_edge("notify_customer", END)

    return graph.compile(checkpointer=checkpointer)


def process(claim: Claim) -> ClaimState:
    """Process a claim end to end and return the final file state."""
    return build_graph().invoke({"claim": claim, "journal": []})


# --------------------------------------------------------------------------- #
# Step 5 — Checkpointing
# --------------------------------------------------------------------------- #
def _serializer() -> JsonPlusSerializer:
    """Serializer for the checkpoints, allowed to rebuild our Pydantic schemas.

    The state contains our business objects (`ClaimDeclaration`...). LangGraph
    only deserializes types it has been told about, so we list the ones from
    ``src.schemas``.
    """
    from . import schemas

    allowed = [(schemas.__name__, name) for name, obj in vars(schemas).items()
               if isinstance(obj, type) and issubclass(obj, schemas.BaseModel) and obj is not schemas.BaseModel]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


def get_checkpointer(path: str | Path | None = None):
    """Create the checkpointer that persists the graph state.

    Parameters
    ----------
    path : str or Path, optional
        Path of a SQLite database. The state then survives a restart of the
        process. Without a path, the state is kept in memory only (lost when
        the process exits): fine for tests and demos.

    Returns
    -------
    BaseCheckpointSaver
        A `SqliteSaver` or an `InMemorySaver`, to pass to `build_graph`.
    """
    if path is None:
        return InMemorySaver(serde=_serializer())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: Streamlit and the evaluation use several threads.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=_serializer())


def _config(thread_id: str) -> dict:
    """The run configuration: the ``thread_id`` identifies the claim's checkpoints."""
    return {"configurable": {"thread_id": thread_id}}


def claim_history(thread_id: str, graph) -> list[dict]:
    """List the checkpoints of a claim, most recent first.

    Every node writes a checkpoint when it finishes: this is the audit trail of
    the processing, and the list of points from which it can be resumed.

    Returns
    -------
    list[dict]
        ``[{"checkpoint_id", "step", "next", "status", "created_at"}, ...]``.
        ``next`` is the tuple of nodes still to run (empty when the claim is done).
    """
    return [
        {
            "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
            "step": snapshot.metadata.get("step"),
            "next": snapshot.next,
            "status": snapshot.values.get("status"),
            "created_at": snapshot.created_at,
        }
        for snapshot in graph.get_state_history(_config(thread_id))
    ]


# --------------------------------------------------------------------------- #
# Step 5 — Streaming
# --------------------------------------------------------------------------- #
def _stream(graph, graph_input, thread_id: str) -> Iterator[dict]:
    """Run (or resume) the graph and translate LangGraph chunks into simple events.

    Two stream modes are combined: ``tasks`` (a node starts / a node ends) and
    ``custom`` (what the nodes publish with the stream writer: tool calls...).
    When ``graph_input`` is ``None``, LangGraph resumes from the last checkpoint.

    Events
    ------
    ``{"type": "node_started", "node"}``
    ``{"type": "tool_call" | "tool_result" | "progress", "node", ...}``
    ``{"type": "node_finished", "node", "update"}``
    ``{"type": "node_error", "node", "error"}``, then the exception is raised
    ``{"type": "done", "thread_id", "state"}``
    """
    config = _config(thread_id)
    for mode, chunk in graph.stream(graph_input, config, stream_mode=["tasks", "custom"]):
        if mode == "custom":
            yield chunk
        elif "result" not in chunk and "error" not in chunk:
            yield {"type": "node_started", "node": chunk["name"]}
        elif chunk.get("error"):
            yield {"type": "node_error", "node": chunk["name"], "error": chunk["error"]}
        else:
            yield {"type": "node_finished", "node": chunk["name"], "update": dict(chunk["result"] or {})}
    yield {"type": "done", "thread_id": thread_id, "state": graph.get_state(config).values}


def stream_claim(claim: Claim, thread_id: str | None = None, graph=None) -> Iterator[dict]:
    """Process a claim and **stream** the progress as events.

    Parameters
    ----------
    claim : Claim
        The claim to process.
    thread_id : str, optional
        Identifier of the processing, used to find its checkpoints later. By
        default, a unique identifier derived from the claim id. Reusing the same
        ``thread_id`` for a new run would append to the previous state.
    graph : CompiledStateGraph, optional
        A graph compiled with a checkpointer (`build_graph(get_checkpointer(...))`).
        By default, a graph with an in-memory checkpointer.

    Yields
    ------
    dict
        The events described in `_stream`. The last one is ``done``.
    """
    graph = graph or build_graph(get_checkpointer())
    thread_id = thread_id or f"{claim.id}-{uuid.uuid4().hex[:8]}"
    yield from _stream(graph, {"claim": claim, "journal": []}, thread_id)


def resume_claim(thread_id: str, graph) -> Iterator[dict]:
    """Resume an interrupted claim from its last checkpoint.

    Only the nodes that did not complete are run again: the work already done
    (and paid for) by the previous agents is read from the checkpoint.

    Parameters
    ----------
    thread_id : str
        The identifier of the interrupted processing.
    graph : CompiledStateGraph
        A graph compiled with the **same checkpointer** as the interrupted run.

    Yields
    ------
    dict
        The same events as `stream_claim`.
    """
    if not graph.get_state(_config(thread_id)).next:
        raise ValueError(f"Nothing to resume for thread '{thread_id}': unknown or already finished.")
    yield from _stream(graph, None, thread_id)
