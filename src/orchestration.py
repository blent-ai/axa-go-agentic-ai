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
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

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
def declaration_agent(state: ClaimState) -> dict:
    """Declaration agent node: structure the claim and check its completeness."""
    declaration, trace = run_declaration(state["claim"])
    return {
        "declaration": declaration,
        "journal": [{"agent": "Declaration", "tools": trace, "summary": _declaration_summary(declaration)}],
    }


def validation_agent(state: ClaimState) -> dict:
    """Validation agent node: compare the declaration with the policy."""
    validation, trace = run_validation(state["declaration"])
    return {
        "validation": validation,
        "journal": [{"agent": "Validation", "tools": trace, "summary": f"Decision: {validation.decision}"}],
    }


def expertise_agent(state: ClaimState) -> dict:
    """Expertise agent node: estimate the damage and the payable amount."""
    report = run_expertise(state["declaration"], state["validation"])
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
