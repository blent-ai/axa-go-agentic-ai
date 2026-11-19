"""Declaration AI agent: guide the policyholder through their declaration.

The agent receives a policyholder's free-text message and attachments. It must:

1. **identify** the claim type and its date (even a relative one: "last night");
2. **extract** the facts useful for the rest of the process (third party
   involved, police report, inventory of items...);
3. **check completeness**: are all the items required by the process
   (``data/claims_process.md``) present?

It does **not** judge coverage: that is the Validation agent's job.

Architecture: a ReAct agent (`create_agent`) with two tools that returns a
structured output (`ClaimInfo`), followed by a **deterministic guardrail**
(`completeness_check`) that applies the process rules. The LLM extracts facts
from the text; the code applies the business rules.
"""

from __future__ import annotations

from datetime import date

from langchain.agents import create_agent
from langchain_core.tools import ToolException, tool

from ..config import get_chat_model
from ..knowledge import WEEKDAYS, process_for_type, shift_date
from ..schemas import Claim, ClaimDeclaration, ClaimInfo
from .common import extract_trace, middlewares

DECLARATION_PROMPT = """\
You are the Declaration agent of HomeShield, a home insurance company.
You receive the free-text message of a policyholder who reports a claim.

Your mission: extract from this message, factually, all the information needed
to handle the file.

Approach:
1. Identify the claim family. Only three families are handled: water damage,
   fire / explosion, theft / burglary / vandalism. Any other claim (storm,
   natural disaster, glass breakage...) is "out_of_scope".
2. Consult the process for this claim type (consult_process tool) to know what
   the policyholder must provide.
3. Determine the loss date. If it is relative ("yesterday", "this morning",
   "Saturday"), compute it with the compute_date tool from the received date:
   never do date arithmetic in your head. If no date can be deduced, leave it
   empty.
4. Note what the policyholder says they did or provided (police report, joint
   report, inventory of items with their value...).

Rules:
- Do not invent anything: an item that is not mentioned is considered not provided.
- Do not rule on acceptance or on policy coverage.
"""


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool(parse_docstring=True)
def consult_process(claim_type: str) -> str:
    """Return the handling process of a claim type: what the policyholder must provide, and within what deadlines.

    Args:
        claim_type: "water_damage", "fire_explosion" or "theft_vandalism".

    Returns:
        The matching section of the process document, in Markdown.
    """
    try:
        return process_for_type(claim_type)
    except KeyError as error:
        raise ToolException(str(error)) from error


@tool(parse_docstring=True)
def compute_date(reference_date: str, offset_days: int = 0, weekday: str | None = None) -> dict:
    """Compute a relative date ("yesterday", "the day before yesterday", "last Saturday") from a reference date.

    Examples: yesterday = offset_days=-1; this morning = offset_days=0;
    "Saturday" = weekday="saturday" (the last Saturday, at the latest on the reference date).

    Args:
        reference_date: Reference date in YYYY-MM-DD format, usually the received date.
        offset_days: Number of days to add (negative to go back in time).
        weekday: Name of a day of the week ("monday"... "sunday"), optional.

    Returns:
        The computed date (YYYY-MM-DD) and the corresponding day of the week.
    """
    try:
        result = shift_date(date.fromisoformat(reference_date), offset_days, weekday)
    except ValueError as error:
        raise ToolException(str(error)) from error
    return {"date": result.isoformat(), "weekday": WEEKDAYS[result.weekday()]}


TOOLS = [consult_process, compute_date]


# --------------------------------------------------------------------------- #
# Completeness guardrail
# --------------------------------------------------------------------------- #
#: Labels of the missing items, for the follow-up message.
ITEM_LABELS = {
    "loss_date": "the date on which the loss occurred",
    "photos": "photos or videos of the damage",
    "joint_report": "a joint water damage report filled in with the third party concerned",
    "police_report": "the receipt of your police report",
    "inventory": "the inventory of the damaged or stolen items, with their value (invoices, photos)",
}


def missing_items(info: ClaimInfo, attachments: list[str]) -> list[str]:
    """List the items required by the process that are missing from the declaration.

    Rules taken from ``data/claims_process.md``:

    =================== ==========================================================
    All types            the loss date
    Water damage         photos; a joint report if a third party is involved
    Fire / explosion     photos; the inventory of items; a police report if
                         criminal suspicion
    Theft / vandalism    the police report; the inventory of items
    =================== ==========================================================

    Photos are counted from the attachments **actually provided**, never from
    what the LLM says: models readily hallucinate documents.

    Parameters
    ----------
    info : ClaimInfo
        The facts extracted by the LLM.
    attachments : list[str]
        The files actually attached to the claim.

    Returns
    -------
    list[str]
        The missing items, in the order of the table above.
    """
    # TODO (Step 1) — Apply the rules of the table above.
    #   - out of scope: no item required (empty list);
    #   - all types: "loss_date" if info.loss_date is None;
    #   - then, depending on info.claim_type, "photos" (if `attachments` is
    #     empty), "joint_report", "police_report", "inventory", relying on the
    #     booleans extracted by the LLM (third_party_involved,
    #     joint_report_provided, criminal_suspicion, police_report_filed,
    #     inventory_provided).
    # Respect the order of the table: the tests compare lists.
    raise NotImplementedError("Step 1 — missing_items")


def follow_up_message(missing: list[str]) -> str:
    """Write the follow-up message sent to the policyholder (template, no LLM)."""
    bullets = "\n".join(f"- {ITEM_LABELS[item]}" for item in missing)
    return (
        "Hello,\n\nWe have received your claim declaration. To continue "
        f"processing it, please send us:\n{bullets}\n\n"
        "Kind regards,\nHomeShield"
    )


def completeness_check(info: ClaimInfo, claim: Claim) -> ClaimDeclaration:
    """Deterministic guardrail: complete the declaration with the missing items.

    Parameters
    ----------
    info : ClaimInfo
        The facts extracted by the LLM.
    claim : Claim
        The original claim (identifier, received date, attachments).

    Returns
    -------
    ClaimDeclaration
        The complete declaration: LLM facts + ``missing_items``, ``complete``
        and, if needed, ``follow_up_message``.
    """
    missing = missing_items(info, claim.attachments)
    return ClaimDeclaration(
        **info.model_dump(),
        claim_id=claim.id,
        received_date=claim.received_date,
        attachments=claim.attachments,
        missing_items=missing,
        complete=not missing,
        follow_up_message=follow_up_message(missing) if missing else None,
    )


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
def build_declaration_agent(model=None):
    """Build the Declaration ReAct agent.

    Parameters
    ----------
    model : BaseChatModel, optional
        Model to use. Defaults to the LiteLLM gateway one.

    Returns
    -------
    CompiledStateGraph
        An agent whose final answer is a `ClaimInfo`
        (``structured_response`` key of the result).
    """
    # TODO (Step 1) — Build the agent with `create_agent`:
    #   - model: `model`, or `get_chat_model()` by default;
    #   - tools: TOOLS;
    #   - system_prompt: DECLARATION_PROMPT;
    #   - response_format: ClaimInfo (the expected structured output);
    #   - middleware: middlewares() (tool errors, call limit).
    raise NotImplementedError("Step 1 — build_declaration_agent")


def format_claim(claim: Claim) -> str:
    """Format the claim for the LLM: received date, attachments, message."""
    day = WEEKDAYS[claim.received_date.weekday()]
    attachments = ", ".join(claim.attachments) or "none"
    return (
        f"Received date: {claim.received_date.isoformat()} ({day})\n"
        f"Attachments: {attachments}\n\n"
        f"Policyholder's message:\n{claim.message}"
    )


def run_declaration(claim: Claim, model=None) -> tuple[ClaimDeclaration, list[dict]]:
    """Run the Declaration agent on a claim.

    Returns
    -------
    tuple[ClaimDeclaration, list[dict]]
        The checked declaration, and the trace of tool calls.
    """
    agent = build_declaration_agent(model)
    result = agent.invoke({"messages": [{"role": "user", "content": format_claim(claim)}]})
    declaration = completeness_check(result["structured_response"], claim)
    return declaration, extract_trace(result["messages"], TOOLS)
