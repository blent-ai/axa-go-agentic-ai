"""Validation AI agent: check the policy coverage.

From the `ClaimDeclaration` produced by the Declaration agent, the Validation
agent:

1. **checks the documents**: does each photo match the declared claim?
2. **compares the claim with the policy** (``data/home_insurance_policy.md``):
   applicable coverage, cap, deductible, exclusions, reporting deadline;
3. **decides what happens next**: ``send_to_expertise``,
   ``request_additional_info`` or ``reject``.

Architecture: a ReAct agent (`create_agent`) with three tools and a structured
output (`ValidationAnalysis`), followed by a deterministic **consistency
guardrail**: a missed deadline cannot lead to an expert assessment, whatever the
LLM says.

This is the agent we evaluate on its ability to use **the right factual
information from the policy** (coverage, cap, deductible, deadline).
"""

from __future__ import annotations

import json
from datetime import date

from langchain.agents import create_agent
from langchain_core.tools import ToolException, tool

from ..config import get_chat_model, get_vision_model
from ..knowledge import CLAIM_TYPE_LABELS, image_block, normalize, policy_sections
from ..knowledge import business_days_between as count_business_days
from ..schemas import ClaimDeclaration, PhotoConsistency, ValidationAnalysis, ValidationDecision
from .common import extract_trace, middlewares, run_agent

VALIDATION_PROMPT = """\
You are the Validation agent of HomeShield, a home insurance company.
You receive a claim declaration already structured by the Declaration agent.

Your mission: check the file against the policyholder's policy, then decide
what happens next.

Approach:
1. Check each attached photo with the verify_photo tool.
2. Consult in the policy (consult_policy tool) the coverage that applies to this
   claim, the general exclusions and the policyholder's obligations (reporting
   deadlines).
3. Compute the business days elapsed between the loss date and the received
   date with the business_days_between tool: never count in your head.
4. Decide:
   - "reject" if no coverage applies, if an exclusion applies, or if the
     reporting deadline was missed;
   - "request_additional_info" if a photo does not match the declared claim;
   - "send_to_expertise" otherwise.

Rules:
- Copy the amounts and deadlines as they appear in the policy. Never give a
  figure you did not read in the policy.
- In the rationale, cite the policy elements you rely on.
"""


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool(parse_docstring=True)
def consult_policy(section: str) -> str:
    """Return a section of the policyholder's home insurance policy.

    Available sections: the coverages ("Water Damage", "Fire / Explosion",
    "Theft / Vandalism", "Glass Breakage", "Natural Disasters", "Third-Party
    Liability"), "General Exclusions", "Policyholder Obligations", "Compensation
    Procedure" and "Header" (policy number, policyholder).

    Args:
        section: Title of the section, or a keyword of this title ("fire", "exclusions").

    Returns:
        The exact title of the section and its content in Markdown.
    """
    sections = policy_sections()
    key = normalize(section)
    for title, content in sections.items():
        if key and (key in normalize(title) or normalize(title) in key):
            return f"## {title}\n\n{content}"
    raise ToolException(
        f"Section \"{section}\" not found. Available sections: {', '.join(sections)}."
    )


@tool(parse_docstring=True)
def business_days_between(start_date: str, end_date: str) -> dict:
    """Count the business days elapsed between two dates (weekends and public holidays excluded).

    Args:
        start_date: Loss date, in YYYY-MM-DD format.
        end_date: Received date of the declaration, in YYYY-MM-DD format.

    Returns:
        The number of business days elapsed after start_date, up to and including end_date.
    """
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError as error:
        raise ToolException(f"Invalid date: {error}. Expected format: YYYY-MM-DD.") from error
    return {"start_date": start_date, "end_date": end_date, "business_days": count_business_days(start, end)}


@tool(parse_docstring=True)
def verify_photo(filename: str, claim_type: str) -> dict:
    """Verify with a vision model that an attached photo matches the declared claim type.

    Args:
        filename: Name of the attached file, for example "WaterDamage_100.jpg".
        claim_type: "water_damage", "fire_explosion" or "theft_vandalism".

    Returns:
        The consistency of the photo with the claim and a description of what it shows.
    """
    try:
        image = image_block(filename)
    except FileNotFoundError as error:
        raise ToolException(str(error)) from error
    question = (
        f"A policyholder reports a claim of type \"{CLAIM_TYPE_LABELS.get(claim_type, claim_type)}\". "
        "Does this photo show damage compatible with this type of claim? "
        "First describe what you see, then answer."
    )
    model = get_vision_model().with_structured_output(PhotoConsistency)
    response = model.invoke([{"role": "user", "content": [{"type": "text", "text": question}, image]}])
    return {"photo": filename, **response.model_dump()}


TOOLS = [consult_policy, business_days_between, verify_photo]


# --------------------------------------------------------------------------- #
# Consistency guardrail
# --------------------------------------------------------------------------- #
def consistency_check(analysis: ValidationAnalysis, declaration: ClaimDeclaration) -> ValidationDecision:
    """Deterministic guardrail: the decision must be consistent with the findings.

    1. The business days elapsed are **recomputed** by the code, and compliance
       with the deadline is deduced from them (with the contractual deadline
       read by the LLM).
    2. Consistency rules, applied in this order:
       - no applicable coverage, an exclusion, or a missed deadline
         → ``reject``;
       - otherwise, a non-compliant photo cannot lead to an expert assessment
         → ``request_additional_info``.

    Every correction is traced in ``corrections``: we keep the LLM's initial
    decision (``llm_decision``) for audit.

    Parameters
    ----------
    analysis : ValidationAnalysis
        The LLM's structured output.
    declaration : ClaimDeclaration
        The validated declaration (loss and received dates).

    Returns
    -------
    ValidationDecision
        The analysis, possibly corrected, with the final decision.
    """
    data = analysis.model_dump()
    corrections = []

    if declaration.loss_date is not None:
        days = count_business_days(declaration.loss_date, declaration.received_date)
        if days != analysis.business_days_elapsed:
            corrections.append(f"Business days elapsed recomputed: {analysis.business_days_elapsed} → {days}.")
        data["business_days_elapsed"] = days
        if analysis.reporting_deadline_business_days is not None:
            data["deadline_met"] = days <= analysis.reporting_deadline_business_days

    decision = analysis.decision
    if analysis.coverage is None or analysis.exclusion or not data["deadline_met"]:
        reason = (
            "no applicable coverage" if analysis.coverage is None
            else f"exclusion \"{analysis.exclusion}\"" if analysis.exclusion
            else "reporting deadline missed"
        )
        if decision != "reject":
            corrections.append(f"Decision {decision} → reject: {reason}.")
        decision = "reject"
    elif not analysis.photos_consistent and decision == "send_to_expertise":
        corrections.append("Decision send_to_expertise → request_additional_info: non-compliant photo.")
        decision = "request_additional_info"

    data["decision"] = decision
    return ValidationDecision(
        **data,
        claim_id=declaration.claim_id,
        llm_decision=analysis.decision,
        corrections=corrections,
    )


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
def build_validation_agent(model=None):
    """Build the Validation ReAct agent (output: `ValidationAnalysis`)."""
    return create_agent(
        model=model or get_chat_model(),
        tools=TOOLS,
        system_prompt=VALIDATION_PROMPT,
        response_format=ValidationAnalysis,
        middleware=middlewares(),
    )


def format_declaration(declaration: ClaimDeclaration) -> str:
    """Format the declaration for the LLM."""
    useful = declaration.model_dump(mode="json", exclude={"follow_up_message", "complete", "missing_items"})
    return "Claim declaration to validate:\n\n" + json.dumps(useful, ensure_ascii=False, indent=2)


def run_validation(
    declaration: ClaimDeclaration, model=None, on_event=None
) -> tuple[ValidationDecision, list[dict]]:
    """Run the Validation agent on a declaration.

    Parameters
    ----------
    declaration : ClaimDeclaration
        The declaration produced by the Declaration agent.
    model : BaseChatModel, optional
        Model to use. Defaults to the LiteLLM gateway one.
    on_event : Callable[[dict], None], optional
        Receives the tool calls and results as soon as they happen (streaming).

    Returns
    -------
    tuple[ValidationDecision, list[dict]]
        The checked decision, and the trace of tool calls.
    """
    agent = build_validation_agent(model)
    result = run_agent(agent, format_declaration(declaration), TOOLS, on_event)
    decision = consistency_check(result["structured_response"], declaration)
    return decision, extract_trace(result["messages"], TOOLS)
