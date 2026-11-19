"""Business schemas exchanged between the agents.

These Pydantic models are the **interface contract** of the multi-agent system:
each agent produces a typed object that the next agent consumes. This is what
guarantees that "every agent has all the information it needs to carry out the
rest of the process".

Two families of schemas:

- what **the LLM** fills in (`ClaimInfo`, `ValidationAnalysis`,
  `DamageEstimate`, `PhotoConsistency`): the field descriptions are read by the
  model, they are part of the prompt;
- what **the code** completes (`ClaimDeclaration`, `ValidationDecision`,
  `ExpertiseReport`): the deterministic business rules (missing items, decision
  consistency, payout computation) are not entrusted to the LLM.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

ClaimType = Literal["water_damage", "fire_explosion", "theft_vandalism", "out_of_scope"]

#: The items the process requires from the policyholder, depending on the claim type.
RequiredItem = Literal["loss_date", "photos", "joint_report", "police_report", "inventory"]

Decision = Literal["send_to_expertise", "request_additional_info", "reject"]

Status = Literal[
    "out_of_scope",           # claim not handled by the system (storm, glass breakage...)
    "customer_follow_up",     # incomplete declaration: the policyholder is asked for more
    "additional_info_requested",  # non-compliant documents: the policyholder must complete the file
    "rejected",               # not covered, excluded, or reported too late
    "forwarded_to_advisor",   # expertise done, report forwarded to an advisor
]


# --------------------------------------------------------------------------- #
# System input
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    """A claim declaration as received (free-text message + attachments)."""

    id: str
    received_date: date
    message: str
    attachments: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Declaration agent
# --------------------------------------------------------------------------- #
class DamagedItem(BaseModel):
    """A damaged, destroyed or stolen item mentioned by the policyholder."""

    label: str = Field(description="Name of the item, as described by the policyholder.")
    value: float | None = Field(
        default=None, description="Value in euros if the policyholder gives it, otherwise null."
    )


class ClaimInfo(BaseModel):
    """The facts extracted from the policyholder's message by the Declaration agent."""

    claim_type: ClaimType = Field(
        description=(
            "Claim family. 'out_of_scope' for anything that is neither water "
            "damage, nor fire / explosion, nor theft / burglary / vandalism "
            "(storm, natural disaster, glass breakage alone...)."
        )
    )
    loss_date: date | None = Field(
        description=(
            "Date the loss occurred (YYYY-MM-DD), computed with the compute_date "
            "tool if it is relative. null if the policyholder does not give it."
        )
    )
    location: str | None = Field(default=None, description="Room(s) or part(s) of the home affected.")
    circumstances: str = Field(description="Factual summary of the circumstances, in one or two sentences.")
    damaged_items: list[DamagedItem] = Field(
        default_factory=list, description="Damaged, destroyed or stolen items mentioned by the policyholder."
    )
    third_party_involved: bool = Field(
        description="A neighbor or a third party caused the loss or is a victim of it."
    )
    joint_report_provided: bool = Field(
        description="The policyholder states that a joint report was filled in or attached."
    )
    criminal_suspicion: bool = Field(
        description="The loss results from a deliberate act (theft, break-in, vandalism, criminal act)."
    )
    police_report_filed: bool = Field(
        description="The policyholder states that a police report was filed (with or without a report number)."
    )
    inventory_provided: bool = Field(
        description=(
            "The policyholder precisely lists the damaged or stolen items, with "
            "their value or an invoice. A vague mention (\"all my devices\") is "
            "not enough."
        )
    )
    fire_brigade_called: bool | None = Field(
        default=None, description="Fire only: the fire brigade intervened. null if not mentioned."
    )


class ClaimDeclaration(ClaimInfo):
    """The structured declaration, completed by the completeness guardrail.

    This is the output of the Declaration agent and the input of the Validation agent.
    """

    claim_id: str
    received_date: date
    attachments: list[str] = Field(default_factory=list)
    missing_items: list[RequiredItem] = Field(default_factory=list)
    complete: bool = True
    follow_up_message: str | None = None


# --------------------------------------------------------------------------- #
# Validation agent
# --------------------------------------------------------------------------- #
class PhotoConsistency(BaseModel):
    """The vision model's answer to "does this photo match the claim?"."""

    consistent: bool = Field(description="The photo shows damage compatible with the declared claim.")
    description: str = Field(description="What the photo shows, in one sentence.")


class ValidationAnalysis(BaseModel):
    """What the Validation agent establishes from the policy and the documents."""

    coverage: str | None = Field(
        description="Exact name of the applicable policy coverage (e.g. \"Fire / Explosion\"), null if none."
    )
    cap: float | None = Field(description="Compensation cap of this coverage, in euros.")
    deductible: float | None = Field(description="Deductible of this coverage, in euros (0 if none).")
    reporting_deadline_business_days: int | None = Field(
        description="Contractual reporting deadline for this claim, in business days."
    )
    business_days_elapsed: int | None = Field(
        description="Business days between the loss date and the received date (business_days_between tool)."
    )
    deadline_met: bool = Field(description="The claim was reported within the contractual deadline.")
    photos_consistent: bool = Field(
        description="All attached photos are consistent with the claim (true if there is no photo)."
    )
    exclusion: str | None = Field(
        description="Applicable policy exclusion, quoted as written in the policy, otherwise null."
    )
    decision: Decision = Field(description="Next step for the file.")
    rationale: str = Field(description="Justification of the decision, citing the policy elements used.")


class ValidationDecision(ValidationAnalysis):
    """The Validation agent's decision, after the consistency guardrail."""

    claim_id: str
    llm_decision: Decision
    corrections: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Expertise agent
# --------------------------------------------------------------------------- #
class CostItem(BaseModel):
    """A line of the estimate (repair, replacement, intervention)."""

    label: str = Field(description="Nature of the item, e.g. \"Hallway ceiling repair (8 m²)\".")
    amount: float = Field(description="Estimated cost in euros, taxes included.")


class DamageEstimate(BaseModel):
    """The estimate produced by the vision model."""

    findings: list[str] = Field(description="Damage observed on the photos or described by the policyholder.")
    severity: Literal["low", "moderate", "severe"] = Field(description="Overall severity of the claim.")
    line_items: list[CostItem] = Field(description="Cost items, repairs and replacements.")
    contractors: list[str] = Field(
        description="Contractors to mobilize (plumber, fire expert, locksmith, painter...)."
    )
    assumptions: str = Field(description="Assumptions and uncertainties of the estimate, for the advisor.")


class ExpertiseReport(BaseModel):
    """The report sent to the advisor: estimate and payable amount."""

    claim_id: str
    estimate: DamageEstimate
    total_cost: float
    deductible: float
    cap: float
    payable_amount: float
    report: str
