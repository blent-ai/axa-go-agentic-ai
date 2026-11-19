"""Expertise AI agent: estimate costs from the photos.

For a validated file, the Expertise agent produces:

- a **damage estimate** line by line, from the photos (vision model) and the
  declaration;
- the **payable amount**, within the policy coverage (deductible, cap);
- a **report** for an advisor: the agent never communicates a decision to the
  policyholder.

Architecture: no ReAct loop here. A single **multimodal** call to the vision
model (text + photos) with a structured output (`DamageEstimate`), then a
deterministic payout computation: contractual arithmetic is not entrusted to
an LLM.

No automatic evaluation is expected for this agent: it is carried out by the
business experts.
"""

from __future__ import annotations

from ..config import get_vision_model
from ..knowledge import CLAIM_TYPE_LABELS, image_block
from ..schemas import ClaimDeclaration, DamageEstimate, ExpertiseReport, ValidationDecision

EXPERTISE_PROMPT = """\
You are the Expertise agent of HomeShield. You prepare, for an advisor, a first
costed estimate of a home insurance claim that has already been validated.

From the attached photos and the declaration:
1. Describe the visible or declared damage (findings), without extrapolating.
2. Estimate the costs line by line (repair, replacement, cleaning, intervention
   of a professional), in euros including taxes, at current market prices.
   For an item declared with its value, use that value.
3. List the contractors to mobilize (plumber, fire expert, locksmith...).
4. State your assumptions and uncertainties: the advisor must be able to judge
   the reliability of the estimate.

Do not compute the payout (deductible, cap): it is computed separately.
"""


def compute_payout(total_cost: float, deductible: float, cap: float) -> float:
    """Payable amount: the cost, minus the deductible, within the cap.

    ``amount = min(max(cost − deductible, 0), cap)``

    Parameters
    ----------
    total_cost : float
        Total estimated cost of the damage, in euros.
    deductible : float
        Deductible of the coverage, in euros.
    cap : float
        Compensation cap of the coverage, in euros.

    Returns
    -------
    float
        The payable amount, rounded to the cent.
    """
    return round(min(max(total_cost - deductible, 0.0), cap), 2)


def build_message(declaration: ClaimDeclaration, validation: ValidationDecision) -> dict:
    """Build the multimodal message sent to the vision model.

    The message content is a **list of blocks**: a text block (the declaration
    and the applicable coverage), then an image block per attached photo
    (`image_block`).

    Returns
    -------
    dict
        ``{"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}, ...]}``
    """
    items = "\n".join(
        f"- {i.label}" + (f": €{i.value:.0f}" if i.value is not None else "")
        for i in declaration.damaged_items
    ) or "- (no item listed)"
    text = (
        f"Claim type: {CLAIM_TYPE_LABELS[declaration.claim_type]}\n"
        f"Date: {declaration.loss_date}\n"
        f"Location: {declaration.location or 'not specified'}\n"
        f"Circumstances: {declaration.circumstances}\n"
        f"Declared items:\n{items}\n"
        f"Applicable coverage: {validation.coverage}\n"
        f"Attached photos: {len(declaration.attachments)}"
    )
    content = [{"type": "text", "text": text}] + [image_block(name) for name in declaration.attachments]
    return {"role": "user", "content": content}


def eur(amount: float, decimals: int = 0) -> str:
    """Format an amount in euros: €25,000."""
    return f"€{amount:,.{decimals}f}"


def write_report(
    declaration: ClaimDeclaration,
    validation: ValidationDecision,
    estimate: DamageEstimate,
    total_cost: float,
    amount: float,
) -> str:
    """Markdown report for the advisor (deterministic template)."""
    lines = "\n".join(f"| {i.label} | {eur(i.amount)} |" for i in estimate.line_items)
    findings = "\n".join(f"- {f}" for f in estimate.findings)
    contractors = ", ".join(estimate.contractors) or "none"
    return f"""\
### Expertise report — claim {declaration.claim_id}

**Claim**: {CLAIM_TYPE_LABELS[declaration.claim_type]} on {declaration.loss_date} — {declaration.location or "location not specified"}
**Coverage**: {validation.coverage} (cap {eur(validation.cap or 0)}, deductible {eur(validation.deductible or 0)})
**Severity**: {estimate.severity}

#### Findings
{findings}

#### Estimate
| Item | Amount |
|---|---:|
{lines}
| **Total** | **{eur(total_cost)}** |

**Proposed payable amount**: **{eur(amount, 2)}**
(cost − deductible, within the cap)

**Contractors to mobilize**: {contractors}

#### Assumptions and uncertainties
{estimate.assumptions}

*Automatic estimate, to be validated by an advisor before any communication to the policyholder.*
"""


def run_expertise(
    declaration: ClaimDeclaration, validation: ValidationDecision, model=None
) -> ExpertiseReport:
    """Run the Expertise agent on a validated file.

    Parameters
    ----------
    declaration : ClaimDeclaration
        The declaration (circumstances, items, photos).
    validation : ValidationDecision
        The validation decision (coverage, deductible, cap).
    model : BaseChatModel, optional
        Vision model. Defaults to the LiteLLM gateway one.

    Returns
    -------
    ExpertiseReport
        The estimate, the payable amount and the report for the advisor.
    """
    llm = (model or get_vision_model()).with_structured_output(DamageEstimate)
    estimate = llm.invoke(
        [{"role": "system", "content": EXPERTISE_PROMPT}, build_message(declaration, validation)]
    )
    total_cost = round(sum(i.amount for i in estimate.line_items), 2)
    deductible, cap = validation.deductible or 0.0, validation.cap or 0.0
    amount = compute_payout(total_cost, deductible, cap)
    return ExpertiseReport(
        claim_id=declaration.claim_id,
        estimate=estimate,
        total_cost=total_cost,
        deductible=deductible,
        cap=cap,
        payable_amount=amount,
        report=write_report(declaration, validation, estimate, total_cost, amount),
    )
