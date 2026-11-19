"""Evaluation of the agents on the golden datasets.

Two agents are evaluated, each on the metric requested by the business:

- **Declaration agent**: the *completeness of the requested information*. Did
  it identify the right claim type, the right date, and above all the right
  missing items? (``data/golden/declaration.jsonl``)
- **Validation agent**: the ability to *use the right factual information from
  the policy*: coverage, cap, deductible and reporting deadline.
  (``data/golden/validation.jsonl``)

The Validation agent is evaluated **independently** of the Declaration agent:
its test cases start from already structured declarations. An error in one
therefore does not propagate to the other's metrics.

No threshold is set: the metrics are a basis for discussion with the business.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable

from .agents.declaration import run_declaration
from .agents.validation import run_validation
from .knowledge import REPORTS_DIR, load_claim, load_jsonl, normalize
from .schemas import ClaimDeclaration, ValidationDecision

#: The policy facts that the Validation agent must report.
POLICY_FACTS = ["coverage", "cap", "deductible", "reporting_deadline_business_days"]


# --------------------------------------------------------------------------- #
# Declaration agent: completeness
# --------------------------------------------------------------------------- #
def score_items(predicted: list[str], expected: list[str]) -> dict:
    """Compare the predicted missing items with the expected ones.

    Parameters
    ----------
    predicted : list[str]
        The missing items detected by the agent.
    expected : list[str]
        The missing items annotated in the golden dataset.

    Returns
    -------
    dict
        ``{"tp": ..., "fp": ..., "fn": ...}``: true positives (missing items
        correctly detected), false positives (false alarms), false negatives
        (missing items not detected).
    """
    # TODO (Step 1) — Count true positives, false positives and false negatives
    # (hint: set operations).
    raise NotImplementedError("Step 1 — score_items")


def evaluate_declaration_case(prediction: ClaimDeclaration, expected: dict) -> dict:
    """Evaluate a declaration produced by the agent against its annotation."""
    dates = expected["loss_date"]
    predicted_date = prediction.loss_date.isoformat() if prediction.loss_date else None
    items = score_items(prediction.missing_items, expected["missing_items"])
    return {
        "id": expected["id"],
        "type_ok": prediction.claim_type == expected["claim_type"],
        "date_ok": predicted_date is None if dates is None else predicted_date in dates,
        "status_ok": prediction.complete == (not expected["missing_items"]),
        "exact": items["fp"] == 0 and items["fn"] == 0,
        **items,
        "predicted_type": prediction.claim_type,
        "predicted_date": predicted_date,
        "predicted_missing": prediction.missing_items,
        "expected_missing": expected["missing_items"],
    }


def aggregate_declaration(cases: list[dict]) -> dict:
    """Aggregate per-case results into global metrics.

    Precision, recall and F1 are **micro-averages**: we sum the true positives,
    false positives and false negatives of all cases before computing.
    """
    n = len(cases)
    tp, fp, fn = (sum(c[k] for c in cases) for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cases": n,
        "type_accuracy": sum(c["type_ok"] for c in cases) / n,
        "date_accuracy": sum(c["date_ok"] for c in cases) / n,
        "status_accuracy": sum(c["status_ok"] for c in cases) / n,
        "missing_items_precision": precision,
        "missing_items_recall": recall,
        "missing_items_f1": f1,
        "exact_match": sum(c["exact"] for c in cases) / n,
    }


# --------------------------------------------------------------------------- #
# Validation agent: policy facts
# --------------------------------------------------------------------------- #
def same_coverage(predicted: str | None, expected: str | None) -> bool:
    """Compare two coverage names, ignoring case, accents and prefixes.

    "Water Damage coverage" and "water damage" designate the same coverage.
    """
    if predicted is None or expected is None:
        return predicted is expected
    p, e = normalize(predicted), normalize(expected)
    return e in p or p in e


def policy_accuracy(prediction: ValidationDecision, expected: dict) -> dict[str, bool]:
    """Check, fact by fact, that the agent reported the right policy information.

    Parameters
    ----------
    prediction : ValidationDecision
        The decision produced by the agent.
    expected : dict
        The expected facts (``expected`` key of the golden dataset).

    Returns
    -------
    dict[str, bool]
        For each fact of `POLICY_FACTS`, True if the agent reported it
        correctly. The coverage is compared with `same_coverage`, amounts and
        deadlines numerically.
    """
    # TODO (Step 2) — One boolean per fact of POLICY_FACTS:
    #   - "coverage": `same_coverage(prediction.coverage, expected["coverage"])`;
    #   - "cap", "deductible": numeric comparison (beware of None);
    #   - "reporting_deadline_business_days": equality.
    raise NotImplementedError("Step 2 — policy_accuracy")


def evaluate_validation_case(prediction: ValidationDecision, case: dict) -> dict:
    """Evaluate a Validation agent decision against its annotation."""
    expected = case["expected"]
    facts = policy_accuracy(prediction, expected)
    return {
        "id": case["id"],
        **{f"{k}_ok": v for k, v in facts.items()},
        "policy_facts_ok": sum(facts.values()) / len(facts),
        "business_days_ok": prediction.business_days_elapsed == expected["business_days_elapsed"],
        "photos_ok": prediction.photos_consistent == expected["photos_consistent"],
        "exclusion_ok": bool(prediction.exclusion) == expected["exclusion"],
        "decision_ok": prediction.decision == expected["decision"],
        "llm_decision_ok": prediction.llm_decision == expected["decision"],
        "predicted_coverage": prediction.coverage,
        "predicted_decision": prediction.decision,
        "expected_decision": expected["decision"],
        "corrections": prediction.corrections,
    }


def aggregate_validation(cases: list[dict]) -> dict:
    """Aggregate per-case results into global metrics."""
    n = len(cases)

    def rate(key: str) -> float:
        return sum(c[key] for c in cases) / n

    return {
        "cases": n,
        "policy_fact_accuracy": rate("policy_facts_ok"),
        **{f"{fact}_accuracy": rate(f"{fact}_ok") for fact in POLICY_FACTS},
        "exclusion_accuracy": rate("exclusion_ok"),
        "photo_accuracy": rate("photos_ok"),
        "llm_decision_accuracy": rate("llm_decision_ok"),
        "decision_accuracy": rate("decision_ok"),
    }


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _run(function: Callable, items: list, workers: int, progress: Callable | None) -> list:
    """Apply `function` in parallel (LLM calls are independent)."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, result in enumerate(executor.map(function, items), start=1):
            results.append(result)
            if progress:
                progress(i, len(items))
    return results


def _save(name: str, metrics: dict, cases: list[dict]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    report = {"date": datetime.now().isoformat(timespec="seconds"), "metrics": metrics, "cases": cases}
    (REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def evaluate_declaration(limit: int | None = None, workers: int = 6, progress=None) -> tuple[dict, list[dict]]:
    """Evaluate the Declaration agent on ``data/golden/declaration.jsonl``.

    Returns
    -------
    tuple[dict, list[dict]]
        The aggregated metrics and the per-case detail. The report is also
        saved to ``reports/eval_declaration.json``.
    """
    golden = load_jsonl("declaration.jsonl")[:limit]

    def one_case(expected: dict) -> dict:
        prediction, _ = run_declaration(load_claim(expected["id"]))
        return evaluate_declaration_case(prediction, expected)

    cases = _run(one_case, golden, workers, progress)
    metrics = aggregate_declaration(cases)
    _save("eval_declaration", metrics, cases)
    return metrics, cases


def evaluate_validation(limit: int | None = None, workers: int = 6, progress=None) -> tuple[dict, list[dict]]:
    """Evaluate the Validation agent on ``data/golden/validation.jsonl``.

    Returns
    -------
    tuple[dict, list[dict]]
        The aggregated metrics and the per-case detail. The report is also
        saved to ``reports/eval_validation.json``.
    """
    golden = load_jsonl("validation.jsonl")[:limit]

    def one_case(case: dict) -> dict:
        prediction, _ = run_validation(ClaimDeclaration(**case["declaration"]))
        return evaluate_validation_case(prediction, case)

    cases = _run(one_case, golden, workers, progress)
    metrics = aggregate_validation(cases)
    _save("eval_validation", metrics, cases)
    return metrics, cases
