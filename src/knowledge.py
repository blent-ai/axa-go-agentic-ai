"""Access to data and business documentation.

Everything the agents need to know about the business comes from two documents:

- ``data/claims_process.md``: what the policyholder must provide and what the
  insurer does, for each claim family;
- ``data/home_insurance_policy.md``: the policyholder's policy (coverages, caps,
  deductibles, exclusions, deadlines).

This module also provides the computations that are **never** entrusted to an
LLM: date arithmetic and business-day counting.
"""

from __future__ import annotations

import base64
import io
import json
import re
import unicodedata
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from PIL import Image

from .schemas import Claim

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ATTACHMENTS_DIR = DATA_DIR / "attachments"
GOLDEN_DIR = DATA_DIR / "golden"
REPORTS_DIR = ROOT / "reports"

#: The claim families handled by the system, and the number of their section in
#: the process document.
PROCESS_SECTION = {
    "water_damage": "1",
    "fire_explosion": "2",
    "theft_vandalism": "5",
}

CLAIM_TYPE_LABELS = {
    "water_damage": "Water damage",
    "fire_explosion": "Fire / explosion",
    "theft_vandalism": "Theft / burglary / vandalism",
    "out_of_scope": "Out of scope",
}

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# --------------------------------------------------------------------------- #
# Claims
# --------------------------------------------------------------------------- #
@lru_cache
def load_claims() -> dict[str, Claim]:
    """Load the claims from ``data/claims.json``, indexed by id."""
    raw = json.loads((DATA_DIR / "claims.json").read_text(encoding="utf-8"))
    return {c["id"]: Claim(**c) for c in raw}


def load_claim(claim_id: str) -> Claim:
    """Return a claim by its identifier (``ex1``, ``d04``...)."""
    claims = load_claims()
    if claim_id not in claims:
        raise KeyError(f"Unknown claim: {claim_id}. Available: {', '.join(claims)}.")
    return claims[claim_id]


def load_jsonl(name: str) -> list[dict]:
    """Load a JSONL file from the ``data/golden`` folder."""
    path = GOLDEN_DIR / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Business documents
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Lowercase, without accents or punctuation: to compare titles."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", no_accents).strip()


@lru_cache
def process_text() -> str:
    """The claims handling process document, in full."""
    return (DATA_DIR / "claims_process.md").read_text(encoding="utf-8")


@lru_cache
def policy_text() -> str:
    """The policyholder's home insurance policy, in full."""
    return (DATA_DIR / "home_insurance_policy.md").read_text(encoding="utf-8")


def process_for_type(claim_type: str) -> str:
    """Return the section of the process that matches a claim type.

    Parameters
    ----------
    claim_type : str
        ``water_damage``, ``fire_explosion`` or ``theft_vandalism``.

    Returns
    -------
    str
        The Markdown section of the process (what the policyholder must provide,
        insurer process, closure).
    """
    number = PROCESS_SECTION.get(claim_type)
    if number is None:
        raise KeyError(
            f"Claim type out of scope: {claim_type}. "
            f"Handled types: {', '.join(PROCESS_SECTION)}."
        )
    sections = re.split(r"\n(?=## )", process_text())
    for section in sections:
        if section.startswith(f"## {number}."):
            return section.strip()
    raise KeyError(f"Section {number} not found in the process.")


@lru_cache
def policy_sections() -> dict[str, str]:
    """Split the policy into sections, indexed by their title.

    Coverages (``###`` titles) and chapters without subsections
    (``## 2. General Exclusions``...) each become a section. The header (policy
    number, policyholder, effective date) is the "Header" section.

    Returns
    -------
    dict[str, str]
        ``{"Water Damage": "...", "General Exclusions": "...", ...}``
    """
    sections: dict[str, str] = {}
    title, lines = "Header", []
    for line in policy_text().splitlines():
        heading = re.match(r"^(#{2,3})\s+(?:\d+\.\s*)?(.+?)\s*$", line)
        if heading:
            if any(l.strip() for l in lines):
                sections[title] = "\n".join(lines).strip()
            title, lines = heading.group(2), []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections[title] = "\n".join(lines).strip()
    return sections


def policy_amount(section: str, label: str) -> float | None:
    """Read an amount ("cap", "Deductible") in a policy section.

    Used by evaluations and tests, not by the agents: reading the policy is the
    LLM's job.
    """
    text = policy_sections().get(section, "")
    found = re.search(rf"{label}[^:]*:\*\*\s*€\s*([\d,\s]+)", text)
    if found:
        return float(re.sub(r"\D", "", found.group(1)))
    if re.search(rf"{label}[^:]*:\*\*\s*None", text):
        return 0.0
    return None


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def _easter(year: int) -> date:
    """Date of Easter Sunday (Meeus/Jones/Butcher algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


@lru_cache
def public_holidays(year: int) -> frozenset[date]:
    """The public holidays of a year (mainland France, where the policy is written)."""
    easter = _easter(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    movable = [easter + timedelta(days=n) for n in (1, 39, 50)]  # Easter Monday, Ascension, Whit Monday
    return frozenset([date(year, m, d) for m, d in fixed] + movable)


def is_business_day(day: date) -> bool:
    """A business day is a day from Monday to Friday, excluding public holidays."""
    return day.weekday() < 5 and day not in public_holidays(day.year)


def business_days_between(start: date, end: date) -> int:
    """Number of business days elapsed after ``start``, up to and including ``end``.

    A loss that occurred on a Saturday and was reported the following Monday was
    reported after 1 business day.
    """
    if end <= start:
        return 0
    return sum(is_business_day(start + timedelta(days=n)) for n in range(1, (end - start).days + 1))


def shift_date(reference: date, offset_days: int = 0, weekday: str | None = None) -> date:
    """Compute a date relative to a reference date.

    Parameters
    ----------
    reference : date
        Reference date (usually the received date of the claim).
    offset_days : int, optional
        Offset in days (``-1`` for "yesterday").
    weekday : str, optional
        If provided ("saturday"), returns the last day with that name, at the
        latest on the shifted date.
    """
    target = reference + timedelta(days=offset_days)
    if weekday:
        name = normalize(weekday)
        if name not in WEEKDAYS:
            raise ValueError(f"Unknown weekday: {weekday}. Expected: {', '.join(WEEKDAYS)}.")
        target -= timedelta(days=(target.weekday() - WEEKDAYS.index(name)) % 7)
    return target


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def attachment_path(filename: str) -> Path:
    """Path of an attachment in ``data/attachments``."""
    return ATTACHMENTS_DIR / filename


@lru_cache(maxsize=64)
def image_data_url(filename: str, max_size: int = 1024) -> str:
    """Encode a photo as a JPEG ``data:`` URL, resized, for a vision model.

    Policyholders' photos can weigh several MB: we shrink them to limit the
    latency and cost of the calls.
    """
    path = attachment_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Attachment not found: {filename}")
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def image_block(filename: str) -> dict:
    """"Image" content block of a multimodal message (OpenAI-compatible format)."""
    return {"type": "image_url", "image_url": {"url": image_data_url(filename)}}
