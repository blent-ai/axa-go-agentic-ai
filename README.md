# Masterclass — Multi-agent home insurance claims system

The non-life insurer **HomeShield** wants to speed up and make more reliable the
handling of claims reported by its policyholders. As an AI Engineer, you build
with **LangGraph** a system of three specialized AI agents, orchestrated by a
workflow, that processes each declaration in the background.

Scope: **water damage**, **fire / explosion**, **theft / burglary / vandalism**.

> **You are on branch `master`.** The project is split into one branch per step:
> each branch builds on the previous one (`git diff step-1 step-2` shows exactly
> what a step adds).

## Branches

| Branch | | Topic | What is implemented | Try it |
|---|---|---|---|---|
| `master` | 👉 | Starting point | Foundation only: schemas, knowledge, config, CLI, UI, tests. Every agent function is a stub. | `uv run python tests/smoke_test.py` |
| `step-1` | ⬜ | Declaration agent | `missing_items`, `build_declaration_agent`, `score_items` | `main.py ex2 --agent declaration` · `main.py --eval declaration` |
| `step-2` | ⬜ | Validation agent | `consult_policy`, `consistency_check`, `policy_accuracy` | `main.py d11 --agent validation` · `main.py --eval validation` |
| `step-3` | ⬜ | Expertise agent | `compute_payout`, `build_message` | `main.py d04 --agent expertise` |
| `step-4` | ⬜ | Full orchestration | `after_declaration`, `after_validation`, `build_graph` | `streamlit run app.py` · `main.py d04` |
| `step-5` | ⬜ | Streaming & checkpointing | `run_agent`, `stream_claim`, `resume_claim`, `get_checkpointer`, `claim_history` | `main.py d04` (live events) · `main.py --resume <thread>` · `main.py --history <thread>` |

Legend: ✅ implemented on this branch · 👉 the step this branch adds · ⬜ still to come (stubs raising `NotImplementedError`).

`uv run python tests/smoke_test.py` always tells you what remains to be completed.

## The system

| Agent | Role | Architecture | Evaluation |
|---|---|---|---|
| 🧾 **Declaration** | Structures the policyholder's free-text message, checks that the items required by the process are present, drafts a follow-up if needed | ReAct agent (`create_agent`) + structured output + completeness guardrail | Completeness of the requested information |
| 📑 **Validation** | Checks the photos (vision model) and the policy coverage, decides what happens next | ReAct agent + structured output + consistency guardrail | Accuracy of the policy facts used |
| 🔍 **Expertise** | Estimates costs from the photos and computes the payable amount | Multimodal call + deterministic computation | Manual (business experts) |
| 🔀 **Orchestration** | Chains the agents according to the business process | `StateGraph` workflow with conditional edges | Smoke test |

```
START → declaration_agent ─┬─ complete ─────────→ validation_agent ─┬─ send ─────→ expertise_agent ─┐
                           │                                         │                              │
                           └─ incomplete / out of scope ─→ notify_customer ←─ reject / more info ───┘
                                                                  ↓
                                                                 END
```

Three principles guide the design:

1. **A typed interface contract between agents** (`src/schemas.py`). Each agent
   produces a Pydantic object that the next one consumes: this is what guarantees
   that every agent has all the information it needs.
2. **The LLM reads, the code decides.** LLMs extract facts from a free-text
   message, a policy or a photo. The business rules (required items, decision
   consistency, payout computation, business-day counting) are deterministic
   code: testable and auditable.
3. **A workflow rather than an LLM supervisor.** In a regulated process, the
   sequence of steps is known in advance: a rejected file must never go to
   expertise. The routing is therefore written in Python.

## Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Access to the LiteLLM gateway (URL + key), with a **multimodal** model
  (or a separate vision model)

## Installation

```bash
uv sync
cp .env.example .env   # then fill in your credentials
```

| Variable | Role |
|---|---|
| `LITELLM_BASE_URL` | Gateway URL, OpenAI-compatible (`.../v1`) |
| `LITELLM_API_KEY` | Virtual key issued by the gateway |
| `LITELLM_MODEL` | Language model (tool calls, structured outputs) |
| `LITELLM_VISION_MODEL` | Vision model (optional, defaults to `LITELLM_MODEL`) |

## Running

```bash
uv run streamlit run app.py                     # test and visualization interface
uv run python main.py d04                       # end-to-end processing, on the command line
uv run python main.py ex2 --agent declaration   # a single agent (or up to validation / expertise)
uv run python main.py --eval declaration        # evaluation on the golden dataset
uv run python main.py --eval validation
uv run python tests/smoke_test.py               # offline check, no API key needed
```

The Streamlit interface is only a test bench: in production, processing runs in
the background for each declaration. It lets you pick or edit a claim, add
photos, follow the graph node by node, answer a follow-up (the "Policyholder's
addendum" field) and run the evaluations.

Until step 4, the Streamlit interface chains the agents **without routing**: a rejected file still goes to expertise.

### Demo scenarios

| Claim | Situation | Expected status |
|---|---|---|
| `d04` | Water heater leak, complete file | `forwarded_to_advisor` (expertise done) |
| `d07` | Burglary, police report and inventory provided | `forwarded_to_advisor` |
| `ex1` | Leak caused by the neighbor, no joint report | `customer_follow_up` → complete it, then `forwarded_to_advisor` |
| `ex2` | Burglary, neither police report nor inventory | `customer_follow_up` |
| `d10` | Fire reported with a flood photo | `additional_info_requested` |
| `d11` | Water damage reported after 8 business days | `rejected` |
| `d09` | Storm | `out_of_scope` |

## Architecture

```
app.py                     Streamlit test and visualization interface
main.py                    Command line
src/
  config.py                The models behind the LiteLLM gateway
  schemas.py               Interface contract between agents (Pydantic)
  knowledge.py             Process, policy, business days, photo encoding
  agents/
    common.py              Middlewares (tool errors, call limit), trace
    declaration.py         Declaration agent                          (Step 1)
    validation.py          Validation agent                           (Step 2)
    expertise.py           Expertise agent                            (Step 3)
  orchestration.py         The LangGraph workflow                                 (Step 4)
  evaluation.py            Metrics on the golden datasets             (Steps 1-2)
data/
  claims_process.md        Claims handling process (reference documentation)
  home_insurance_policy.md The policyholder's home insurance policy
  claims.json              12 claim declarations
  attachments/             Photos attached to the declarations
  golden/                  Golden datasets: declaration.jsonl (12 cases), validation.jsonl (9 cases)
tests/smoke_test.py        Offline check, with a fake model
```

## Evaluations

No threshold is defined: the metrics are a basis for discussion with the
business. Detailed reports are saved in `reports/`.

**Declaration agent: completeness of the requested information** (`data/golden/declaration.jsonl`)

- accuracy of the claim type, of the date and of the complete / incomplete status;
- precision, recall and F1 (micro-average) on the **missing items**;
- exact-match rate of the missing items.

A missing item that goes undetected blocks the file later in the process, while a
false alarm only costs a follow-up: **recall** is the metric to watch first.

**Validation agent: use of the right policy information** (`data/golden/validation.jsonl`)

- **policy fact accuracy**: coverage, cap, deductible and reporting deadline
  reported correctly (the metric requested by the business);
- accuracy of the exclusions, of the photo compliance and of the decision, before
  (`llm_decision`) and after the consistency guardrail.

The Validation agent is evaluated **independently** of the Declaration agent: its
test cases start from already structured declarations.

> The policy, the process and the claims are fictitious and for teaching purposes.

## Session outline

| Step | Topic | Files to complete (from the previous branch) |
|---|---|---|
| 1 | Declaration agent | `src/agents/declaration.py`: `missing_items`, `build_declaration_agent` · `src/evaluation.py`: `score_items` |
| 2 | Validation agent | `src/agents/validation.py`: `consult_policy`, `consistency_check` · `src/evaluation.py`: `policy_accuracy` |
| 3 | Expertise agent | `src/agents/expertise.py`: `compute_payout`, `build_message` |
| 4 | Orchestration | `src/orchestration.py`: `after_declaration`, `after_validation`, `build_graph` |
| 5 | Streaming & checkpointing | `src/agents/common.py`: `run_agent` · `src/orchestration.py`: `stream_claim`, `resume_claim`, `get_checkpointer`, `claim_history` |
