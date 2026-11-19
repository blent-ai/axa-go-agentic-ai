"""Project check, without any model call and without an API key.

The agents run with a **fake model** whose answers (tool calls, structured
outputs) are written in advance. This verifies the wiring (tools, guardrails,
routing, graph, metrics) independently of the LiteLLM gateway. The quality of
the answers is judged with the real model and the evaluations on the golden
datasets.

The checks follow the order of the session. Each one is independent: an
unfinished step is flagged (⏳) without preventing the following ones from running.

Usage
-----
    uv run python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import ToolException  # noqa: E402

from src import orchestration  # noqa: E402
from src.agents import declaration as declaration_module  # noqa: E402
from src.agents import expertise as expertise_module  # noqa: E402
from src.agents import validation as validation_module  # noqa: E402
from src.evaluation import (  # noqa: E402
    aggregate_declaration,
    evaluate_validation_case,
    policy_accuracy,
    score_items,
)
from src.knowledge import business_days_between, load_claim, load_jsonl, policy_sections  # noqa: E402
from src.schemas import (  # noqa: E402
    ClaimDeclaration,
    ClaimInfo,
    DamageEstimate,
    ValidationAnalysis,
    ValidationDecision,
)


# --------------------------------------------------------------------------- #
# Fake model
# --------------------------------------------------------------------------- #
class ScriptedModel(BaseChatModel):
    """Fake model that replays a list of prepared answers."""

    responses: list = []
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name: str, args: dict, n: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}_{n}"}])


INFO_EX1 = {
    "claim_type": "water_damage", "loss_date": "2025-09-15", "location": "Kitchen",
    "circumstances": "Leak from the upstairs neighbor's dishwasher.", "damaged_items": [],
    "third_party_involved": True, "joint_report_provided": False, "criminal_suspicion": False,
    "police_report_filed": False, "inventory_provided": False, "fire_brigade_called": None,
}

ANALYSIS_V01 = {
    "coverage": "Water Damage", "cap": 25000, "deductible": 150,
    "reporting_deadline_business_days": 5, "business_days_elapsed": 1, "deadline_met": True,
    "photos_consistent": True, "exclusion": None, "decision": "send_to_expertise",
    "rationale": "Water Damage coverage, deadline met.",
}


def info(**overrides) -> ClaimInfo:
    return ClaimInfo(**{**INFO_EX1, **overrides})


def golden_declaration(case_id: str) -> ClaimDeclaration:
    case = next(c for c in load_jsonl("validation.jsonl") if c["id"] == case_id)
    return ClaimDeclaration(**case["declaration"])


def decision(**overrides) -> ValidationDecision:
    data = {**ANALYSIS_V01, "claim_id": "d04", "llm_decision": "send_to_expertise", **overrides}
    return ValidationDecision(**data)


# --------------------------------------------------------------------------- #
# Foundation (provided)
# --------------------------------------------------------------------------- #
def foundation_documents():
    sections = policy_sections()
    assert "Fire / Explosion" in sections and "General Exclusions" in sections, list(sections)
    assert business_days_between(date(2025, 9, 20), date(2025, 9, 22)) == 1
    assert business_days_between(date(2025, 9, 22), date(2025, 10, 2)) == 8
    assert business_days_between(date(2025, 11, 10), date(2025, 11, 12)) == 1, "November 11 is a public holiday"


# --------------------------------------------------------------------------- #
# Step 1 — Declaration agent
# --------------------------------------------------------------------------- #
def s1_missing_items():
    f = declaration_module.missing_items
    assert f(info(), ["WaterDamage_100.jpg"]) == ["joint_report"], "third party involved without a joint report"
    assert f(info(), []) == ["photos", "joint_report"], "photos are counted on the attachments"
    assert f(info(third_party_involved=False, loss_date=None), ["x.jpg"]) == ["loss_date"]
    theft = info(claim_type="theft_vandalism", third_party_involved=False, criminal_suspicion=True)
    assert f(theft, []) == ["police_report", "inventory"], f(theft, [])
    fire = info(claim_type="fire_explosion", third_party_involved=False, criminal_suspicion=True)
    assert f(fire, ["f.png"]) == ["inventory", "police_report"], f(fire, ["f.png"])
    assert f(info(claim_type="out_of_scope", loss_date=None), []) == []


def s1_declaration_agent():
    model = ScriptedModel(responses=[
        call("consult_process", {"claim_type": "water_damage"}),
        call("compute_date", {"reference_date": "2025-09-16", "offset_days": -1}, 1),
        call("ClaimInfo", INFO_EX1, 2),
    ])
    declaration, trace = declaration_module.run_declaration(load_claim("ex1"), model=model)
    assert [t["tool"] for t in trace] == ["consult_process", "compute_date"], trace
    assert "Water damage" in trace[0]["result"], "consult_process must return the process"
    assert declaration.missing_items == ["joint_report"] and not declaration.complete
    assert declaration.follow_up_message and "joint water damage report" in declaration.follow_up_message


def s1_score_items():
    assert score_items(["police_report", "photos"], ["police_report", "inventory"]) == {"tp": 1, "fp": 1, "fn": 1}
    assert score_items([], []) == {"tp": 0, "fp": 0, "fn": 0}
    metrics = aggregate_declaration([
        {"tp": 2, "fp": 0, "fn": 0, "type_ok": True, "date_ok": True, "status_ok": True, "exact": True},
        {"tp": 0, "fp": 1, "fn": 1, "type_ok": True, "date_ok": False, "status_ok": True, "exact": False},
    ])
    assert round(metrics["missing_items_precision"], 3) == 0.667
    assert round(metrics["missing_items_recall"], 3) == 0.667


# --------------------------------------------------------------------------- #
# Step 2 — Validation agent
# --------------------------------------------------------------------------- #
def s2_consult_policy():
    text = validation_module.consult_policy.invoke({"section": "fire"})
    assert "100,000" in text and "300" in text, text
    assert "wear and tear" in validation_module.consult_policy.invoke({"section": "General Exclusions"})
    assert "2 business days" in validation_module.consult_policy.invoke({"section": "Policyholder Obligations"})
    try:
        validation_module.consult_policy.invoke({"section": "garden"})
    except ToolException as error:
        assert "Water Damage" in str(error), "the error message must list the available sections"
    else:
        raise AssertionError("an unknown section must raise a ToolException")


def s2_consistency_check():
    f = validation_module.consistency_check
    d04 = golden_declaration("v01")
    ok = f(ValidationAnalysis(**ANALYSIS_V01), d04)
    assert ok.decision == "send_to_expertise" and not ok.corrections, ok.corrections

    late = f(ValidationAnalysis(**ANALYSIS_V01), golden_declaration("v05"))  # 8 business days
    assert late.business_days_elapsed == 8 and not late.deadline_met, "business days must be recomputed"
    assert late.decision == "reject" and late.llm_decision == "send_to_expertise"
    assert late.corrections, "every correction must be traced"

    exclusion = f(ValidationAnalysis(**{**ANALYSIS_V01, "exclusion": "Normal wear and tear or lack of maintenance."}), d04)
    assert exclusion.decision == "reject"

    photo = f(ValidationAnalysis(**{**ANALYSIS_V01, "photos_consistent": False}), d04)
    assert photo.decision == "request_additional_info", photo.decision


def s2_validation_agent():
    model = ScriptedModel(responses=[
        call("consult_policy", {"section": "Water Damage"}),
        call("business_days_between", {"start_date": "2025-09-20", "end_date": "2025-09-22"}, 1),
        call("ValidationAnalysis", ANALYSIS_V01, 2),
    ])
    result, trace = validation_module.run_validation(golden_declaration("v01"), model=model)
    assert [t["tool"] for t in trace] == ["consult_policy", "business_days_between"], trace
    assert result.decision == "send_to_expertise" and result.claim_id == "d04"


def s2_policy_accuracy():
    expected = {"coverage": "Water Damage", "cap": 25000, "deductible": 150, "reporting_deadline_business_days": 5}
    all_good = policy_accuracy(decision(coverage="Water Damage coverage"), expected)
    assert all_good == {"coverage": True, "cap": True, "deductible": True, "reporting_deadline_business_days": True}, all_good
    wrong = policy_accuracy(decision(coverage="Fire / Explosion", deductible=300, cap=None), expected)
    assert wrong == {"coverage": False, "cap": False, "deductible": False, "reporting_deadline_business_days": True}, wrong
    case = next(c for c in load_jsonl("validation.jsonl") if c["id"] == "v01")
    assert evaluate_validation_case(decision(), case)["policy_facts_ok"] == 1.0


# --------------------------------------------------------------------------- #
# Step 3 — Expertise agent
# --------------------------------------------------------------------------- #
def s3_payout():
    f = expertise_module.compute_payout
    assert f(1650, 150, 25000) == 1500
    assert f(100, 150, 25000) == 0, "never negative"
    assert f(30000, 300, 20000) == 20000, "capped"


def s3_multimodal_message():
    message = expertise_module.build_message(golden_declaration("v08"), decision(coverage="Fire / Explosion"))
    blocks = message["content"]
    assert message["role"] == "user" and blocks[0]["type"] == "text"
    assert "Fire / Explosion" in blocks[0]["text"]
    images = [b for b in blocks if b["type"] == "image_url"]
    assert len(images) == 2, "one image block per attached photo"
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def s3_expertise_agent():
    estimate = {
        "findings": ["Stained false ceiling"], "severity": "moderate",
        "line_items": [{"label": "False ceiling", "amount": 1200}, {"label": "Paint", "amount": 450}],
        "contractors": ["Plasterer"], "assumptions": "A single photo.",
    }
    model = ScriptedModel(responses=[call("DamageEstimate", estimate)])
    report = expertise_module.run_expertise(golden_declaration("v01"), decision(), model=model)
    assert report.total_cost == 1650 and report.payable_amount == 1500, report
    assert "€1,500.00" in report.report


# --------------------------------------------------------------------------- #
# Step 4 — Orchestration
# --------------------------------------------------------------------------- #
EXPECTED_EDGES = {
    ("__start__", "declaration_agent"),
    ("declaration_agent", "validation_agent"),
    ("declaration_agent", "notify_customer"),
    ("validation_agent", "expertise_agent"),
    ("validation_agent", "notify_customer"),
    ("expertise_agent", "notify_customer"),
    ("notify_customer", "__end__"),
}


def s4_routing():
    complete = ClaimDeclaration(**{**INFO_EX1, "claim_id": "ex1", "received_date": "2025-09-16"})
    incomplete = complete.model_copy(update={"complete": False, "missing_items": ["joint_report"]})
    out_of_scope = complete.model_copy(update={"claim_type": "out_of_scope"})
    assert orchestration.after_declaration({"declaration": complete}) == "validation_agent"
    assert orchestration.after_declaration({"declaration": incomplete}) == "notify_customer"
    assert orchestration.after_declaration({"declaration": out_of_scope}) == "notify_customer"
    assert orchestration.after_validation({"validation": decision()}) == "expertise_agent"
    for refusal in ("reject", "request_additional_info"):
        assert orchestration.after_validation({"validation": decision(decision=refusal)}) == "notify_customer"


def s4_graph():
    graph = orchestration.build_graph()
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    missing, extra = EXPECTED_EDGES - edges, edges - EXPECTED_EDGES
    assert not missing and not extra, f"missing edges: {missing}; extra edges: {extra}"


def fake_agents(calls: list, fail_expertise_once: list | None = None):
    """Replace the three agents with fakes (no LLM). Returns a function restoring the originals."""
    originals = (orchestration.run_declaration, orchestration.run_validation, orchestration.run_expertise)

    def fake_declaration(claim, **kwargs):
        calls.append("declaration")
        complete = claim.id != "ex1"
        return ClaimDeclaration(
            **{**INFO_EX1, "claim_id": claim.id, "received_date": claim.received_date,
               "complete": complete, "missing_items": [] if complete else ["joint_report"],
               "follow_up_message": None if complete else "Please send the joint report."}
        ), []

    def fake_validation(declaration, **kwargs):
        calls.append("validation")
        return decision(decision="reject" if declaration.claim_id == "d11" else "send_to_expertise",
                        claim_id=declaration.claim_id), []

    def fake_expertise(declaration, validation, **kwargs):
        calls.append("expertise")
        if fail_expertise_once:
            fail_expertise_once.pop()
            raise RuntimeError("gateway timeout")
        return expertise_module.ExpertiseReport(
            claim_id=declaration.claim_id, total_cost=1000, deductible=150, cap=25000,
            payable_amount=850, report="...",
            estimate=DamageEstimate(findings=[], severity="low", line_items=[], contractors=[], assumptions=""),
        )

    orchestration.run_declaration, orchestration.run_validation, orchestration.run_expertise = (
        fake_declaration, fake_validation, fake_expertise)

    def restore():
        orchestration.run_declaration, orchestration.run_validation, orchestration.run_expertise = originals

    return restore


def s4_end_to_end():
    """Run the graph with simulated agents (no LLM) on three scenarios."""
    calls: list = []
    restore = fake_agents(calls)
    try:
        graph = orchestration.build_graph()
        expected = {"ex1": ("customer_follow_up", ["declaration"]),
                    "d11": ("rejected", ["declaration", "validation"]),
                    "d04": ("forwarded_to_advisor", ["declaration", "validation", "expertise"])}
        for claim_id, (status, agents) in expected.items():
            calls.clear()
            state = graph.invoke({"claim": load_claim(claim_id), "journal": []})
            assert state["status"] == status, f"{claim_id}: status {state['status']} instead of {status}"
            assert calls == agents, f"{claim_id}: agents called {calls} instead of {agents}"
            assert len(state["journal"]) == len(agents) + 1, "the journal must accumulate the entries of each node"
    finally:
        restore()


# --------------------------------------------------------------------------- #
# Step 5 — Streaming and checkpointing
# --------------------------------------------------------------------------- #
def _require_step5():
    for name in ("stream_claim", "resume_claim", "get_checkpointer", "claim_history"):
        if not hasattr(orchestration, name):
            raise NotImplementedError(f"orchestration.{name}")


def s5_streaming():
    """`stream_claim` yields typed events, in processing order."""
    _require_step5()
    calls: list = []
    restore = fake_agents(calls)
    try:
        events = list(orchestration.stream_claim(load_claim("d04"), thread_id="smoke-stream"))
    finally:
        restore()
    started = [e["node"] for e in events if e["type"] == "node_started"]
    finished = [e["node"] for e in events if e["type"] == "node_finished"]
    expected = ["declaration_agent", "validation_agent", "expertise_agent", "notify_customer"]
    assert started == expected and finished == expected, (started, finished)
    positions = [i for i, e in enumerate(events) if e["type"] in ("node_started", "node_finished")]
    assert [events[i]["type"] for i in positions[:2]] == ["node_started", "node_finished"], "events must be incremental"
    assert events[-1]["type"] == "done" and events[-1]["state"]["status"] == "forwarded_to_advisor", events[-1]


def s5_tool_events():
    """An agent reports its tool calls as they happen (`on_event`), not only at the end."""
    if not hasattr(orchestration, "stream_claim"):
        raise NotImplementedError("agents.common.run_agent")
    model = ScriptedModel(responses=[
        call("consult_process", {"claim_type": "water_damage"}),
        call("compute_date", {"reference_date": "2025-09-16", "offset_days": -1}, 1),
        call("ClaimInfo", INFO_EX1, 2),
    ])
    events: list = []
    declaration_module.run_declaration(load_claim("ex1"), model=model, on_event=events.append)
    kinds = [(e["type"], e["tool"]) for e in events]
    assert kinds == [
        ("tool_call", "consult_process"), ("tool_result", "consult_process"),
        ("tool_call", "compute_date"), ("tool_result", "compute_date"),
    ], kinds


def s5_checkpoint_resume():
    """After a crash in a node, `resume_claim` restarts from the last checkpoint."""
    _require_step5()
    calls: list = []
    restore = fake_agents(calls, fail_expertise_once=[True])
    checkpointer = orchestration.get_checkpointer(None)  # in memory
    try:
        graph = orchestration.build_graph(checkpointer)
        try:
            list(orchestration.stream_claim(load_claim("d04"), thread_id="smoke-resume", graph=graph))
        except RuntimeError:
            pass
        else:
            raise AssertionError("the simulated crash must interrupt the processing")
        assert calls == ["declaration", "validation", "expertise"], calls
        snapshot = graph.get_state({"configurable": {"thread_id": "smoke-resume"}})
        assert snapshot.next == ("expertise_agent",), f"the checkpoint must point to the failed node: {snapshot.next}"
        assert "validation" in snapshot.values, "the work already done must be persisted"

        calls.clear()
        events = list(orchestration.resume_claim("smoke-resume", graph=graph))
        assert calls == ["expertise"], f"only the failed node must be replayed, got {calls}"
        assert events[-1]["type"] == "done" and events[-1]["state"]["status"] == "forwarded_to_advisor"
        assert len(events[-1]["state"]["journal"]) == 4, "the journal must not contain duplicates"

        history = orchestration.claim_history("smoke-resume", graph=graph)
        assert history and history[0]["next"] == (), "history lists checkpoints, most recent first"
    finally:
        restore()


def s5_sqlite_persistence():
    """A SQLite checkpointer survives a restart: another graph instance can resume the thread."""
    _require_step5()
    import tempfile

    calls: list = []
    restore = fake_agents(calls, fail_expertise_once=[True])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "claims.sqlite"
        try:
            graph = orchestration.build_graph(orchestration.get_checkpointer(path))
            try:
                list(orchestration.stream_claim(load_claim("d04"), thread_id="smoke-sqlite", graph=graph))
            except RuntimeError:
                pass
            calls.clear()
            new_graph = orchestration.build_graph(orchestration.get_checkpointer(path))  # "restart"
            events = list(orchestration.resume_claim("smoke-sqlite", graph=new_graph))
            assert calls == ["expertise"], calls
            assert events[-1]["state"]["status"] == "forwarded_to_advisor"
        finally:
            restore()


# --------------------------------------------------------------------------- #
CHECKS = [
    ("Foundation", "Business documents and business days", foundation_documents),
    ("Step 1", "Completeness guardrail: missing_items", s1_missing_items),
    ("Step 1", "Declaration agent: build_declaration_agent", s1_declaration_agent),
    ("Step 1", "Completeness metric: score_items", s1_score_items),
    ("Step 2", "Tool consult_policy", s2_consult_policy),
    ("Step 2", "Consistency guardrail: consistency_check", s2_consistency_check),
    ("Step 2", "Validation agent end to end", s2_validation_agent),
    ("Step 2", "Factual metric: policy_accuracy", s2_policy_accuracy),
    ("Step 3", "Payout computation", s3_payout),
    ("Step 3", "Multimodal message: build_message", s3_multimodal_message),
    ("Step 3", "Expertise agent end to end", s3_expertise_agent),
    ("Step 4", "Routing: after_declaration, after_validation", s4_routing),
    ("Step 4", "Graph: build_graph", s4_graph),
    ("Step 4", "End-to-end processing (simulated agents)", s4_end_to_end),
    ("Step 5", "Streaming: stream_claim", s5_streaming),
    ("Step 5", "Agent tool events: on_event", s5_tool_events),
    ("Step 5", "Checkpointing: crash, then resume_claim", s5_checkpoint_resume),
    ("Step 5", "Persistence across restarts (SQLite)", s5_sqlite_persistence),
]


def main() -> int:
    print("\nProject check (fake model, no gateway call)\n")
    results = []
    current_step = None
    for step, label, check in CHECKS:
        if step != current_step:
            print(f"── {step} {'─' * (60 - len(step))}")
            current_step = step
        try:
            check()
            status, detail = "✅", ""
        except NotImplementedError as error:
            status, detail = "⏳", f"to complete — {error}"
        except AssertionError as error:
            status, detail = "❌", str(error) or "unexpected result"
        except Exception as error:  # noqa: BLE001
            frame = traceback.extract_tb(error.__traceback__)[-1]
            status = "💥"
            detail = f"{type(error).__name__}: {error} ({Path(frame.filename).name}:{frame.lineno})"
        results.append(status)
        print(f"  {status} {label}")
        if detail:
            print(f"      {detail}")
    passed = results.count("✅")
    print(f"\n{passed}/{len(results)} checks passed.")
    if "⏳" in results:
        nxt = CHECKS[results.index("⏳")]
        print(f"Next: {nxt[0]} — {nxt[1]}.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
