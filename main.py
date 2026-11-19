"""Command line of the HomeShield multi-agent claims system.

Usage
-----
    uv run python main.py --list                      # available claims
    uv run python main.py d04                         # end-to-end processing (graph)
    uv run python main.py ex2 --agent declaration     # the Declaration agent alone
    uv run python main.py d11 --agent validation      # Declaration, then Validation
    uv run python main.py d04 --agent expertise       # the three agents, without orchestration
    uv run python main.py --eval declaration          # evaluation on the golden dataset
    uv run python main.py --eval validation
    uv run python main.py --graph                     # the graph, in Mermaid format
"""

from __future__ import annotations

import argparse
import json
import time

from src.knowledge import load_claim, load_claims

WIDTH = 78


def title(text: str) -> None:
    print(f"\n{'─' * WIDTH}\n {text}\n{'─' * WIDTH}")


def show_trace(trace: list[dict]) -> None:
    for call in trace:
        status = "✗" if call["error"] else "✓"
        result = call["result"].replace("\n", " ")
        print(f"  ⚙  {call['tool']}({json.dumps(call['arguments'], ensure_ascii=False)})")
        print(f"  {status}  {result[:150]}{'…' if len(result) > 150 else ''}")


def show_model(obj, exclude: set[str] | None = None) -> None:
    print(json.dumps(obj.model_dump(mode="json", exclude=exclude), ensure_ascii=False, indent=2))


def run_agents(claim_id: str, until: str) -> None:
    """Chain the agents by hand, without orchestration (to test steps 1 to 3)."""
    from src.agents.declaration import run_declaration

    claim = load_claim(claim_id)
    title(f"Claim {claim.id} — received on {claim.received_date}")
    print(claim.message)

    title("Declaration agent")
    declaration, trace = run_declaration(claim)
    show_trace(trace)
    show_model(declaration, {"follow_up_message"})
    if declaration.follow_up_message:
        print(f"\nFollow-up message:\n{declaration.follow_up_message}")
    if until == "declaration":
        return

    from src.agents.validation import run_validation

    title("Validation agent")
    validation, trace = run_validation(declaration)
    show_trace(trace)
    show_model(validation)
    if until == "validation":
        return

    from src.agents.expertise import run_expertise

    title("Expertise agent")
    report = run_expertise(declaration, validation)
    print(report.report)


def process(claim_id: str) -> None:
    """Process a claim with the orchestration graph, following each node."""
    from src.orchestration import build_graph

    claim = load_claim(claim_id)
    title(f"Claim {claim.id} — received on {claim.received_date}")
    print(claim.message)

    graph = build_graph()
    start = time.time()
    state = {"claim": claim, "journal": []}
    for update in graph.stream(state, stream_mode="updates"):
        for node, values in update.items():
            state.update({k: v for k, v in values.items() if k != "journal"})
            for entry in values.get("journal", []):
                title(f"[{time.time() - start:5.1f} s] {node} — {entry['summary']}")
                show_trace(entry["tools"])
            if node == "validation_agent" and state["validation"].corrections:
                print("  Guardrail:", " ; ".join(state["validation"].corrections))
            if node == "expertise_agent":
                print(state["expertise"].report)

    title(f"Final status: {state['status']}")
    print(state["customer_message"])


def evaluate(agent: str) -> None:
    from src.evaluation import evaluate_declaration, evaluate_validation

    function = evaluate_declaration if agent == "declaration" else evaluate_validation
    title(f"Evaluation of the {agent} agent on the golden dataset")
    metrics, cases = function(progress=lambda i, n: print(f"  {i}/{n} cases evaluated", end="\r"))
    print()
    for c in cases:
        ok = c.get("exact", c.get("decision_ok"))
        print(f"  {'✓' if ok else '✗'} {json.dumps(c, ensure_ascii=False, default=str)[:160]}")
    title("Metrics")
    for name, value in metrics.items():
        print(f"  {name:<44} {value:.2f}" if isinstance(value, float) else f"  {name:<44} {value}")
    print(f"\nDetailed report: reports/eval_{agent}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="HomeShield — multi-agent claims handling system")
    parser.add_argument("claim", nargs="?", help="claim identifier (ex1, d04...)")
    parser.add_argument("--agent", choices=["declaration", "validation", "expertise"],
                        help="chain the agents by hand up to this one, without the graph")
    parser.add_argument("--eval", choices=["declaration", "validation"], help="evaluate an agent")
    parser.add_argument("--graph", action="store_true", help="print the graph in Mermaid format")
    parser.add_argument("--list", action="store_true", help="list the available claims")
    args = parser.parse_args()

    if args.list:
        for c in load_claims().values():
            print(f"  {c.id:<4} {c.received_date}  {c.message.splitlines()[0][:60]}… ({len(c.attachments)} photo(s))")
    elif args.graph:
        from src.orchestration import build_graph

        print(build_graph().get_graph().draw_mermaid())
    elif args.eval:
        evaluate(args.eval)
    elif args.claim and args.agent:
        run_agents(args.claim, args.agent)
    elif args.claim:
        process(args.claim)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
