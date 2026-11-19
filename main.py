"""Command line of the HomeShield multi-agent claims system.

Usage
-----
    uv run python main.py --list                      # available claims
    uv run python main.py d04                         # end-to-end processing, streamed live (graph)
    uv run python main.py --resume d04-1a2b3c4d       # resume an interrupted claim from its checkpoint
    uv run python main.py --history d04-1a2b3c4d      # checkpoints of a claim
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

from src.knowledge import ROOT, load_claim, load_claims

WIDTH = 78

#: Checkpoint database (step 5): lets an interrupted claim be resumed, even after a restart.
CHECKPOINT_DB = ROOT / "checkpoints" / "claims.sqlite"


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


def follow(events, start: float) -> dict:
    """Display the streamed events as they arrive. Returns the final state."""
    state: dict = {}
    for event in events:
        elapsed = f"[{time.time() - start:5.1f} s]"
        kind = event["type"]
        if kind == "node_started":
            print(f"\n{elapsed} ▶ {event['node']}…")
        elif kind == "tool_call":
            print(f"  ⚙  {event['tool']}({json.dumps(event['arguments'], ensure_ascii=False)})")
        elif kind == "tool_result":
            result = event["result"].replace("\n", " ")
            print(f"  {'✗' if event['error'] else '✓'}  {result[:150]}{'…' if len(result) > 150 else ''}")
        elif kind == "progress":
            print(f"  …  {event['message']}")
        elif kind == "node_finished":
            update = event["update"]
            for entry in update.get("journal", []):
                title(f"{elapsed} {event['node']} — {entry['summary']}")
            if "validation" in update and update["validation"].corrections:
                print("  Guardrail:", " ; ".join(update["validation"].corrections))
            if "expertise" in update:
                print(update["expertise"].report)
        elif kind == "node_error":
            print(f"\n{elapsed} ✗ {event['node']} failed: {event['error']}")
        elif kind == "done":
            state = event["state"]
    return state


def process(claim_id: str, thread_id: str | None = None) -> None:
    """Process a claim with the orchestration graph, streaming each node live.

    The state is checkpointed after every node in ``checkpoints/claims.sqlite``.
    If the run crashes, it can be resumed with ``--resume <thread id>``.
    """
    from src.orchestration import build_graph, get_checkpointer, stream_claim

    claim = load_claim(claim_id)
    title(f"Claim {claim.id} — received on {claim.received_date}")
    print(claim.message)

    graph = build_graph(get_checkpointer(CHECKPOINT_DB))
    thread_id = thread_id or f"{claim.id}-{int(time.time())}"
    print(f"\nThread: {thread_id}   (resume with: --resume {thread_id})")
    try:
        state = follow(stream_claim(claim, thread_id, graph), time.time())
    except Exception as error:  # noqa: BLE001
        title(f"Interrupted: {type(error).__name__}: {error}")
        print(f"The progress is saved. Resume with:\n  uv run python main.py --resume {thread_id}")
        raise SystemExit(1) from error
    title(f"Final status: {state['status']}")
    print(state["customer_message"])


def resume(thread_id: str) -> None:
    """Resume an interrupted claim from its last checkpoint."""
    from src.orchestration import build_graph, get_checkpointer, resume_claim

    graph = build_graph(get_checkpointer(CHECKPOINT_DB))
    title(f"Resuming thread {thread_id}")
    state = follow(resume_claim(thread_id, graph), time.time())
    title(f"Final status: {state['status']}")
    print(state["customer_message"])


def history(thread_id: str) -> None:
    """List the checkpoints of a claim."""
    from src.orchestration import build_graph, claim_history, get_checkpointer

    graph = build_graph(get_checkpointer(CHECKPOINT_DB))
    title(f"Checkpoints of thread {thread_id} (most recent first)")
    for item in claim_history(thread_id, graph):
        print(f"  step {item['step']:>2}  next={item['next'] or '—'}  status={item['status'] or '—'}  {item['created_at']}")


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
    parser.add_argument("--thread-id", help="identifier of the processing (checkpoints), default: generated")
    parser.add_argument("--resume", metavar="THREAD_ID", help="resume an interrupted claim from its checkpoint")
    parser.add_argument("--history", metavar="THREAD_ID", help="list the checkpoints of a claim")
    parser.add_argument("--eval", choices=["declaration", "validation"], help="evaluate an agent")
    parser.add_argument("--graph", action="store_true", help="print the graph in Mermaid format")
    parser.add_argument("--list", action="store_true", help="list the available claims")
    args = parser.parse_args()

    if args.list:
        for c in load_claims().values():
            print(f"  {c.id:<4} {c.received_date}  {c.message.splitlines()[0][:60]}… ({len(c.attachments)} photo(s))")
    elif args.resume:
        resume(args.resume)
    elif args.history:
        history(args.history)
    elif args.graph:
        from src.orchestration import build_graph

        print(build_graph().get_graph().draw_mermaid())
    elif args.eval:
        evaluate(args.eval)
    elif args.claim and args.agent:
        run_agents(args.claim, args.agent)
    elif args.claim:
        process(args.claim, args.thread_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
