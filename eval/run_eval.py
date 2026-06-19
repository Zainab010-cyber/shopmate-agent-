"""Batch evaluation runner for ShopMate agent across scenarios and baselines."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.database import init_database
from agent.graph import invoke_agent
from agent.kb_store import init_knowledge_base
from agent.memory import SessionContext

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _tools_used(session: SessionContext) -> list[str]:
    tools = []
    for step in session.traces:
        if step.step_type == "action" and step.metadata.get("tool"):
            tools.append(step.metadata["tool"])
    return tools


def _check_success(scenario: dict, session: SessionContext, response: str) -> dict:
    expected = set(scenario.get("expected_tools", []))
    used = set(_tools_used(session))
    tool_match = expected.issubset(used) if expected else True

    criteria = scenario.get("success_criteria", "").lower()
    keyword_checks = []
    if "30-day" in criteria or "30 day" in criteria:
        keyword_checks.append(any(k in response.lower() for k in ("30 day", "30-day", "thirty day")))
    if "delivered" in criteria:
        keyword_checks.append("delivered" in response.lower())
    if "not found" in criteria:
        keyword_checks.append(any(k in response.lower() for k in ("not found", "couldn't find", "could not find", "no order")))
    if "escalat" in criteria:
        keyword_checks.append(session.escalated or "escalat" in response.lower() or "ticket" in response.lower())
    if "clarif" in criteria or "order id" in criteria:
        keyword_checks.append(any(k in response.lower() for k in ("order id", "email", "which order", "provide", "share")))

    content_ok = all(keyword_checks) if keyword_checks else bool(response.strip())
    return {
        "tool_match": tool_match,
        "expected_tools": list(expected),
        "used_tools": list(used),
        "content_ok": content_ok,
        "escalated": session.escalated,
        "success": tool_match and content_ok,
    }


def run_evaluation(tool_mode: str = "full", limit: int | None = None) -> dict:
    """Run all scenarios for a given agent mode."""
    load_dotenv(PROJECT_ROOT / ".env")
    init_database()
    init_knowledge_base()

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if limit:
        scenarios = scenarios[:limit]

    results = []
    for scenario in scenarios:
        session = SessionContext()
        try:
            response = invoke_agent(scenario["input"], session, tool_mode=tool_mode)
            check = _check_success(scenario, session, response)
        except Exception as exc:
            response = f"ERROR: {exc}"
            check = {"success": False, "error": str(exc)}

        results.append(
            {
                "id": scenario["id"],
                "category": scenario["category"],
                "input": scenario["input"],
                "response": response,
                "traces": [
                    {"type": t.step_type, "content": t.content[:300], "metadata": t.metadata}
                    for t in session.traces
                ],
                **check,
            }
        )

    success_count = sum(1 for r in results if r.get("success"))
    summary = {
        "tool_mode": tool_mode,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "success_count": success_count,
        "success_rate": round(success_count / len(results) * 100, 1) if results else 0,
        "escalation_count": sum(1 for r in results if r.get("escalated")),
        "avg_tools_per_scenario": round(
            sum(len(r.get("used_tools", [])) for r in results) / len(results), 2
        )
        if results
        else 0,
        "results": results,
    }
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ShopMate agent")
    parser.add_argument("--mode", choices=["full", "kb_only", "none", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    modes = ["full", "kb_only", "none"] if args.mode == "all" else [args.mode]
    all_summaries = []

    for mode in modes:
        print(f"\nRunning evaluation: {mode}")
        summary = run_evaluation(tool_mode=mode, limit=args.limit)
        all_summaries.append(summary)
        print(f"  Success rate: {summary['success_rate']}% ({summary['success_count']}/{summary['total']})")
        print(f"  Escalations: {summary['escalation_count']}")
        print(f"  Avg tools/scenario: {summary['avg_tools_per_scenario']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
