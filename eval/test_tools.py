"""Offline tool tests (no LLM API required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.database import init_database, lookup_order, lookup_orders_by_email
from agent.kb_store import init_knowledge_base, search_knowledge_base
from agent.tools import create_escalation_tool, lookup_order_tool, search_knowledge_base_tool


def test_order_lookup():
    init_database()
    order = lookup_order(1042)
    assert order is not None
    assert order["status"] == "delivered"
    result = json.loads(lookup_order_tool.invoke({"order_id": 9999}))
    assert result["found"] is False
    print("PASS: order lookup")


def test_email_lookup():
    orders = lookup_orders_by_email("omar.hassan@student.buid.ac.ae")
    assert len(orders) == 2
    print("PASS: email lookup")


def test_kb_search():
    init_knowledge_base()
    hits = search_knowledge_base("student discount")
    assert any("Student" in h["title"] for h in hits)
    result = json.loads(search_knowledge_base_tool.invoke({"query": "shipping times"}))
    assert len(result["results"]) > 0
    print("PASS: kb search")


def test_escalation():
    init_database()
    result = json.loads(
        create_escalation_tool.invoke(
            {
                "reason": "damaged_item",
                "summary": "Test escalation",
                "order_id": 1042,
                "customer_email": "test@student.buid.ac.ae",
            }
        )
    )
    assert result["escalated"] is True
    assert result["ticket"]["ticket_id"] > 0
    print("PASS: escalation")


if __name__ == "__main__":
    test_order_lookup()
    test_email_lookup()
    test_kb_search()
    test_escalation()
    print("\nAll offline tool tests passed.")
