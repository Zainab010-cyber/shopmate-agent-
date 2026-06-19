"""LangChain tools for the Campus Shop support agent."""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from agent.database import create_escalation, lookup_order, lookup_orders_by_email
from agent.kb_store import search_knowledge_base
from agent.prompts import ESCALATION_KEYWORDS, HIGH_RISK_REASONS

KB_CONFIDENCE_THRESHOLD = 0.35


def _detect_escalation_reason(text: str) -> str | None:
    """Map user text to an escalation reason if policy triggers apply."""
    lower = text.lower()
    if any(k in lower for k in ("double charge", "charged twice", "two charges")):
        return "double_charge"
    if any(k in lower for k in ("damaged", "defective", "broken", "arrived broken")):
        return "damaged_item"
    if any(k in lower for k in ("payment dispute", "unauthorized", "fraud")):
        return "payment_dispute"
    if any(k in lower for k in ("speak to human", "human agent", "talk to someone", "manager")):
        return "customer_requested_human"
    if re.search(r"refund.*(2000|2,?000|3000|3,?000|3499)", lower):
        return "high_value_refund"
    return None


@tool
def search_knowledge_base_tool(query: str) -> str:
    """Search Campus Shop FAQ and policy articles. Use for shipping, returns, refunds, warranty, payment, and account questions."""
    hits = search_knowledge_base(query, n_results=3)
    if not hits:
        return json.dumps({"results": [], "message": "No relevant articles found.", "low_confidence": True})

    low_confidence = hits[0]["similarity"] < KB_CONFIDENCE_THRESHOLD
    return json.dumps(
        {
            "results": hits,
            "low_confidence": low_confidence,
            "note": "Consider escalating if confidence is low and the issue is urgent."
            if low_confidence
            else None,
        },
        indent=2,
    )


@tool
def lookup_order_tool(order_id: int) -> str:
    """Look up order status, items, tracking number, and customer by order ID. Order IDs are numeric (e.g. 1042)."""
    order = lookup_order(order_id)
    if not order:
        return json.dumps({"found": False, "message": f"Order #{order_id} was not found. Ask the customer to verify the order ID."})
    return json.dumps({"found": True, "order": order}, indent=2, default=str)


@tool
def lookup_orders_by_email_tool(email: str) -> str:
    """Look up all orders associated with a customer email address."""
    orders = lookup_orders_by_email(email)
    if not orders:
        return json.dumps({"found": False, "message": f"No orders found for {email}."})
    return json.dumps({"found": True, "orders": orders}, indent=2, default=str)


@tool
def create_escalation_tool(
    reason: str,
    summary: str,
    order_id: int | None = None,
    customer_email: str | None = None,
) -> str:
    """Create a human escalation ticket. Reasons: damaged_item, defective_item, double_charge, payment_dispute, high_value_refund, account_security, customer_requested_human, low_kb_confidence."""
    if reason not in HIGH_RISK_REASONS and reason != "low_kb_confidence":
        reason = "customer_requested_human"

    ticket = create_escalation(
        reason=reason,
        transcript=summary,
        order_id=order_id,
        customer_email=customer_email,
    )
    return json.dumps({"escalated": True, "ticket": ticket, "reason": reason}, indent=2)


def get_all_tools():
    """Return tools available to the full agent."""
    return [
        search_knowledge_base_tool,
        lookup_order_tool,
        lookup_orders_by_email_tool,
        create_escalation_tool,
    ]


def get_kb_only_tools():
    """Return KB-only tools for baseline evaluation."""
    return [search_knowledge_base_tool]


def should_auto_escalate(user_message: str, kb_hits: list[dict] | None = None) -> str | None:
    """Policy-aware pre-check for automatic escalation suggestions."""
    reason = _detect_escalation_reason(user_message)
    if reason:
        return reason
    if kb_hits and kb_hits[0]["similarity"] < KB_CONFIDENCE_THRESHOLD:
        urgent = any(w in user_message.lower() for w in ESCALATION_KEYWORDS)
        if urgent:
            return "low_kb_confidence"
    return None
