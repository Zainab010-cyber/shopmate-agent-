"""Session memory helpers for multi-turn conversations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTraceStep:
    """A single step in the agent execution trace."""

    step_type: str  # thought, action, observation, response
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionContext:
    """Per-session context passed to the agent."""

    customer_email: str = ""
    preferred_order_id: str = ""
    chat_history: list[dict[str, str]] = field(default_factory=list)
    traces: list[AgentTraceStep] = field(default_factory=list)
    last_kb_hits: list[dict] = field(default_factory=list)
    escalated: bool = False
    last_ticket_id: int | None = None

    def add_message(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def add_trace(self, step_type: str, content: str, **metadata: Any) -> None:
        self.traces.append(AgentTraceStep(step_type=step_type, content=content, metadata=metadata))

    def clear(self) -> None:
        self.chat_history.clear()
        self.traces.clear()
        self.last_kb_hits.clear()
        self.escalated = False
        self.last_ticket_id = None

    def context_prompt(self) -> str:
        """Build supplemental context from sidebar fields."""
        parts = []
        if self.customer_email:
            parts.append(f"Customer email on file: {self.customer_email}")
        if self.preferred_order_id:
            parts.append(f"Preferred order ID: {self.preferred_order_id}")
        return "\n".join(parts) if parts else ""

    def to_langchain_messages(self) -> list[tuple[str, str]]:
        """Convert chat history to (role, content) tuples for the LLM."""
        return [(m["role"], m["content"]) for m in self.chat_history]
