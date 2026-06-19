"""Agent trace viewer for debugging and demo."""

from __future__ import annotations

import streamlit as st

from agent.memory import AgentTraceStep


def render_trace_viewer(traces: list[AgentTraceStep]) -> None:
    """Render expandable agent execution trace."""
    if not traces:
        st.info("No agent trace yet. Send a message to see reasoning steps.")
        return

    st.markdown(f"**{len(traces)} trace steps**")
    for i, step in enumerate(traces):
        icon = {
            "thought": "💭",
            "action": "🔧",
            "observation": "👁️",
            "response": "✅",
        }.get(step.step_type, "•")

        label = f"{icon} Step {i + 1}: {step.step_type.title()}"
        with st.expander(label, expanded=(step.step_type == "response")):
            st.markdown(step.content)
            if step.metadata:
                st.caption("Metadata: " + ", ".join(f"{k}={v}" for k, v in step.metadata.items()))


def render_kb_hits(hits: list[dict]) -> None:
    """Show last retrieved KB snippets in sidebar."""
    if not hits:
        st.caption("No knowledge base results yet.")
        return

    for hit in hits:
        with st.expander(f"{hit['title']} (sim: {hit['similarity']})"):
            st.markdown(hit["content"][:500] + ("..." if len(hit["content"]) > 500 else ""))
