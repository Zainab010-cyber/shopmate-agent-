"""Chat UI components for ShopMate."""

from __future__ import annotations

import streamlit as st

QUICK_PROMPTS = [
    "Track my order #1042",
    "What is your return policy?",
    "I was charged twice for my order",
    "I'd like to speak to a human agent",
]


def render_chat_history() -> None:
    """Display conversation history."""
    for msg in st.session_state.session.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_quick_actions(on_select) -> None:
    """Render quick-start prompt buttons."""
    st.markdown("**Quick actions**")
    cols = st.columns(2)
    for i, prompt in enumerate(QUICK_PROMPTS):
        if cols[i % 2].button(prompt, key=f"quick_{i}", use_container_width=True):
            on_select(prompt)


def render_escalation_banner() -> None:
    """Show banner when an escalation was created."""
    session = st.session_state.session
    if session.escalated and session.last_ticket_id:
        st.warning(
            f"Escalated to human agent — Ticket **#{session.last_ticket_id}**. "
            "A team member will respond within 4 business hours."
        )
