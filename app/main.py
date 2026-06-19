"""ShopMate — Campus Shop Customer Support Agent (Streamlit UI)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.database import init_database
from agent.graph import get_agent_info, invoke_agent
from agent.kb_store import init_knowledge_base
from agent.memory import SessionContext
from app.ui.chat import render_chat_history, render_escalation_banner, render_quick_actions
from app.ui.trace import render_kb_hits, render_trace_viewer

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="ShopMate — Campus Shop Support",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def bootstrap_data():
    """Initialize SQLite DB and ChromaDB knowledge base once."""
    init_database()
    count = init_knowledge_base()
    return count


def init_session_state():
    if "session" not in st.session_state:
        st.session_state.session = SessionContext()
    if "tool_mode" not in st.session_state:
        st.session_state.tool_mode = "full"
    if "openai_api_key" not in st.session_state:
        env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        st.session_state.openai_api_key = "" if env_key in ("", "sk-your-key-here") else env_key


def get_api_key() -> str:
    """Return API key from sidebar session state, falling back to .env."""
    ui_key = (st.session_state.get("openai_api_key") or "").strip()
    if ui_key:
        return ui_key
    env_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if env_key and env_key != "sk-your-key-here":
        return env_key
    return ""


def handle_user_message(prompt: str):
    session: SessionContext = st.session_state.session
    session.customer_email = st.session_state.get("customer_email", "")
    session.preferred_order_id = st.session_state.get("preferred_order_id", "")

    api_key = get_api_key()
    if not api_key:
        st.warning("Add your OpenAI API key in the sidebar to start chatting.")
        return

    with st.chat_message("assistant"):
        with st.spinner("ShopMate is thinking..."):
            try:
                response = invoke_agent(
                    prompt,
                    session,
                    tool_mode=st.session_state.tool_mode,
                    api_key=api_key,
                )
                st.markdown(response)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.info("Check your API key in the sidebar and try again.")


def main():
    init_session_state()
    kb_count = bootstrap_data()
    agent_info = get_agent_info()

    st.title("ShopMate")
    st.caption("Campus Shop AI Customer Support Agent — INF612 Assignment 2")

    tab_chat, tab_trace, tab_about = st.tabs(["Chat", "Agent Trace", "About"])

    with st.sidebar:
        st.header("API settings")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            placeholder="sk-proj-...",
            help="Paste your key here. It is kept only for this browser session.",
        )
        if get_api_key():
            st.success("API key configured")
        else:
            st.warning("API key required to chat")

        st.divider()
        st.header("Customer context")
        st.session_state.customer_email = st.text_input(
            "Email",
            value=st.session_state.get("customer_email", ""),
            placeholder="aisha.khan@student.buid.ac.ae",
        )
        st.session_state.preferred_order_id = st.text_input(
            "Order ID",
            value=st.session_state.get("preferred_order_id", ""),
            placeholder="1042",
        )

        st.divider()
        st.subheader("Retrieved KB")
        render_kb_hits(st.session_state.session.last_kb_hits)

        st.divider()
        if st.button("New session", use_container_width=True):
            st.session_state.session = SessionContext()
            st.rerun()

        st.divider()
        st.caption(f"KB articles loaded: {kb_count}")
        st.caption(f"Model: {agent_info['model']}")

        with st.expander("Demo settings"):
            st.session_state.tool_mode = st.selectbox(
                "Agent mode",
                options=["full", "kb_only", "none"],
                format_func=lambda x: {
                    "full": "Full agent (all tools)",
                    "kb_only": "Baseline: KB only",
                    "none": "Baseline: LLM only",
                }[x],
            )

    with tab_chat:
        if not get_api_key():
            st.info("Enter your OpenAI API key in the sidebar under **API settings**, then try a quick action or send a message.")

        render_escalation_banner()
        render_chat_history()
        if get_api_key():
            render_quick_actions(handle_user_message)

        chat_placeholder = "Add API key in sidebar to chat..." if not get_api_key() else "How can we help you today?"
        if prompt := st.chat_input(chat_placeholder, disabled=not get_api_key()):
            with st.chat_message("user"):
                st.markdown(prompt)
            handle_user_message(prompt)
            st.rerun()

    with tab_trace:
        st.subheader("Agent execution trace")
        st.caption("Thought → Action → Observation → Response")
        render_trace_viewer(st.session_state.session.traces)

    with tab_about:
        st.markdown(
            """
            ### ShopMate Architecture

            - **LLM:** OpenAI via LangChain
            - **Planning:** LangGraph ReAct loop (reason → act → observe)
            - **Tools:** Knowledge base search, order lookup, escalation
            - **Memory:** Multi-turn session history + sidebar context
            - **Data:** SQLite (orders) + ChromaDB (FAQ vectors)

            ### Sample order IDs for demo
            | Order | Status | Customer |
            |-------|--------|----------|
            | 1042 | delivered | aisha.khan@student.buid.ac.ae |
            | 1043 | shipped | omar.hassan@student.buid.ac.ae |
            | 1044 | processing | sara.almaktoum@student.buid.ac.ae |

            ### Setup
            ```bash
            pip install -r requirements.txt
            streamlit run app/main.py
            ```

            Add your OpenAI API key in the sidebar under **API settings**, or optionally in a `.env` file.
            """
        )
        if not get_api_key():
            st.warning("No API key configured. Add one in the sidebar under **API settings**.")


if __name__ == "__main__":
    main()
