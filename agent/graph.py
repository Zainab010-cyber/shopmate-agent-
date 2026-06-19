"""LangGraph ReAct agent for Campus Shop customer support."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from agent.memory import SessionContext
from agent.prompts import SYSTEM_PROMPT
from agent.tools import get_all_tools, get_kb_only_tools, should_auto_escalate

load_dotenv()

MAX_ITERATIONS = 8


class AgentState(TypedDict):
    messages: Annotated[list, lambda a, b: a + b]
    session: SessionContext
    iteration: int


def _build_llm(api_key: str | None = None):
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    kwargs: dict[str, Any] = {"model": model, "temperature": 0.2}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def create_agent_graph(tool_mode: str = "full", api_key: str | None = None):
    """Create LangGraph agent. tool_mode: 'full', 'kb_only', or 'none'."""
    if tool_mode == "none":
        tools = []
    elif tool_mode == "kb_only":
        tools = get_kb_only_tools()
    else:
        tools = get_all_tools()

    llm = _build_llm(api_key=api_key)
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    tool_node = ToolNode(tools) if tools else None

    def call_model(state: AgentState) -> dict:
        session: SessionContext = state["session"]
        system_content = SYSTEM_PROMPT
        ctx = session.context_prompt()
        if ctx:
            system_content += f"\n\nSession context:\n{ctx}"

        messages = [SystemMessage(content=system_content)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        session.add_trace("thought", getattr(response, "content", "") or "Planning next action...")
        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                session.add_trace(
                    "action",
                    f"{tc['name']}({json.dumps(tc['args'])})",
                    tool=tc["name"],
                    args=tc["args"],
                )
        return {"messages": [response], "iteration": state.get("iteration", 0) + 1}

    def run_tools(state: AgentState) -> dict:
        if tool_node is None:
            return {"messages": []}
        session: SessionContext = state["session"]
        result = tool_node.invoke(state)
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                session.add_trace("observation", msg.content, tool=msg.name)
                if msg.name == "search_knowledge_base_tool":
                    try:
                        data = json.loads(msg.content)
                        session.last_kb_hits = data.get("results", [])
                    except json.JSONDecodeError:
                        pass
                if msg.name == "create_escalation_tool":
                    try:
                        data = json.loads(msg.content)
                        if data.get("escalated"):
                            session.escalated = True
                            session.last_ticket_id = data.get("ticket", {}).get("ticket_id")
                    except json.JSONDecodeError:
                        pass
        return result

    def should_continue(state: AgentState) -> str:
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "end"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    if tools:
        workflow.add_node("tools", run_tools)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
    else:
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)

    return workflow.compile()


def invoke_agent(
    user_message: str,
    session: SessionContext,
    tool_mode: str = "full",
    api_key: str | None = None,
) -> str:
    """Run the agent on a user message and return the final response."""
    session.add_message("user", user_message)

    auto_reason = should_auto_escalate(user_message, session.last_kb_hits or None)
    if auto_reason:
        session.add_trace(
            "thought",
            f"Policy check flagged potential escalation: {auto_reason}",
            policy_trigger=auto_reason,
        )

    graph = create_agent_graph(tool_mode=tool_mode, api_key=api_key)
    history_messages = []
    for role, content in session.to_langchain_messages()[:-1]:
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        else:
            history_messages.append(AIMessage(content=content))
    history_messages.append(HumanMessage(content=user_message))

    result = graph.invoke(
        {
            "messages": history_messages,
            "session": session,
            "iteration": 0,
        }
    )

    final_messages = result["messages"]
    response_text = ""
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            response_text = msg.content
            break

    if not response_text:
        response_text = "I apologise, I could not complete your request. Please try again or ask to speak with a human agent."

    session.add_message("assistant", response_text)
    session.add_trace("response", response_text)
    return response_text


def get_agent_info() -> dict[str, Any]:
    """Return metadata about the configured agent."""
    return {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "tools": [t.name for t in get_all_tools()],
        "max_iterations": MAX_ITERATIONS,
        "framework": "LangGraph + LangChain",
    }
