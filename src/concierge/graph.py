"""The Meridian National Customer Service Concierge graph.

A custom LangGraph StateGraph implementing the classic agent loop:

    START -> agent -> (tools? -> agent)* -> END

Exported as `graph` for LangSmith / LangGraph CLI deployment.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from concierge.context import get_prompt
from concierge.state import ConciergeState
from concierge.tools import TOOLS, TransactionLimitError

load_dotenv(override=True)

# The system prompt (AGENTS.md) is pulled from LangSmith Context Hub at module
# import; a hub edit is picked up on the next process start. Falls back to the
# seed in concierge.prompts.SYSTEM_PROMPT when the hub is unreachable.
SYSTEM_PROMPT = get_prompt()


def _make_model() -> ChatOpenAI:
    model_name = os.getenv("CONCIERGE_MODEL", "gpt-4o-mini")
    base_url = os.getenv("BASE_URL")
    if base_url:
        # Route through the LangSmith LLM Gateway: callers authenticate with
        # their LangSmith API key; provider keys live in Provider Secrets.
        client = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            base_url=base_url,
            api_key=os.environ["LANGSMITH_API_KEY"],
        )
    else:
        client = ChatOpenAI(model=model_name, temperature=0.2)
    return client.bind_tools(TOOLS)


def agent_node(state: ConciergeState) -> dict:
    """Call the LLM with the message history plus the system prompt."""
    model = _make_model()
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = model.invoke(messages)

    retrieval_calls = state.get("retrieval_calls", 0)
    tool_calls = getattr(response, "tool_calls", None) or []
    new_retrievals = sum(
        1 for call in tool_calls if call.get("name") == "search_banking_docs"
    )

    return {
        "messages": [response],
        "retrieval_calls": retrieval_calls + new_retrievals,
    }


def _handle_tool_error(error: Exception) -> str:
    """Preserve tool failures and give guidance appropriate to the failed input."""
    if isinstance(error, TransactionLimitError):
        return (
            f"Error: {error} "
            "Use a positive limit of at most 50; disclose any cap or change to "
            "the requested limit and report actual returned and listed counts. "
            "Do not change the customer ID."
        )
    return (
        f"Error: {error} "
        "Do not guess, construct, pad, or enumerate identifiers, or retry with "
        "altered identifiers. For lookup or format failures, ask the representative "
        "for the correct input. A format failure does not mean the customer is absent."
    )


def _build_graph():
    builder = StateGraph(ConciergeState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=_handle_tool_error))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


graph = _build_graph()
