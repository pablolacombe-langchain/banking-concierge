"""The Meridian National Customer Service Concierge graph.

A custom LangGraph StateGraph implementing the classic agent loop:

    START -> agent -> (tools? -> agent)* -> END

Exported as `graph` for LangSmith / LangGraph CLI deployment.
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import Client, configure

from concierge.context import get_prompt
from concierge.state import ConciergeState
from concierge.tools import TOOLS

load_dotenv(override=True)


def _redact_text(text: str) -> str:
    def redact_card(match: re.Match) -> str:
        digits = re.sub(r"[ -]", "", match.group())
        if not 13 <= len(digits) <= 19:
            return match.group()
        doubled = [
            int(digit) * (2 if index % 2 else 1)
            for index, digit in enumerate(reversed(digits))
        ]
        if sum(value - 9 if value > 9 else value for value in doubled) % 10:
            return match.group()
        return f"[card ending {digits[-4:]}]"

    text = re.sub(
        r"\b(?:CVV|CVC|security[ -]code)\b\s*(?::|=|is)?\s*[0-9]{3,4}\b",
        "[CVV removed]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b[0-9]{3}-[0-9]{2}-([0-9]{4})\b", r"[SSN ending \1]", text)
    return re.sub(r"(?<!\w)[0-9](?:[ -]?[0-9]){12,}(?!\w)", redact_card, text)


def _redact(data: Any) -> Any:
    if isinstance(data, str):
        return _redact_text(data)
    if isinstance(data, dict):
        return {key: _redact(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_redact(value) for value in data]
    return data


configure(client=Client(anonymizer=_redact))

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
    history = [
        message.model_copy(update={"content": _redact(message.content)})
        if isinstance(message, HumanMessage)
        else message
        for message in state["messages"]
    ]
    messages = [SystemMessage(content=SYSTEM_PROMPT), *history]
    response = model.invoke(messages)

    retrieval_calls = state.get("retrieval_calls", 0)
    tool_calls = getattr(response, "tool_calls", None) or []
    new_retrievals = sum(
        1 for call in tool_calls if call.get("name") == "search_banking_docs"
    )

    return {
        "messages": [*history, response],
        "retrieval_calls": retrieval_calls + new_retrievals,
    }


def _build_graph():
    builder = StateGraph(ConciergeState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


graph = _build_graph()
