import asyncio
import importlib
import json
import unittest
from unittest.mock import patch

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, traceable, tracing_context

with (
    patch("langsmith.Client.pull_agent", side_effect=RuntimeError("offline")),
    patch(
        "langsmith.Client",
        side_effect=lambda **kwargs: Client(
            api_key="test", auto_batch_tracing=False, **kwargs
        ),
    ),
):
    graph_module = importlib.import_module("concierge.graph")


RAW = "4111 1111 1111 1111, SSN 123-45-6789, CVV: 987"
SAFE = "[card ending 1111], SSN [SSN ending 6789], [CVV removed]"


class RedactionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            "os.environ",
            {
                "LANGSMITH_TRACING": "false",
                "LANGCHAIN_TRACING_V2": "false",
                "LANGCHAIN_CALLBACKS_BACKGROUND": "false",
            },
        )
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_card_numbers(self):
        for card in (
            "4111111111111111",
            "4111 1111 1111 1111",
            "4111-1111-1111-1111",
            "4222222222222",
            "378282246310005",
            "4000000000000000006",
        ):
            with self.subTest(card=card):
                digits = card.replace(" ", "").replace("-", "")
                self.assertEqual(
                    graph_module._redact_text(card), f"[card ending {digits[-4:]}]"
                )
        for value in (
            "4111111111111112",
            "411111111111",
            "41111111111111111111",
            "4111 1111 1111 1111 1111",
            "CUST-0001",
            "12345",
            "reference 1234567890123456",
        ):
            with self.subTest(value=value):
                self.assertEqual(graph_module._redact_text(value), value)

    def test_ssn_and_security_codes(self):
        self.assertEqual(graph_module._redact_text(RAW), SAFE)
        for label in ("cvv", "CVC", "Security-Code", "SECURITY CODE"):
            for separator in (" ", ": ", "=", " is "):
                for code in ("123", "1234"):
                    with self.subTest(label=label, separator=separator, code=code):
                        self.assertEqual(
                            graph_module._redact_text(label + separator + code),
                            "[CVV removed]",
                        )
        for value in ("123456789", "12-345-6789", "CVV: 12 CVC: 12345 order 1234"):
            self.assertEqual(graph_module._redact_text(value), value)
        self.assertEqual(graph_module._redact_text(SAFE), SAFE)

    def test_message_metadata_and_non_human_messages(self):
        human = HumanMessage(
            content=[{"type": "text", "text": RAW}],
            id="human",
            name="rep",
            additional_kwargs={"source": "chat"},
            response_metadata={"received": True},
        )
        others = [
            SystemMessage(content="system", id="system"),
            AIMessage(content="assistant", id="ai"),
            ToolMessage(content="tool", tool_call_id="call", id="tool"),
        ]
        seen = []

        def model(messages):
            seen.extend(messages)
            return AIMessage(content="done")

        with patch.object(
            graph_module, "_make_model", return_value=RunnableLambda(model)
        ):
            result = graph_module.agent_node({"messages": [human, *others]})
        sanitized = result["messages"][0]
        self.assertEqual(sanitized.content, [{"type": "text", "text": SAFE}])
        self.assertEqual(
            sanitized.model_dump(exclude={"content"}),
            human.model_dump(exclude={"content"}),
        )
        self.assertEqual(human.content, [{"type": "text", "text": RAW}])
        for actual, expected in zip(result["messages"][1:-1], others):
            self.assertIs(actual, expected)
        self.assertEqual(seen[1].content, sanitized.content)

    def test_graph_history_tools_and_active_tracing(self):
        sent = []
        model_inputs = []
        tool_inputs = []

        @tool
        def echo(value: str) -> str:
            """Echo the supplied value."""
            tool_inputs.append(value)
            return value

        @traceable
        def traced_payload(value):
            return {"nested": [value]}

        def model(messages):
            traced_payload(RAW)
            model_inputs.append(messages)
            if not isinstance(messages[-1], ToolMessage):
                human = next(
                    m for m in reversed(messages) if isinstance(m, HumanMessage)
                )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "echo-call",
                            "name": "echo",
                            "args": {"value": human.content},
                        }
                    ],
                )
            return AIMessage(content="done")

        def capture(session, request, **kwargs):
            if "/runs" in request.url:
                sent.append((request.method, request.body))
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            return response

        with (
            patch.object(graph_module, "TOOLS", [echo]),
            patch.object(
                graph_module, "_make_model", return_value=RunnableLambda(model)
            ),
            patch.object(requests.Session, "send", autospec=True, side_effect=capture),
            tracing_context(enabled=True),
        ):
            graph = graph_module._build_graph()
            graph.checkpointer = InMemorySaver()
            config = {"configurable": {"thread_id": "redaction-test"}}
            first = graph.invoke(
                {"messages": [HumanMessage(content=RAW, id="first")]}, config
            )
            second = asyncio.run(
                graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(content="Continue: " + RAW, id="second")
                        ]
                    },
                    config,
                )
            )

        self.assertEqual(
            [m.content for m in first["messages"] if isinstance(m, HumanMessage)],
            [SAFE],
        )
        self.assertEqual(
            [m.content for m in second["messages"] if isinstance(m, HumanMessage)],
            [SAFE, "Continue: " + SAFE],
        )
        self.assertEqual(tool_inputs, [SAFE, "Continue: " + SAFE])
        for messages in model_inputs:
            self.assertNotIn(RAW, repr(messages))
        self.assertTrue(sent)
        payloads = []
        roots = []
        for method, payload in sent:
            if isinstance(payload, (bytes, str)):
                payload = json.loads(payload)
            payloads.append(payload)
            if method == "POST" and payload.get("name") == "LangGraph":
                roots.append(payload)
        serialized = json.dumps(payloads, default=str)
        for secret in ("4111 1111 1111 1111", "123-45-6789", "987"):
            self.assertNotIn(secret, serialized)
        self.assertEqual(len(roots), 2)
        self.assertIn(SAFE, json.dumps(roots, default=str))
        self.assertTrue(any(p.get("parent_run_id") and "inputs" in p for p in payloads))
        self.assertTrue(any("outputs" in p for p in payloads))


if __name__ == "__main__":
    unittest.main()
