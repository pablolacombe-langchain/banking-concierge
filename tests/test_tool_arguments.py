import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from concierge import context
from concierge.mock_data import TRANSACTIONS
from concierge.prompts import SYSTEM_PROMPT, TOOL_ARGUMENT_RULES
from concierge.state import ConciergeState
from concierge.tools import (
    TOOLS,
    TransactionLimitError,
    account_lookup,
    find_branch,
    recent_transactions,
)

with patch.object(context, "get_prompt", return_value="Test instructions"):
    graph_module = importlib.import_module("concierge.graph")


class ToolArgumentsTest(unittest.TestCase):
    def test_hub_prompt_keeps_local_rules_without_duplication(self):
        for content in ("Hub instructions", SYSTEM_PROMPT):
            with (
                self.subTest(content=content),
                patch.object(context, "Client") as client,
            ):
                client.return_value.pull_agent.return_value = SimpleNamespace(
                    files={"AGENTS.md": SimpleNamespace(content=content)}
                )
                result = context.get_prompt()
                self.assertTrue(result.startswith(content))
                self.assertEqual(result.count(TOOL_ARGUMENT_RULES), 1)

    def test_hub_failure_uses_local_fallback(self):
        with patch.object(context, "Client", side_effect=RuntimeError("offline")):
            self.assertEqual(context.get_prompt(), SYSTEM_PROMPT)

    def test_invalid_customer_inputs_do_not_return_records(self):
        for customer_id in (
            "Alex Rivera",
            "552-19-4488",
            "(415) 555-0142",
            "CUST-1",
            "CUST-00001",
            "CUST-０００１",
            "CUST-0001\n",
            "",
        ):
            for lookup in (account_lookup, recent_transactions):
                with (
                    self.subTest(customer_id=customer_id, tool=lookup.name),
                    self.assertRaisesRegex(ValueError, "Ask the representative"),
                ):
                    lookup.invoke({"customer_id": customer_id})

    def test_valid_customer_id_returns_intended_customer(self):
        result = account_lookup.invoke({"customer_id": "CUST-0001"})
        self.assertEqual(result["customer_id"], "CUST-0001")

    def test_unknown_valid_id_is_distinct_from_format_failure(self):
        with self.assertRaisesRegex(ValueError, "No customer found"):
            account_lookup.invoke({"customer_id": "CUST-9999"})

    def test_invalid_zip_is_not_padded(self):
        for zip_code in ("1234", "123456", "９４１０５", "94105\n", ""):
            with (
                self.subTest(zip_code=zip_code),
                self.assertRaisesRegex(ValueError, "Ask the representative"),
            ):
                find_branch.invoke({"zip_code": zip_code})

    def test_transaction_limits_are_rejected_without_silent_capping(self):
        for limit in (0, -1, 51, 200):
            with self.subTest(limit=limit), self.assertRaises(TransactionLimitError):
                recent_transactions.invoke({"customer_id": "CUST-0001", "limit": limit})
        result = recent_transactions.invoke({"customer_id": "CUST-0001", "limit": 50})
        self.assertEqual(result, TRANSACTIONS["CUST-0001"][:50])
        self.assertLess(len(result), 50)

    def test_tool_node_preserves_errors_and_separates_limit_recovery(self):
        builder = StateGraph(ConciergeState)
        builder.add_node(
            "tools", ToolNode(TOOLS, handle_tool_errors=graph_module._handle_tool_error)
        )
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        node = builder.compile()
        for name, args, expected in (
            ("account_lookup", {"customer_id": "CUST-1"}, "Ask the representative"),
            ("find_branch", {"zip_code": "1234"}, "do not invent or pad"),
            (
                "recent_transactions",
                {"customer_id": "CUST-0001", "limit": 200},
                "limit 200 exceeds the maximum of 50",
            ),
        ):
            with self.subTest(tool=name):
                result = node.invoke(
                    ConciergeState(
                        retrieval_calls=0,
                        messages=[
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {"name": name, "args": args, "id": "lookup"}
                                ],
                            )
                        ],
                    )
                )
                message = result["messages"][-1]
                self.assertEqual(message.status, "error")
                self.assertEqual(message.tool_call_id, "lookup")
                self.assertIn(expected, message.content)
                self.assertNotIn("Please fix your mistakes", message.content)
                if name == "recent_transactions":
                    self.assertIn("disclose any cap", message.content)
                    self.assertIn("actual returned and listed counts", message.content)
                else:
                    self.assertIn("ask the representative", message.content)
                    self.assertIn(
                        "does not mean the customer is absent", message.content
                    )


if __name__ == "__main__":
    unittest.main()
