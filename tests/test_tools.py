import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from concierge.mock_data import CUSTOMERS
from concierge.tools import account_lookup


class AccountLookupTests(unittest.TestCase):
    def test_lookup_preserves_contact_and_account_data_without_sensitive_fields(self):
        for customer_id, customer in CUSTOMERS.items():
            with self.subTest(customer_id=customer_id):
                result = account_lookup.invoke({"customer_id": customer_id})
                self.assertEqual(
                    set(result),
                    {
                        "customer_id",
                        "name",
                        "phone",
                        "email",
                        "accounts",
                        "ssn_last4",
                        "credit_cards",
                    },
                )
                for field in ("customer_id", "name", "phone", "email", "accounts"):
                    self.assertEqual(result[field], customer[field])
                self.assertEqual(result["ssn_last4"], customer["ssn"][-4:])
                self.assertEqual(
                    len(result["credit_cards"]), len(customer["credit_cards"])
                )
                serialized = json.dumps(result)
                self.assertNotIn(customer["ssn"], serialized)
                for card, stored in zip(
                    result["credit_cards"], customer["credit_cards"]
                ):
                    self.assertEqual(
                        card,
                        {
                            "brand": stored["brand"],
                            "last4": stored["number"].replace(" ", "")[-4:],
                            "exp": stored["exp"],
                        },
                    )
                    self.assertNotIn(stored["number"], serialized)
                    self.assertNotIn(stored["number"].replace(" ", ""), serialized)

    def test_lookup_extracts_last_four_after_removing_spaces(self):
        customer = deepcopy(CUSTOMERS["CUST-0001"])
        customer["credit_cards"] = [
            {
                "brand": "Visa",
                "number": "1234 5678 9012 3456 ",
                "cvv": "987",
                "exp": "08/29",
            },
            {
                "brand": "Visa",
                "number": "1234567890123456",
                "cvv": "987",
                "exp": "08/29",
            },
        ]
        original = deepcopy(customer)
        with patch.dict(CUSTOMERS, {"CUST-0001": customer}):
            result = account_lookup.invoke({"customer_id": "CUST-0001"})
        self.assertEqual(
            result["credit_cards"],
            [{"brand": "Visa", "last4": "3456", "exp": "08/29"}] * 2,
        )
        self.assertEqual(customer, original)

    def test_lookup_preserves_errors(self):
        with self.assertRaises(ValueError):
            account_lookup.invoke({"customer_id": "CUST-9999"})
        with self.assertRaises(RuntimeError):
            account_lookup.invoke({"customer_id": "X0001"})
