"""Tools available to the Meridian National customer service concierge agent.

A few tools have deliberate rough edges so LangSmith Engine has something
to cluster after the load generator runs:

- search_banking_docs has a vague description so the model occasionally
  re-queries multiple times rephrasing
- account_lookup raises on malformed customer IDs and on IDs prefixed with
  "X" (simulated downstream outage)
- recent_transactions raises if the model passes a runaway limit
- find_branch raises on non-zip inputs
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from concierge.mock_data import (
    BRANCHES,
    CUSTOMERS,
    TRANSACTIONS,
    find_branch_by_zip,
)
from concierge.retrieval import retrieve


@tool
def search_banking_docs(query: str, k: int = 4) -> str:
    """Search Meridian National banking documentation.

    Args:
        query: A natural-language search query.
        k: Number of relevant chunks to return. Defaults to 4.
    """
    chunks = retrieve(query, k=k)
    if not chunks:
        return "No relevant documentation found."
    blocks = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        blocks.append(f"[source: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(blocks)


class TransactionLimitError(ValueError):
    """A transaction count outside the supported range."""


def _validate_customer_id(customer_id: str) -> None:
    """Require an exact customer ID before accessing customer records."""
    if not re.fullmatch(r"CUST-[0-9]{4}", customer_id):
        raise ValueError(
            "Customer IDs must be CUST- followed by exactly four digits. "
            "Ask the representative for the exact customer ID; do not construct, "
            "pad, guess, or enumerate IDs. This format failure does not establish "
            "whether the customer exists."
        )


@tool
def account_lookup(customer_id: str) -> dict:
    """Look up accounts using the exact CUST-#### ID supplied by the representative or a tool for the intended customer; ask if missing or invalid, never construct, pad, guess, or enumerate IDs."""
    if customer_id.startswith("X"):
        raise RuntimeError(
            "Customer record service is temporarily unavailable. Try again later."
        )
    _validate_customer_id(customer_id)
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise ValueError(f"No customer found with ID {customer_id!r}.")
    return dict(customer)


@tool
def recent_transactions(customer_id: str, limit: int = 5) -> list[dict]:
    """Retrieve up to limit transactions (default 5, maximum 50) using the exact CUST-#### ID supplied by the representative or a tool for the intended customer; ask if missing or invalid, never construct, pad, guess, or enumerate IDs; disclose caps and report actual returned and listed counts."""
    _validate_customer_id(customer_id)
    if limit <= 0:
        raise TransactionLimitError("limit must be positive")
    if limit > 50:
        raise TransactionLimitError(f"limit {limit} exceeds the maximum of 50.")
    if customer_id not in CUSTOMERS:
        raise ValueError(f"No customer found with ID {customer_id!r}.")
    txs = TRANSACTIONS.get(customer_id, [])
    return [dict(t) for t in txs[:limit]]


@tool
def find_branch(zip_code: str) -> dict:
    """Find a branch using the representative's exact five-digit U.S. ZIP; ask for clarification if missing or invalid, never invent or pad it."""
    if not re.fullmatch(r"[0-9]{5}", zip_code):
        raise ValueError(
            "zip_code must be a five-digit U.S. ZIP code. "
            "Ask the representative for the exact ZIP; do not invent or pad it."
        )
    branch = find_branch_by_zip(zip_code)
    if branch is None:
        return {
            "match": False,
            "message": "No Meridian National branch found in our directory for that ZIP code.",
            "nearest_known": BRANCHES[0],
        }
    return {"match": True, **branch}


@tool
def transfer_funds(from_account: str, to_account: str, amount: float) -> dict:
    """Initiate a transfer between two Meridian National accounts owned by the same customer.

    Args:
        from_account: The source account ID.
        to_account: The destination account ID.
        amount: The dollar amount to transfer.
    """
    if amount <= 0:
        raise ValueError("amount must be positive")
    confirmation = f"MNB-XFER-{abs(hash((from_account, to_account, amount))) % 10_000_000:07d}"
    return {
        "status": "submitted",
        "from_account": from_account,
        "to_account": to_account,
        "amount": round(amount, 2),
        "confirmation": confirmation,
        "estimated_post": "immediately",
    }


TOOLS = [
    search_banking_docs,
    account_lookup,
    recent_transactions,
    find_branch,
    transfer_funds,
]
