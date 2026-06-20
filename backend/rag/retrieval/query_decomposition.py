import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
from typing import List


def decompose_query(query: str) -> List[str]:
    """
    Rule-based query decomposition. Detects comma/and-separated items
    (e.g. "MR1, MR2, and MR3") and generates one sub-question per item.
    More reliable than LLM-based decomposition for small local models,
    which struggle with structured multi-part instructions.
    """
    # Look for patterns like "X, Y, and Z" or "X, Y, Z"
    # First, find what's being asked about each item
    match = re.search(
        r"(explain|describe|what (?:is|are)|list|define)\s+(.+)",
        query, re.IGNORECASE
    )

    if not match:
        return [query]

    verb = match.group(1)
    items_text = match.group(2)

    # Split on commas and "and"
    items_text = re.sub(r"\s+and\s+", ",", items_text)
    items = [item.strip().rstrip("?.!") for item in items_text.split(",")]
    items = [item for item in items if item]

    if len(items) <= 1:
        return [query]

    # Generate one sub-question per item, capped at 5
    sub_queries = [f"{verb} {item}" for item in items[:5]]
    return sub_queries


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_decomposition.py '<question>'")
        sys.exit(1)

    query = sys.argv[1]
    sub_queries = decompose_query(query)

    print(f"\nOriginal: {query}")
    print(f"\nDecomposed into {len(sub_queries)} sub-queries:")
    for i, q in enumerate(sub_queries):
        print(f"  {i+1}. {q}")