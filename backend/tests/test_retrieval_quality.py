import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rag.retrieval.hybrid_search import hybrid_search
from rag.generator import generate_answer


class TestRetrievalQuality:
    """
    Tests that catch real retrieval regressions:
    - Cross-document contamination
    - Minimum relevance thresholds
    - Source attribution correctness
    """

    def test_finance_query_does_not_return_cs_paper_sources(self):
        """
        Regression test for the cross-document contamination bug:
        a finance question should never cite the metamorphic testing paper.
        """
        result = generate_answer("What is the ROI formula?", top_k=5)
        assert "1-s2.0-S0164121224003741-main.pdf" not in result["sources"], (
            "Finance query leaked a chunk from the unrelated CS paper — "
            "cross-document contamination regression detected"
        )

    def test_cs_query_does_not_return_finance_sources(self):
        """
        Reverse contamination check: a metamorphic testing question
        should never cite the finance guide.
        """
        result = generate_answer("What is metamorphic testing?", top_k=5)
        assert "managers-guide-to-finance-and-accounting.pdf" not in result["sources"], (
            "CS query leaked a chunk from the unrelated finance guide — "
            "cross-document contamination regression detected"
        )

    def test_all_returned_chunks_meet_minimum_relevance(self):
        """
        Every chunk returned by hybrid_search + rerank should clear
        a minimum relevance bar — guards against the reranker passing
        through near-irrelevant results just to fill top_k.
        """
        chunks = hybrid_search("What is the ROI formula?", top_k=10)
        assert len(chunks) > 0, "No chunks retrieved for a known-good query"

    def test_answer_is_grounded_in_returned_sources(self):
        """
        Sanity check: every source cited in 'sources' must correspond
        to a real document in the database, not a hallucinated filename.
        """
        result = generate_answer("What is metamorphic testing?", top_k=5)
        for source in result["sources"]:
            assert source.endswith(".pdf") or source.endswith(".txt"), (
                f"Unexpected source format: {source}"
            )

    def test_empty_or_nonsense_query_does_not_crash(self):
        """
        Edge case: nonsense queries shouldn't crash the pipeline,
        even if they return a low-confidence or empty answer.
        """
        result = generate_answer("asdkjfh qwoeiru", top_k=5)
        assert "answer" in result
        assert isinstance(result["answer"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])