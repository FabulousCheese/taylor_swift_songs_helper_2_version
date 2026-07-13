"""
Tests for retrieval search logic (pure functions, no LLM/index dependencies)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from rag.retrieval.retrieval_search import RetrievalSearch


class TestReciprocalRankFusion:
    """Test RRF algorithm with synthetic ranked lists"""

    def _make_doc(self, content):
        return Document(page_content=content, metadata={})

    def test_rrf_basic(self):
        retriever = RetrievalSearch()
        doc_a = self._make_doc("doc A content here")
        doc_b = self._make_doc("doc B content here")
        doc_c = self._make_doc("doc C content here")

        # Two ranked lists with different orderings
        list1 = [(doc_a, 0.9), (doc_b, 0.7), (doc_c, 0.5)]
        list2 = [(doc_b, 0.8), (doc_c, 0.6), (doc_a, 0.4)]

        fused = retriever._reciprocal_rank_fusion([list1, list2], k=60)

        assert len(fused) == 3
        # doc_b appears at rank 1 and rank 1 → strongest fused score
        fused_contents = [doc.page_content for doc, score in fused]
        assert fused_contents[0] == "doc B content here"

    def test_rrf_unique_docs(self):
        retriever = RetrievalSearch()
        doc_a = self._make_doc("unique doc A xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        doc_b = self._make_doc("unique doc B xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        list1 = [(doc_a, 0.9)]
        list2 = [(doc_b, 0.8)]

        fused = retriever._reciprocal_rank_fusion([list1, list2], k=60)

        assert len(fused) == 2

    def test_rrf_empty(self):
        retriever = RetrievalSearch()
        fused = retriever._reciprocal_rank_fusion([], k=60)
        assert fused == []


class TestSmartRouteIntent:
    def test_lyrics_keyword(self):
        rs = RetrievalSearch()
        assert rs.smart_route_intent("what are the lyrics of love story") == "lyrics"

    def test_theme_keyword(self):
        rs = RetrievalSearch()
        assert rs.smart_route_intent("recommend some happy songs") == "theme"

    def test_default_to_theme(self):
        rs = RetrievalSearch()
        assert rs.smart_route_intent("what is the meaning of life") == "theme"


class TestWantsFullLyrics:
    def test_detects_full_lyrics(self):
        rs = RetrievalSearch()
        assert rs.wants_full_lyrics("give me the complete lyrics") is True
        assert rs.wants_full_lyrics("show me the whole song") is True

    def test_no_full_lyrics(self):
        rs = RetrievalSearch()
        assert rs.wants_full_lyrics("what song has the word love") is False
