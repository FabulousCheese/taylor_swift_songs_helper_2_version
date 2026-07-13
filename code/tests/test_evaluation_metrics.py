"""
Unit tests for RAG evaluation metrics
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import (
    calculate_mrr,
    calculate_hit_rate,
    calculate_ndcg,
    calculate_f1,
    calculate_recall_at_k,
    normalize_text,
    normalize_for_comparison,
    check_song_hit,
)


class TestNormalizeText:
    def test_curly_quotes(self):
        curly = chr(0x2019)  # RIGHT SINGLE QUOTATION MARK
        text = f"I{curly}m happy"
        assert normalize_text(text) == "I’m happy"

    def test_em_dash(self):
        assert normalize_text("hello—world") == "hello-world"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_value(self):
        assert normalize_text(None) == ""


class TestNormalizeForComparison:
    def test_lowercase_and_strip(self):
        assert normalize_for_comparison("Love Story") == "lovestory"

    def test_remove_hyphens_and_quotes(self):
        assert normalize_for_comparison("Don't Blame Me") == "dontblameme"

    def test_unicode_normalization(self):
        assert normalize_for_comparison("Anti-Hero") == "antihero"


class TestCalculateMRR:
    def test_basic(self):
        assert calculate_mrr([1]) == 1.0

    def test_multiple_queries(self):
        result = calculate_mrr([1, 3, 2])
        expected = (1.0 + 1.0/3 + 1.0/2) / 3
        assert abs(result - expected) < 0.001

    def test_empty(self):
        assert calculate_mrr([]) == 0.0

    def test_zero_rank(self):
        assert calculate_mrr([0]) == 0.0


class TestCalculateHitRate:
    def test_hit_rate_basic(self):
        assert calculate_hit_rate([1, 0, 3]) == 2/3

    def test_hit_rate_at_k(self):
        hits = [1, 3, 6]
        assert calculate_hit_rate(hits, k=3) == 2/3
        assert calculate_hit_rate(hits, k=5) == 2/3

    def test_empty(self):
        assert calculate_hit_rate([]) == 0.0


class TestCalculateNDCG:
    def test_perfect_ranking(self):
        scores = [1, 1, 0, 0, 0]
        assert calculate_ndcg(scores) == 1.0

    def test_zero_relevance(self):
        scores = [0, 0, 0]
        assert calculate_ndcg(scores) == 0.0

    def test_partial_ranking_at_k(self):
        scores = [1, 0, 1, 0, 0]
        ndcg5 = calculate_ndcg(scores, k=5)
        assert 0 < ndcg5 < 1.0


class TestCalculateF1:
    def test_perfect(self):
        assert calculate_f1(1.0, 1.0) == 1.0

    def test_zero_both(self):
        assert calculate_f1(0, 0) == 0.0

    def test_partial(self):
        f1 = calculate_f1(0.5, 0.8)
        assert abs(f1 - 0.615) < 0.01


class TestCalculateRecallAtK:
    def test_full_recall(self):
        assert calculate_recall_at_k(5, 5) == 1.0

    def test_partial(self):
        assert calculate_recall_at_k(3, 5) == 0.6

    def test_zero_expected(self):
        assert calculate_recall_at_k(0, 0) == 0.0


class TestCheckSongHit:
    def test_exact_hit(self):
        result = check_song_hit(
            ["Love Story", "Blank Space", "Style"],
            ["Love Story"]
        )
        assert result["is_hit"] is True
        assert "Love Story" in result["hit_songs"]
        assert result["hit_rank"] == 1

    def test_no_hit(self):
        result = check_song_hit(
            ["Style", "Delicate"],
            ["Love Story"]
        )
        assert result["is_hit"] is False
        assert len(result["missed_songs"]) == 1

    def test_multiple_expected(self):
        result = check_song_hit(
            ["Love Story", "Style", "Delicate"],
            ["Love Story", "Style"]
        )
        assert len(result["hit_songs"]) == 2
        assert result["recall"] == 1.0
        assert result["precision"] == 2/3

    def test_partial_match_normalized(self):
        result = check_song_hit(
            ["Anti Hero", "Style"],
            ["Anti-Hero"]
        )
        assert result["is_hit"] is True
