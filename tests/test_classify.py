from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# Make `starter/classify.py` importable without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

import classify  # noqa: E402  (must come after sys.path mutation)
from classify import ReviewClassification, cost_usd, PRICES
from evaluate import exact_match, jaccard  # noqa: E402


# ---------------------------------------------------------------------------
# cost_usd
# ---------------------------------------------------------------------------

class TestCostUsd:

    def test_known_model_input_only(self):
        """1 M input tokens at the model's input price."""
        in_p, _ = PRICES[classify.MODEL]
        assert cost_usd(1_000_000, 0) == pytest.approx(in_p)

    def test_known_model_output_only(self):
        """1 M output tokens at the model's output price."""
        _, out_p = PRICES[classify.MODEL]
        assert cost_usd(0, 1_000_000) == pytest.approx(out_p)

    def test_known_model_combined(self):
        in_p, out_p = PRICES[classify.MODEL]
        expected = (500_000 * in_p + 500_000 * out_p) / 1_000_000
        assert cost_usd(500_000, 500_000) == pytest.approx(expected)

    def test_zero_tokens_returns_zero(self):
        assert cost_usd(0, 0) == 0.0

    def test_unknown_model_falls_back_to_zero(self):
        """PRICES.get(MODEL, (0, 0)) must return 0 for an unpriced model."""
        with patch.object(classify, "MODEL", "unknown/model-xyz"):
            assert cost_usd(999_999, 999_999) == 0.0

    def test_cloud_model_has_positive_price(self):
        in_p, out_p = PRICES[classify.MODEL]
        assert in_p > 0 and out_p > 0

    def test_local_model_is_free(self):
        in_p, out_p = PRICES["llama3.2:3b"]
        assert in_p == 0.0 and out_p == 0.0


# ---------------------------------------------------------------------------
# ReviewClassification pydantic model
# ---------------------------------------------------------------------------

class TestReviewClassificationModel:

    def test_valid_single_category(self):
        rc = ReviewClassification(categories=["staff"], summary="Friendly staff overall")
        assert rc.categories == ["staff"]
        assert rc.summary == "Friendly staff overall"

    def test_valid_multiple_categories(self):
        rc = ReviewClassification(
            categories=["location", "food"],
            summary="Great location, average breakfast",
        )
        assert "location" in rc.categories
        assert "food" in rc.categories
        assert len(rc.categories) == 2

    def test_invalid_category_raises(self):
        with pytest.raises(ValidationError):
            ReviewClassification(categories=["parking"], summary="Rooms were spotless")

    def test_missing_summary_raises(self):
        with pytest.raises(ValidationError):
            ReviewClassification(categories=["other"])

    def test_all_valid_categories_accepted(self):
        categories = ("staff", "location", "food", "amenities", "room", "cleanliness", "price", "other")
        for cat in categories:
            rc = ReviewClassification(categories=[cat], summary="test")
            assert rc.categories == [cat]


# ---------------------------------------------------------------------------
# evaluate.py metrics
# ---------------------------------------------------------------------------

class TestExactMatch:

    def test_identical_sets_match(self):
        assert exact_match({"staff", "food"}, {"staff", "food"})

    def test_different_sets_do_not_match(self):
        assert not exact_match({"staff"}, {"staff", "food"})

    def test_both_empty_match(self):
        assert exact_match(set(), set())


class TestJaccard:

    def test_identical_sets_is_one(self):
        assert jaccard({"staff", "food"}, {"staff", "food"}) == 1.0

    def test_disjoint_sets_is_zero(self):
        assert jaccard({"staff"}, {"food"}) == 0.0

    def test_partial_overlap(self):
        # {staff, food} vs {staff, room} -> intersection 1, union 3
        assert jaccard({"staff", "food"}, {"staff", "room"}) == pytest.approx(1 / 3)

    def test_both_empty_is_one(self):
        assert jaccard(set(), set()) == 1.0
