"""Unit tests for src/validate.py validation, SEO scoring, and readability logic."""

import pytest
from src.schema import ProductDescription
from src.validate import (
    validate_product_description,
    check_keyword_in_text,
    calculate_flesch_reading_ease,
    calculate_seo_score
)


@pytest.fixture
def sample_description():
    return ProductDescription(
        sku="TEST-ELEC",
        title="AuraSound X200 Wireless Earbuds with Active Noise Cancellation",
        short_description="High-fidelity wireless earbuds with noise cancelling technology.",
        long_description=(
            "The Wireless Earbuds X200 provide world-class acoustic engineering with powerful "
            "active noise cancellation. Perfect for daily commutes, travel, and fitness. "
            "Designed for all-day comfort and crystal-clear calls wherever you go."
        ),
        bullet_points=[
            "Up to 40 hours total playback time",
            "Advanced hybrid active noise cancellation",
            "IPX5 water-resistant sweatproof design"
        ],
        meta_description="Discover AuraSound X200 wireless earbuds with active noise cancellation and 40h battery. Elevate your everyday audio experience today.",
        keywords_used=[],
        tone="premium",
        category="electronics"
    )


def test_keyword_containment_word_boundary():
    text = "Experience the best wireless earbuds with active noise cancellation."
    assert check_keyword_in_text("wireless earbuds", text)
    assert check_keyword_in_text("noise cancellation", text)
    assert not check_keyword_in_text("ergonomic keyboard", text)
    # Ensure disjoint words do not trigger a false match
    disjoint_text = "The wireless connection was great. Later I cleaned my earbuds."
    # Since 'wireless' and 'earbuds' are separated by a sentence, phrase match should be False
    assert not check_keyword_in_text("wireless earbuds", disjoint_text)


def test_flesch_reading_ease():
    sample_text = (
        "The quick brown fox jumps over the lazy dog. It is an easy afternoon. "
        "Simple words make this very accessible to every reader."
    )
    score, tier = calculate_flesch_reading_ease(sample_text)
    assert 0.0 <= score <= 100.0
    assert isinstance(tier, str)
    assert len(tier) > 0


def test_seo_scorecard_calculation(sample_description):
    targets = ["wireless earbuds", "active noise cancellation"]
    score, breakdown = calculate_seo_score(sample_description, targets)
    assert 0 <= score <= 100
    assert "title_length" in breakdown
    assert "meta_description_length" in breakdown
    assert "primary_kw_in_title" in breakdown
    assert "primary_kw_in_hero" in breakdown
    assert breakdown["primary_kw_in_title"]["passed"] is True


def test_validation_passes_when_keywords_present(sample_description):
    targets = ["wireless earbuds", "active noise cancellation"]
    result = validate_product_description(sample_description, target_keywords=targets)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.seo_score >= 80
    assert result.readability_score > 0
    assert set(sample_description.keywords_used) == {"wireless earbuds", "active noise cancellation"}


def test_validation_fails_when_keyword_missing(sample_description):
    targets = ["wireless earbuds", "missing keyword xyz"]
    result = validate_product_description(sample_description, target_keywords=targets, enforce_keywords=True)
    assert result.is_valid is False
    assert any("missing keyword xyz" in err for err in result.errors)
    assert result.retry_prompt_feedback is not None
    assert "missing keyword xyz" in result.retry_prompt_feedback
