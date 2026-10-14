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


def test_short_title_scores_zero_seo_points():
    """Titles shorter than 40 chars must score 0 pts per the documented rubric."""
    from src.validate import calculate_seo_score
    desc = ProductDescription(
        sku="SHORT-001",
        title="Short Title",  # 11 chars — well below 40
        short_description="Valid short description.",
        long_description="A long description that is at least twenty characters long.",
        bullet_points=["B1", "B2", "B3"],
        meta_description="Valid meta description for SEO.",
        keywords_used=[],
        tone="casual",
        category="electronics",
    )
    score, breakdown = calculate_seo_score(desc, target_keywords=[])
    assert breakdown["title_length"]["score"] == 0, (
        "Titles <40 chars should score 0 SEO points per the documented rubric"
    )


def test_flesch_reading_ease_empty_or_whitespace():
    score, level = calculate_flesch_reading_ease("   ")
    assert score == 70.0
    assert "Optimal" in level


def test_flesch_reading_ease_dense_technical():
    technical_text = (
        "Thermodynamic electrodynamic microcontroller implementations necessitate "
        "multidimensional algorithmic synchronization and piezoelectric oscillation."
    )
    score, level = calculate_flesch_reading_ease(technical_text)
    assert score < 60.0
    assert "Complex" in level or "Dense" in level


def test_validation_warnings_when_not_enforced(sample_description):
    targets = ["nonexistent keyword xyz"]
    result = validate_product_description(sample_description, target_keywords=targets, enforce_keywords=False)
    # Should still be valid because enforce_keywords is False
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert any("nonexistent keyword xyz" in w for w in result.warnings)


def test_validation_warns_when_hero_keyword_missing():
    desc = ProductDescription(
        sku="TEST-HERO",
        title="AuraSound Headphones with Great Sound Quality",
        short_description="Clean sound and comfort.",
        long_description=(
            "Enjoy listening to music wherever you go with high fidelity acoustics and "
            "cushioned ear pads designed for continuous wear."
        ),
        bullet_points=["Battery", "Comfort", "Bluetooth"],
        meta_description="Meta description with wireless earbuds keyword included here for SEO testing.",
        keywords_used=[],
        tone="premium",
        category="electronics"
    )
    # "wireless earbuds" is only in meta_description, NOT in title or first 100 words of long_description
    result = validate_product_description(desc, target_keywords=["wireless earbuds"], enforce_keywords=False)
    assert any("No primary keyword found in title or first 100 words" in w for w in result.warnings)


def test_validate_constructed_invalid_lengths():
    # Use model_construct to bypass initial pydantic validator and test validate_product_description runtime guards
    desc = ProductDescription.model_construct(
        sku="OVERSIZED-01",
        title="A" * 75,
        short_description="B" * 170,
        long_description="A standard long description for the oversized product.",
        bullet_points=["Only one bullet"],
        meta_description="C" * 170,
        keywords_used=[],
        tone="casual",
        category="electronics"
    )
    result = validate_product_description(desc)
    assert result.is_valid is False
    assert any("Title exceeds 70" in e for e in result.errors)
    assert any("Short description exceeds 160" in e for e in result.errors)
    assert any("Meta description exceeds 160" in e for e in result.errors)
    assert any("Bullet points count must be 3-5" in e for e in result.errors)

