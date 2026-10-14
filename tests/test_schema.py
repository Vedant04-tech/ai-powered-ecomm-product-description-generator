"""Unit tests for src/schema.py ProductDescription Pydantic model."""

import pytest
from pydantic import ValidationError
from src.schema import ProductDescription


def test_valid_product_description():
    """Verify that a compliant product description passes validation."""
    desc = ProductDescription(
        sku="TEST-001",
        title="Premium Wireless Noise-Cancelling Earbuds",
        short_description="Immerse yourself in studio-grade sound with 40-hour battery life.",
        long_description=(
            "Designed for discerning audiophiles, these earbuds feature custom-tuned 11mm "
            "graphene drivers and adaptive active noise cancellation. Enjoy ultra-low latency "
            "streaming and crystal-clear calls wherever life takes you."
        ),
        bullet_points=[
            "40-hour total battery life with wireless charging case",
            "Adaptive Active Noise Cancellation with transparency mode",
            "IPX5 water and sweat resistance for high-intensity workouts"
        ],
        meta_description="Shop premium wireless earbuds with active noise cancellation and 40h battery.",
        keywords_used=["wireless earbuds", "noise cancelling"],
        tone="premium",
        category="electronics"
    )
    assert desc.sku == "TEST-001"
    assert len(desc.title) <= 70
    assert len(desc.short_description) <= 160
    assert len(desc.meta_description) <= 160
    assert 3 <= len(desc.bullet_points) <= 5


def test_title_too_long_fails():
    """Verify that a title over 70 characters triggers a validation error."""
    long_title = "A" * 71
    with pytest.raises(ValidationError):
        ProductDescription(
            sku="FAIL-001",
            title=long_title,
            short_description="Valid short description",
            long_description="A long description that is at least twenty characters long.",
            bullet_points=["Bullet 1", "Bullet 2", "Bullet 3"],
            meta_description="Valid meta description",
            tone="premium",
            category="electronics"
        )


def test_short_description_too_long_fails():
    """Verify that short description over 160 characters triggers a validation error."""
    long_desc = "B" * 161
    with pytest.raises(ValidationError):
        ProductDescription(
            sku="FAIL-002",
            title="Valid Title",
            short_description=long_desc,
            long_description="A long description that is at least twenty characters long.",
            bullet_points=["Bullet 1", "Bullet 2", "Bullet 3"],
            meta_description="Valid meta description",
            tone="premium",
            category="electronics"
        )


def test_meta_description_too_long_fails():
    """Verify that meta description over 160 characters triggers a validation error."""
    long_meta = "C" * 161
    with pytest.raises(ValidationError):
        ProductDescription(
            sku="FAIL-003",
            title="Valid Title",
            short_description="Valid short description",
            long_description="A long description that is at least twenty characters long.",
            bullet_points=["Bullet 1", "Bullet 2", "Bullet 3"],
            meta_description=long_meta,
            tone="premium",
            category="electronics"
        )


def test_bullet_points_count_constraints():
    """Verify that bullet points must have between 3 and 5 items."""
    # Under 3 bullets
    with pytest.raises(ValidationError):
        ProductDescription(
            sku="FAIL-004",
            title="Valid Title",
            short_description="Valid short description",
            long_description="A long description that is at least twenty characters long.",
            bullet_points=["Only One", "Only Two"],
            meta_description="Valid meta description",
            tone="premium",
            category="electronics"
        )

    # Over 5 bullets
    with pytest.raises(ValidationError):
        ProductDescription(
            sku="FAIL-005",
            title="Valid Title",
            short_description="Valid short description",
            long_description="A long description that is at least twenty characters long.",
            bullet_points=["B1", "B2", "B3", "B4", "B5", "B6"],
            meta_description="Valid meta description",
            tone="premium",
            category="electronics"
        )
