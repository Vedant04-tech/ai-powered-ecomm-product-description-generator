"""ProductDescription schema definition conforming to PRD.md §7."""

from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, field_validator


class ProductDescription(BaseModel):
    """Structured, validated eCommerce product description.
    
    Adheres strictly to the contract defined in PRD.md §7:
    - sku: Unique product identifier
    - title: <= 70 chars, SEO-friendly
    - short_description: <= 160 chars
    - long_description: Comprehensive product narrative
    - bullet_points: 3-5 feature bullets
    - meta_description: <= 160 chars for search engines
    - keywords_used: List of target keywords successfully incorporated
    - tone: Tone used (e.g., premium, casual, technical)
    - category: Product category (e.g., electronics, apparel, home_goods)
    """

    sku: str = Field(..., description="Unique product SKU or identifier")
    title: str = Field(..., max_length=70, description="SEO-friendly title, maximum 70 characters")
    short_description: str = Field(..., max_length=160, description="Concise summary, maximum 160 characters")
    long_description: str = Field(..., min_length=20, description="Engaging, multi-paragraph product copy")
    bullet_points: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Key feature and benefit bullets, between 3 and 5 items"
    )
    meta_description: str = Field(..., max_length=160, description="Search engine meta description, maximum 160 characters")
    keywords_used: List[str] = Field(default_factory=list, description="Target keywords used in the copy")
    tone: str = Field(..., description="Tone applied (e.g. premium, casual, technical)")
    category: str = Field(..., description="Product category (e.g. electronics, apparel, home_goods)")

    @field_validator("title", "short_description", "meta_description", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("bullet_points")
    @classmethod
    def validate_bullets(cls, v: List[str]) -> List[str]:
        cleaned = [b.strip() for b in v if isinstance(b, str) and b.strip()]
        if len(cleaned) < 3 or len(cleaned) > 5:
            raise ValueError(f"bullet_points must contain between 3 and 5 items, got {len(cleaned)}")
        return cleaned
