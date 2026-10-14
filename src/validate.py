"""Validation, SEO scoring, and readability check logic for ProductDescription models."""

from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from src.schema import ProductDescription

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of validating a ProductDescription instance with SEO & readability metrics."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retry_prompt_feedback: Optional[str] = None
    seo_score: int = 100
    seo_breakdown: Dict[str, Any] = field(default_factory=dict)
    readability_score: float = 70.0
    readability_level: str = "Standard (Optimal for eCommerce)"


def normalize_text(text: str) -> str:
    """Lowercase and strip excessive punctuation for text processing."""
    return re.sub(r"[^\w\s-]", " ", text.lower())


def count_syllables_in_word(word: str) -> int:
    """Estimate syllables in an English word using phonetic heuristics."""
    clean = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not clean:
        return 1
    if len(clean) <= 3:
        return 1
    
    # Remove trailing silent 'e' unless preceded by 'l' (e.g., 'table')
    if clean.endswith("e") and not clean.endswith("le") and not clean.endswith("ee"):
        clean = clean[:-1]
    
    # Count vowel groups
    vowel_runs = re.findall(r"[aeiouy]+", clean)
    count = len(vowel_runs)
    return max(1, count)


def calculate_flesch_reading_ease(text: str) -> Tuple[float, str]:
    """Calculate the Flesch Reading Ease score (0 to 100+) and its readability tier.
    
    Formula: 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    Ideal eCommerce marketing copy scores between 60.0 and 75.0 (engaging, accessible).
    """
    clean_text = text.strip()
    if not clean_text:
        return 70.0, "Standard (Optimal for eCommerce)"

    # Split sentences by punctuation (. ! ? ;)
    raw_sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    sentence_count = max(1, len(raw_sentences))

    # Split words
    words = [w.strip() for w in re.split(r"[\s\-_]+", clean_text) if w.strip()]
    word_count = max(1, len(words))

    # Count syllables
    total_syllables = sum(count_syllables_in_word(w) for w in words)

    asl = word_count / sentence_count
    asw = total_syllables / word_count

    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    clamped_score = max(0.0, min(100.0, round(score, 1)))

    if clamped_score >= 80:
        level = "Very Easy (Casual / Conversational)"
    elif clamped_score >= 60:
        level = "Standard (Optimal for eCommerce)"
    elif clamped_score >= 45:
        level = "Moderately Complex (Technical / Spec-Heavy)"
    else:
        level = "Dense (Academic / Engineering)"

    return clamped_score, level


def check_keyword_in_text(keyword: str, text: str) -> bool:
    """Verify if target keyword appears using word-boundary regex phrase matching."""
    cleaned_kw = keyword.strip().lower()
    if not cleaned_kw:
        return True

    text_lower = text.lower()
    # Create word-boundary pattern allowing flexible internal whitespace
    escaped_tokens = [re.escape(t) for t in cleaned_kw.split() if t]
    if not escaped_tokens:
        return True

    pattern = r"\b" + r"\s+".join(escaped_tokens) + r"\b"
    return bool(re.search(pattern, text_lower))


def calculate_seo_score(
    desc: ProductDescription,
    target_keywords: Optional[List[str]] = None
) -> Tuple[int, Dict[str, Any]]:
    """Compute a comprehensive 100-point SEO quality score for eCommerce listings.
    
    Rubric:
    - Title Length (50-70 chars): +20 pts (40-49 chars: +14 pts, <40 or >70: +0 pts)
    - Meta Description Length (120-160 chars): +20 pts (90-119 chars: +14 pts, <90 or >160: +0 pts)
    - Primary Keyword in Title: +20 pts
    - Primary Keyword in First 100 Words: +20 pts
    - Secondary Keywords Coverage: +10 pts
    - 3-5 Feature Bullets: +10 pts
    """
    points = 0
    breakdown: Dict[str, Any] = {}

    # 1. Title sweet spot
    title_len = len(desc.title)
    if 50 <= title_len <= 70:
        title_pts = 20
    elif 40 <= title_len < 50:
        title_pts = 14
    elif title_len <= 70:
        title_pts = 8
    else:
        title_pts = 0
    points += title_pts
    breakdown["title_length"] = {"score": title_pts, "max": 20, "actual_chars": title_len}

    # 2. Meta description sweet spot
    meta_len = len(desc.meta_description)
    if 120 <= meta_len <= 160:
        meta_pts = 20
    elif 90 <= meta_len < 120:
        meta_pts = 14
    elif meta_len <= 160:
        meta_pts = 8
    else:
        meta_pts = 0
    points += meta_pts
    breakdown["meta_description_length"] = {"score": meta_pts, "max": 20, "actual_chars": meta_len}

    # 3. Keywords analysis
    cleaned_targets = [k.strip() for k in (target_keywords or []) if k.strip()]
    primary_kw = cleaned_targets[0] if cleaned_targets else None

    # Primary in Title
    if primary_kw:
        primary_in_title = check_keyword_in_text(primary_kw, desc.title)
        p_title_pts = 20 if primary_in_title else 0
    else:
        p_title_pts = 20  # No target specified
        primary_in_title = True
    points += p_title_pts
    breakdown["primary_kw_in_title"] = {"score": p_title_pts, "max": 20, "passed": primary_in_title}

    # Primary in first 100 words of long description
    if primary_kw:
        first_100_words = " ".join(desc.long_description.split()[:100])
        primary_in_hero = check_keyword_in_text(primary_kw, first_100_words)
        p_hero_pts = 20 if primary_in_hero else 0
    else:
        p_hero_pts = 20
        primary_in_hero = True
    points += p_hero_pts
    breakdown["primary_kw_in_hero"] = {"score": p_hero_pts, "max": 20, "passed": primary_in_hero}

    # Secondary coverage
    if len(cleaned_targets) > 1:
        full_text = f"{desc.title} {desc.short_description} {desc.long_description} {desc.meta_description}"
        matched = sum(1 for kw in cleaned_targets[1:] if check_keyword_in_text(kw, full_text))
        sec_ratio = matched / (len(cleaned_targets) - 1)
        sec_pts = round(sec_ratio * 10)
    else:
        sec_pts = 10
    points += sec_pts
    breakdown["secondary_kw_coverage"] = {"score": sec_pts, "max": 10}

    # Bullets
    bullets_count = len(desc.bullet_points)
    bullets_pts = 10 if (3 <= bullets_count <= 5) else 0
    points += bullets_pts
    breakdown["bullet_points_structure"] = {"score": bullets_pts, "max": 10, "count": bullets_count}

    total_score = min(100, max(0, points))
    return total_score, breakdown


def validate_product_description(
    desc: ProductDescription,
    target_keywords: Optional[List[str]] = None,
    enforce_keywords: bool = True
) -> ValidationResult:
    """Validate SEO constraints, keyword presence, character counts, and readability.
    
    Checks:
    1. Title length <= 70 chars.
    2. Short description <= 160 chars.
    3. Meta description <= 160 chars.
    4. Bullet points between 3 and 5 items.
    5. Target keyword presence: verifies each keyword appears in the title,
       long description, or meta description using word-boundary matching.
    6. Keyword placement: verifies primary keyword in title and first 100 words.
    7. Computes Flesch Reading Ease score and 100-point SEO Scorecard.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Length checks
    if len(desc.title) > 70:
        errors.append(f"Title exceeds 70 characters (got {len(desc.title)}): '{desc.title}'")
    if len(desc.short_description) > 160:
        errors.append(f"Short description exceeds 160 characters (got {len(desc.short_description)})")
    if len(desc.meta_description) > 160:
        errors.append(f"Meta description exceeds 160 characters (got {len(desc.meta_description)})")

    # 2. Bullet count
    if not (3 <= len(desc.bullet_points) <= 5):
        errors.append(f"Bullet points count must be 3-5 (got {len(desc.bullet_points)})")

    # 3. Keyword verification
    found_keywords = []
    full_text = f"{desc.title} {desc.short_description} {desc.long_description} {desc.meta_description}"
    
    if target_keywords:
        cleaned_targets = [k.strip() for k in target_keywords if k.strip()]
        missing_keywords = []
        for kw in cleaned_targets:
            if check_keyword_in_text(kw, full_text):
                found_keywords.append(kw)
            else:
                missing_keywords.append(kw)
        
        # Check first 100 words of long description + title
        first_100_words = " ".join(desc.long_description.split()[:100])
        hero_text = f"{desc.title} {first_100_words}"
        hero_keyword_present = any(check_keyword_in_text(kw, hero_text) for kw in cleaned_targets)

        if not hero_keyword_present:
            warnings.append("No primary keyword found in title or first 100 words of long description.")

        if missing_keywords:
            msg = f"Missing required keyword(s): {', '.join(missing_keywords)}"
            if enforce_keywords:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Update description with verified keywords
        desc.keywords_used = list(dict.fromkeys(found_keywords))

    # 4. Readability and SEO scoring
    flesch_score, readability_level = calculate_flesch_reading_ease(desc.long_description)
    seo_score, seo_breakdown = calculate_seo_score(desc, target_keywords)

    is_valid = len(errors) == 0
    feedback = None
    if not is_valid:
        feedback = (
            "Your previous output failed the following validation checks:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\nPlease regenerate the JSON correcting these exact issues."
        )

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        retry_prompt_feedback=feedback,
        seo_score=seo_score,
        seo_breakdown=seo_breakdown,
        readability_score=flesch_score,
        readability_level=readability_level
    )
