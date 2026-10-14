"""Unit tests for src/generator.py with category aliasing and column normalization."""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.generator import (
    load_category_template,
    list_available_categories,
    resolve_category,
    normalize_product_dict,
    DescriptionGenerator
)
from src.schema import ProductDescription


def test_list_available_categories():
    cats = list_available_categories()
    assert "electronics" in cats
    assert "apparel" in cats
    assert "home_goods" in cats


def test_category_aliasing_and_resolution():
    # Direct templates
    assert resolve_category("electronics") == "electronics"
    assert resolve_category("apparel") == "apparel"
    assert resolve_category("home_goods") == "home_goods"

    # Common eCommerce aliases
    assert resolve_category("tech") == "electronics"
    assert resolve_category("audio") == "electronics"
    assert resolve_category("headphones") == "electronics"
    assert resolve_category("clothing") == "apparel"
    assert resolve_category("shoes") == "apparel"
    assert resolve_category("footwear") == "apparel"
    assert resolve_category("furniture") == "home_goods"
    assert resolve_category("kitchen") == "home_goods"

    # Compound token matching
    assert resolve_category("men's activewear") == "apparel"

    # Unknown category falls back gracefully without crashing
    assert resolve_category("completely_unknown_category_xyz") in ["home_goods", "electronics"]


def test_normalize_product_dict():
    raw_merchant_row = {
        "Product Title": "Mechanical Keyboard",
        "Item #": "KB-99",
        "Details": "Tactile blue switches with RGB backlighting",
        "Retail Price": "$129.50",
        "Vendor": "KeyForge",
        "Product Type": "tech",
        "Brand Voice": "technical",
        "SEO Keywords": "mechanical keyboard, gaming keyboard"
    }
    normalized = normalize_product_dict(raw_merchant_row)
    assert normalized["name"] == "Mechanical Keyboard"
    assert normalized["sku"] == "KB-99"
    assert normalized["specs"] == "Tactile blue switches with RGB backlighting"
    assert normalized["price"] == "$129.50"
    assert normalized["brand"] == "KeyForge"
    assert normalized["category"] == "tech"
    assert normalized["tone"] == "technical"
    assert normalized["keywords"] == "mechanical keyboard, gaming keyboard"


def test_load_category_template():
    data = load_category_template("electronics")
    assert data["category"] == "electronics"
    assert "required_variables" in data
    assert "sku" in data["required_variables"]
    assert "tone_options" in data
    assert "technical" in data["tone_options"]
    assert "premium" in data["tone_options"]
    assert "casual" in data["tone_options"]


def test_generator_single_mocked_success():
    valid_json = {
        "sku": "ELEC-MOCK",
        "title": "AuraSound Wireless Noise Cancelling Earbuds",
        "short_description": "Crystal clear sound with active noise cancellation and 40h battery.",
        "long_description": (
            "Experience studio audio with AuraSound wireless earbuds featuring active noise cancellation. "
            "Engineered with premium drivers and ergonomic comfort for all-day listening."
        ),
        "bullet_points": [
            "Up to 40 hours continuous playback",
            "State of the art active noise cancellation",
            "IPX5 water-resistant sweatproof casing"
        ],
        "meta_description": "Discover AuraSound wireless earbuds with active noise cancellation online.",
        "keywords_used": ["wireless earbuds", "active noise cancellation"],
        "tone": "premium",
        "category": "electronics"
    }

    gen = DescriptionGenerator(api_key="fake-key-for-test")
    with patch.object(gen, "_call_llm_with_backoff", return_value=json.dumps(valid_json)):
        product = {
            "sku": "ELEC-MOCK",
            "category": "electronics",
            "name": "Wireless Earbuds",
            "brand": "AuraSound",
            "specs": "40h battery, ANC",
            "price": "79.99",
            "tone": "premium",
            "keywords": "wireless earbuds, active noise cancellation"
        }
        desc, val_result, attempts = gen.generate_single(product)
        assert desc is not None
        assert desc.sku == "ELEC-MOCK"
        assert val_result.is_valid is True
        assert attempts == 1


def test_generator_retry_on_initial_failure():
    invalid_json = {
        "sku": "ELEC-RETRY",
        "title": "Basic Audio Earphones",
        "short_description": "Simple earphones for audio playback.",
        "long_description": "Generic earphones for listening to music without any target keywords here.",
        "bullet_points": [
            "Standard wired audio cable",
            "Comfortable silicone tips",
            "Lightweight carrying pouch"
        ],
        "meta_description": "Basic earphones available now.",
        "keywords_used": [],
        "tone": "casual",
        "category": "electronics"
    }

    valid_json = {
        "sku": "ELEC-RETRY",
        "title": "AuraSound Wireless Earbuds ANC",
        "short_description": "Premium wireless earbuds with active noise cancellation.",
        "long_description": (
            "Experience studio sound with our wireless earbuds featuring active noise cancellation. "
            "Engineered for all-day performance."
        ),
        "bullet_points": [
            "40 hours total battery life",
            "Active noise cancellation technology",
            "IPX5 sweatproof protection"
        ],
        "meta_description": "Get premium wireless earbuds with active noise cancellation today.",
        "keywords_used": ["wireless earbuds", "active noise cancellation"],
        "tone": "premium",
        "category": "electronics"
    }

    gen = DescriptionGenerator(api_key="fake-key-for-test", max_validation_retries=1)

    with patch.object(
        gen,
        "_call_llm_with_backoff",
        side_effect=[json.dumps(invalid_json), json.dumps(valid_json)]
    ):
        product = {
            "sku": "ELEC-RETRY",
            "category": "electronics",
            "name": "Wireless Earbuds",
            "specs": "40h battery, ANC",
            "tone": "premium",
            "keywords": "wireless earbuds, active noise cancellation"
        }
        desc, val_result, attempts = gen.generate_single(product)
        assert desc is not None
        assert attempts == 2
        assert val_result.is_valid is True


def test_generator_rejects_empty_product_without_api_call():
    """Products with no name and no specs must be rejected pre-flight (zero API calls)."""
    gen = DescriptionGenerator(api_key="fake-key-for-test")
    # Pass a product row where normalization produces "Unnamed Product" and empty specs
    empty_product = {"some_column": "some_value"}  # no recognized name/specs keys
    desc, val_result, attempts = gen.generate_single(empty_product)
    assert desc is None
    assert attempts == 0
    assert val_result.is_valid is False
    assert any("Insufficient product data" in e for e in val_result.errors)


def test_default_model_is_valid_gemini_name():
    """Confirm the default model name is a real Gemini model, not the old broken name."""
    gen = DescriptionGenerator(api_key="fake-key")
    assert gen.model_name != "gemini-3.6-flash", (
        "gemini-3.6-flash does not exist; default must be a real model"
    )
    assert gen.model_name.startswith("gemini-"), "Model name must start with 'gemini-'"


def test_generator_llm_property_raises_without_key():
    gen = DescriptionGenerator(api_key=None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        _ = gen.llm


def test_backoff_handles_429_and_recovers():
    gen = DescriptionGenerator(api_key="fake-key")
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "recovered successfully"
    mock_llm.invoke.side_effect = [Exception("HTTP 429: Resource exhausted / quota exceeded"), mock_response]
    gen._llm = mock_llm

    with patch("time.sleep") as mock_sleep:
        content = gen._call_llm_with_backoff("test-prompt")
        assert content == "recovered successfully"
        mock_sleep.assert_called_once()


def test_backoff_handles_503_and_recovers():
    gen = DescriptionGenerator(api_key="fake-key")
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "service ready"
    mock_llm.invoke.side_effect = [Exception("503 Service Unavailable"), mock_response]
    gen._llm = mock_llm

    with patch("time.sleep") as mock_sleep:
        content = gen._call_llm_with_backoff("test-prompt")
        assert content == "service ready"
        mock_sleep.assert_called_once()


def test_backoff_exhaustion_raises():
    gen = DescriptionGenerator(api_key="fake-key")
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Persistent API Error")
    gen._llm = mock_llm

    with patch("time.sleep"):
        with pytest.raises(Exception, match="Persistent API Error"):
            gen._call_llm_with_backoff("test-prompt", max_attempts=2)


def test_generate_batch_empty():
    gen = DescriptionGenerator(api_key="fake-key")
    res = gen.generate_batch([])
    assert res["total_processed"] == 0
    assert res["success_count"] == 0
    assert res["fail_count"] == 0
    assert res["records"] == []


def test_generate_batch_success_and_progress():
    gen = DescriptionGenerator(api_key="fake-key")
    sample_desc = ProductDescription(
        sku="BATCH-01",
        title="Batch Audio Earbuds",
        short_description="Short desc",
        long_description="Long description here for audio earbuds.",
        bullet_points=["B1", "B2", "B3"],
        meta_description="Meta desc",
        keywords_used=["audio earbuds"],
        tone="casual",
        category="electronics"
    )
    val_res = MagicMock(is_valid=True, seo_score=85, readability_score=70.0, readability_level="Plain English", seo_breakdown={}, errors=[], warnings=[])

    progress_calls = []
    def on_progress(cur, tot, rec):
        progress_calls.append((cur, tot, rec["sku"]))

    with patch.object(gen, "generate_single", return_value=(sample_desc, val_res, 1)):
        with patch("time.sleep") as mock_sleep:
            products = [{"sku": "BATCH-01", "name": "Item 1", "category": "electronics", "specs": "Good specs"}]
            result = gen.generate_batch(products, delay_seconds=0.5, progress_callback=on_progress)
            assert result["success_count"] == 1
            assert len(result["successful_descriptions"]) == 1
            assert len(progress_calls) == 1
            assert progress_calls[0] == (1, 1, "BATCH-01")


def test_generate_batch_crash_handled():
    gen = DescriptionGenerator(api_key="fake-key")
    with patch.object(gen, "generate_single", side_effect=RuntimeError("Unexpected pipeline crash")):
        products = [{"sku": "CRASH-01", "name": "Crashing Product", "category": "electronics", "specs": "bad specs"}]
        result = gen.generate_batch(products, delay_seconds=0)
        assert result["total_processed"] == 1
        assert result["success_count"] == 0
        assert result["fail_count"] == 1
        assert result["records"][0]["status"] == "CRASH_SKIPPED"
        assert "Unexpected pipeline crash" in result["records"][0]["errors"][0]


def test_load_category_template_missing_file(tmp_path):
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        load_category_template("nonexistent_cat", templates_dir=tmp_path)


def test_list_available_categories_nonexistent(tmp_path):
    empty_dir = tmp_path / "does_not_exist"
    assert list_available_categories(empty_dir) == []

