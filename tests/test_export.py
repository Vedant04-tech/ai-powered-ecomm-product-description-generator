"""Unit tests for Shopify and WooCommerce export mappers."""

import csv
import io
import pytest
from src.schema import ProductDescription
from src.export.shopify import to_shopify_csv_row, to_shopify_csv, generate_handle, format_body_html, format_currency
from src.export.woocommerce import to_woocommerce_payload


@pytest.fixture
def sample_desc():
    return ProductDescription(
        sku="TEST-SKU-1",
        title="Eco-Friendly Cotton T-Shirt",
        short_description="Soft organic cotton t-shirt with modern casual fit.",
        long_description=(
            "Crafted from 100% GOTS-certified organic ring-spun cotton. "
            "Features durable double-needle stitching and an ultra-breathable weave."
        ),
        bullet_points=[
            "100% organic ring-spun cotton",
            "Pre-shrunk fabric for lasting fit",
            "Non-toxic vegetable-based dyes"
        ],
        meta_description="Shop our eco-friendly organic cotton t-shirt for sustainable comfort.",
        keywords_used=["organic cotton t-shirt", "sustainable apparel"],
        tone="casual",
        category="apparel"
    )


def test_format_currency():
    assert format_currency("29.99") == "29.99"
    assert format_currency("$29.9") == "29.90"
    assert format_currency("1,299.5") == "1299.50"
    assert format_currency(48) == "48.00"
    assert format_currency("") == ""
    assert format_currency(None) == ""


def test_generate_handle():
    assert generate_handle("Eco-Friendly Cotton T-Shirt") == "eco-friendly-cotton-t-shirt"
    # With collision avoidance suffix
    handle_with_sku = generate_handle("Eco-Friendly Cotton T-Shirt", sku="SKU-101", unique_suffix=True)
    assert handle_with_sku == "eco-friendly-cotton-t-shirt-sku-101"


def test_shopify_csv_row_mapping(sample_desc):
    extra = {"brand": "GreenWear", "price": "29.9"}
    row = to_shopify_csv_row(sample_desc, extra)

    assert row["Handle"] == "eco-friendly-cotton-t-shirt"
    assert row["Title"] == "Eco-Friendly Cotton T-Shirt"
    assert row["Vendor"] == "GreenWear"
    assert row["Variant SKU"] == "TEST-SKU-1"
    assert row["Variant Price"] == "29.90"  # Formatted currency
    assert row["SEO Title"] == sample_desc.title
    assert row["SEO Description"] == sample_desc.meta_description
    assert "<li>100% organic ring-spun cotton</li>" in row["Body (HTML)"]
    assert "<p>" in row["Body (HTML)"]


def test_shopify_csv_generation(sample_desc):
    csv_str = to_shopify_csv([sample_desc], [{"brand": "BrandX", "price": "19.99"}])
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["Variant SKU"] == "TEST-SKU-1"
    assert rows[0]["Title"] == "Eco-Friendly Cotton T-Shirt"
    assert rows[0]["Variant Price"] == "19.99"


def test_woocommerce_payload_mapping(sample_desc):
    extra = {"brand": "GreenWear", "price": "$29.9"}
    payload = to_woocommerce_payload(sample_desc, extra)

    assert payload["name"] == sample_desc.title
    assert payload["sku"] == "TEST-SKU-1"
    assert payload["regular_price"] == "29.90"
    assert payload["type"] == "simple"
    assert payload["categories"] == [{"name": "Apparel"}]
    assert payload["manage_stock"] is True
    assert payload["stock_status"] == "instock"
    assert payload["stock_quantity"] == 100
    assert any(tag["name"] == "organic cotton t-shirt" for tag in payload["tags"])

    # Check SEO meta_data
    meta_dict = {m["key"]: m["value"] for m in payload["meta_data"]}
    assert meta_dict["_yoast_wpseo_title"] == sample_desc.title
    assert meta_dict["_yoast_wpseo_metadesc"] == sample_desc.meta_description
    assert meta_dict["rank_math_title"] == sample_desc.title


def test_shopify_csv_empty_list():
    assert to_shopify_csv([]) == ""


def test_woocommerce_payload_json_helper(sample_desc):
    import json
    from src.export.woocommerce import to_woocommerce_payload_json
    json_str = to_woocommerce_payload_json([sample_desc], [{"price": "49.00"}])
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["regular_price"] == "49.00"


def test_format_currency_malformed():
    assert format_currency("Contact for price") == "Contact for price"
    assert format_currency("  ") == ""


def test_generate_handle_special_symbols():
    handle = generate_handle("!@# Special $$$ Product ???", sku="XYZ-9")
    assert "special-product" in handle

