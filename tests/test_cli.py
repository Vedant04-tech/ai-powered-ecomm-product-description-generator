"""Unit and integration tests for src/cli.py."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from typer.testing import CliRunner

from src.cli import app, read_products_from_file
from src.schema import ProductDescription

runner = CliRunner()


@pytest.fixture
def sample_desc():
    return ProductDescription(
        sku="CLI-001",
        title="Test Ergonomic Wireless Keyboard for Office",
        short_description="Ergonomic wireless keyboard with low latency and tactile typing.",
        long_description=(
            "Upgrade your workspace setup with this wireless keyboard featuring ergonomic "
            "wrist support, whisper-quiet switches, and long-lasting rechargeable battery life. "
            "Engineered for daily business productivity."
        ),
        bullet_points=[
            "Wireless 2.4GHz and Bluetooth dual-mode",
            "Cushioned memory foam wrist rest",
            "Up to 3 months battery life per charge"
        ],
        meta_description="Shop our ergonomic wireless keyboard with Bluetooth and rechargeable battery.",
        keywords_used=["wireless keyboard", "ergonomic"],
        tone="professional",
        category="electronics"
    )


def test_read_products_csv(tmp_path: Path):
    csv_file = tmp_path / "products.csv"
    csv_file.write_text(
        "sku,name,category,specs\n"
        "SKU-1,Product One,electronics,spec1\n"
        "SKU-2,Product Two,apparel,spec2\n",
        encoding="utf-8"
    )
    rows = read_products_from_file(csv_file)
    assert len(rows) == 2
    assert rows[0]["sku"] == "SKU-1"
    assert rows[1]["category"] == "apparel"


def test_read_products_json_list(tmp_path: Path):
    json_file = tmp_path / "products.json"
    data = [{"sku": "SKU-1", "name": "Item 1"}, {"sku": "SKU-2", "name": "Item 2"}]
    json_file.write_text(json.dumps(data), encoding="utf-8")
    rows = read_products_from_file(json_file)
    assert len(rows) == 2
    assert rows[0]["sku"] == "SKU-1"


def test_read_products_json_single_dict(tmp_path: Path):
    json_file = tmp_path / "single.json"
    data = {"sku": "SKU-SINGLE", "name": "Single Item"}
    json_file.write_text(json.dumps(data), encoding="utf-8")
    rows = read_products_from_file(json_file)
    assert len(rows) == 1
    assert rows[0]["sku"] == "SKU-SINGLE"


def test_read_products_json_invalid_primitive(tmp_path: Path):
    json_file = tmp_path / "invalid.json"
    json_file.write_text("12345", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON file must contain an array"):
        read_products_from_file(json_file)


def test_read_products_unsupported_format(tmp_path: Path):
    txt_file = tmp_path / "products.txt"
    txt_file.write_text("sku,name\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_products_from_file(txt_file)


def test_cli_generate_missing_file():
    result = runner.invoke(app, ["--input", "nonexistent_file_path_12345.csv"])
    assert result.exit_code != 0
    assert "Input file not found" in result.stdout


def test_cli_generate_batch_success(tmp_path: Path, sample_desc):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "sku,name,category,specs,price,brand\n"
        "CLI-001,Test Keyboard,electronics,Bluetooth quiet switches,$79.99,LogiTech\n",
        encoding="utf-8"
    )
    out_json = tmp_path / "output.json"
    shopify_csv = tmp_path / "shopify.csv"
    woo_json = tmp_path / "woocommerce.json"

    mock_result = {
        "total_processed": 1,
        "success_count": 1,
        "fail_count": 0,
        "wall_clock_seconds": 0.45,
        "records": [
            {
                "sku": "CLI-001",
                "name": "Test Keyboard",
                "category": "electronics",
                "status": "PASS",
                "attempts": 1,
                "seo_score": 90,
                "readability_score": 68.0,
                "readability_level": "Standard / Plain English",
                "seo_breakdown": {},
                "errors": [],
                "warnings": [],
                "description": sample_desc.model_dump(),
                "extra_attrs": {"sku": "CLI-001", "name": "Test Keyboard", "brand": "LogiTech", "price": "$79.99"}
            }
        ],
        "successful_descriptions": [sample_desc]
    }

    with patch("src.cli.DescriptionGenerator") as MockGenClass:
        mock_instance = MagicMock()
        mock_instance.generate_batch.return_value = mock_result
        MockGenClass.return_value = mock_instance

        res = runner.invoke(app, [
            "--input", str(input_csv),
            "--out", str(out_json),
            "--export-shopify", str(shopify_csv),
            "--export-woo", str(woo_json),
            "--tone", "premium",
            "--model", "gemini-3.8-flash",
            "--delay", "0.0"
        ])

        assert res.exit_code == 0
        assert "Generation Results Summary" in res.stdout
        assert "CLI-001" in res.stdout
        assert "PASS" in res.stdout

        # Verify output files exist and contain valid contents
        assert out_json.exists()
        saved_data = json.loads(out_json.read_text(encoding="utf-8"))
        assert saved_data["total_processed"] == 1
        assert saved_data["successful_descriptions"][0]["sku"] == "CLI-001"

        assert shopify_csv.exists()
        shopify_content = shopify_csv.read_text(encoding="utf-8")
        assert "Handle" in shopify_content
        assert "CLI-001" in shopify_content

        assert woo_json.exists()
        woo_content = json.loads(woo_json.read_text(encoding="utf-8"))
        assert len(woo_content) == 1
        assert woo_content[0]["sku"] == "CLI-001"


def test_cli_generate_with_failed_record(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("sku,name,category,specs\nFAIL-001,Broken Item,electronics,nothing\n", encoding="utf-8")

    mock_result = {
        "total_processed": 1,
        "success_count": 0,
        "fail_count": 1,
        "wall_clock_seconds": 0.2,
        "records": [
            {
                "sku": "FAIL-001",
                "name": "Broken Item",
                "category": "electronics",
                "status": "RETRY_FAILED",
                "attempts": 2,
                "seo_score": 0,
                "readability_score": 0.0,
                "readability_level": "N/A",
                "seo_breakdown": {},
                "errors": ["Failed to satisfy constraints"],
                "warnings": [],
                "description": None,
                "extra_attrs": {"sku": "FAIL-001"}
            }
        ],
        "successful_descriptions": []
    }

    with patch("src.cli.DescriptionGenerator") as MockGenClass:
        mock_instance = MagicMock()
        mock_instance.generate_batch.return_value = mock_result
        MockGenClass.return_value = mock_instance

        res = runner.invoke(app, ["--input", str(input_csv)])
        assert res.exit_code == 0
        assert "FAIL-001" in res.stdout
        assert "Complete: 0/1 succeeded" in res.stdout
        assert "RETRY_" in res.stdout


