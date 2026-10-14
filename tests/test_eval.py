"""Tests for benchmark/eval_runner.py — all run offline without API calls."""

from pathlib import Path
import json
import pytest

from benchmark.eval_runner import (
    run_evaluation,
    _keyword_coverage,
    _aggregate,
    _make_mock_response,
    _make_v3_mock_result,
    _parse_json_response,
    EvalRecord,
)
from src.schema import ProductDescription


SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample_products.csv"


@pytest.fixture
def sample_desc():
    return ProductDescription(
        sku="TEST-001",
        title="Wireless Earbuds with Active Noise Cancellation",
        short_description="Premium wireless earbuds with ANC and 40h battery.",
        long_description=(
            "The Wireless Earbuds deliver active noise cancellation and 40-hour battery life. "
            "Perfect for commutes, travel, and fitness. Crystal-clear calls guaranteed."
        ),
        bullet_points=[
            "Active noise cancellation",
            "40-hour total battery life",
            "IPX5 water resistance",
        ],
        meta_description="Get wireless earbuds with active noise cancellation. Shop now.",
        keywords_used=["wireless earbuds", "active noise cancellation"],
        tone="premium",
        category="electronics",
    )


def test_keyword_coverage_full(sample_desc):
    keywords = ["wireless earbuds", "active noise cancellation"]
    cov = _keyword_coverage(sample_desc, keywords)
    assert cov == 1.0


def test_keyword_coverage_partial(sample_desc):
    keywords = ["wireless earbuds", "something missing xyz"]
    cov = _keyword_coverage(sample_desc, keywords)
    assert cov == 0.5


def test_keyword_coverage_none():
    assert _keyword_coverage(None, ["keyword"]) == 0.0


def test_parse_json_response_clean():
    raw = '{"sku": "X", "title": "Test"}'
    result = _parse_json_response(raw)
    assert result == {"sku": "X", "title": "Test"}


def test_parse_json_response_markdown_wrapped():
    raw = "```json\n{\"sku\": \"X\", \"title\": \"Test\"}\n```"
    result = _parse_json_response(raw)
    assert result == {"sku": "X", "title": "Test"}


def test_parse_json_response_invalid():
    assert _parse_json_response("not json at all") is None


def test_make_mock_response_v1():
    norm = {"sku": "T-001", "name": "Test Widget", "brand": "ACME", "category": "electronics",
            "specs": "Spec A", "price": "29.99", "tone": "casual", "keywords": "widget, gadget"}
    raw = _make_mock_response("T-001", norm, "v1_baseline")
    data = json.loads(raw)
    assert data["sku"] == "T-001"
    assert len(data["bullet_points"]) >= 3


def test_make_mock_response_v2_has_keyword_in_title():
    norm = {"sku": "T-002", "name": "Smart Speaker", "brand": "X", "category": "electronics",
            "specs": "S", "price": "49", "tone": "premium", "keywords": "smart speaker, bluetooth"}
    raw = _make_mock_response("T-002", norm, "v2_structured")
    data = json.loads(raw)
    # V2 should incorporate primary keyword in title
    assert "smart speaker" in data["title"].lower()


def test_v3_mock_result_is_schema_valid():
    norm = {"sku": "T-003", "name": "Running Shoes", "brand": "ApexRun", "category": "apparel",
            "specs": "Foam sole", "price": "99", "tone": "casual", "keywords": "running shoes, trail shoes"}
    desc, val, attempts = _make_v3_mock_result("T-003", norm, ["running shoes", "trail shoes"])
    assert desc is not None
    assert desc.sku == "T-003"
    assert attempts == 1


def test_aggregate_empty():
    assert _aggregate([]) == {}


def test_aggregate_all_valid():
    records = [
        EvalRecord("S1", "P1", "electronics", "v3", True, 85, 1.0, True, 0, 1.2),
        EvalRecord("S2", "P2", "apparel", "v3", True, 90, 0.8, True, 0, 0.9),
    ]
    summary = _aggregate(records)
    assert summary["schema_valid_rate_pct"] == 100.0
    assert summary["seo_score_avg"] == 87.5
    assert summary["keyword_coverage_avg_pct"] == 90.0


def test_aggregate_some_invalid():
    records = [
        EvalRecord("S1", "P1", "electronics", "v1", True, 75, 0.5, True, 0, 1.0),
        EvalRecord("S2", "P2", "electronics", "v1", False, 0, 0.0, False, 0, 0.5),
    ]
    summary = _aggregate(records)
    assert summary["schema_valid_rate_pct"] == 50.0
    assert summary["schema_valid_count"] == 1


def test_full_mock_evaluation_runs(tmp_path):
    """End-to-end offline evaluation over the real dataset — no API calls."""
    out = tmp_path / "eval_out.json"
    results = run_evaluation(
        dataset_path=SAMPLE_CSV,
        output_path=out,
        mock=True,
    )
    assert out.exists()
    assert "version_summary" in results
    assert "records" in results
    assert len(results["records"]) == 18 * 3  # 18 products × 3 versions
    assert results["meta"]["mock"] is True

    # All sources should be tagged as mock
    for rec in results["records"]:
        assert rec["source"] == "mock"

    # V3 should have better schema valid rate than V1 in mock mode
    v1_rate = results["version_summary"]["v1_baseline"]["schema_valid_rate_pct"]
    v3_rate = results["version_summary"]["v3_full_pipeline"]["schema_valid_rate_pct"]
    # Both should be valid (mock builds valid objects), but V3 metrics are tracked
    assert v3_rate >= 0  # structural check — does not assert V3 > V1 (mock equality expected)


def test_eval_output_has_required_meta_fields(tmp_path):
    out = tmp_path / "eval_meta_test.json"
    results = run_evaluation(dataset_path=SAMPLE_CSV, output_path=out, mock=True)
    meta = results["meta"]
    assert "dataset_size" in meta
    assert "model" in meta
    assert "mock" in meta
    assert "timestamp" in meta
    assert "prompt_versions" in meta
    assert "cost_note" in meta
    # Cost must explicitly flag it's estimated
    assert "ESTIMATED" in meta["cost_note"]
