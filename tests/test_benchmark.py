"""Unit tests for benchmark/effort_benchmark.py with financial ROI."""

from benchmark.effort_benchmark import calculate_effort_reduction, BenchmarkResult


def test_effort_reduction_calculation():
    # 100 SKUs:
    # Manual: 100 * 12 mins = 1200 mins = 20.0 hours
    # Pipeline: 120s = 0.033 hrs + 100 * 1 min review = 1.667 hrs review => ~1.7 hours total
    # Net saved: ~18.3 hours => ~91.5% reduction
    res = calculate_effort_reduction(
        sku_count=100,
        pipeline_wall_clock_seconds=120.0,
        manual_minutes_per_sku=12.0,
        review_minutes_per_sku=1.0,
        hourly_labor_rate=30.0,
        api_cost_per_sku=0.002
    )
    assert res.sku_count == 100
    assert res.manual_total_hours == 20.0
    assert res.effort_reduction_pct > 70.0
    assert res.hours_saved > 15.0

    # Financial ROI tests
    assert res.manual_cost_usd == 600.0  # 20 hours * $30/hr
    assert res.ai_token_cost_usd == 0.20  # 100 * $0.002
    assert res.review_labor_cost_usd == 50.0  # 1.67 hours * $30/hr
    assert res.dollars_saved > 500.0
    assert res.cost_reduction_pct > 90.0


def test_zero_sku_handling():
    res = calculate_effort_reduction(0, 0.0)
    assert res.sku_count == 0
    assert res.effort_reduction_pct == 0.0
    assert res.manual_cost_usd == 0.0
    assert res.dollars_saved == 0.0


def test_benchmark_result_to_dict_and_summary_table():
    res = calculate_effort_reduction(50, 60.0)
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["sku_count"] == 50
    assert "dollars_saved" in d

    table = res.summary_table()
    assert "EFFORT & ROI BENCHMARK" in table
    assert "Products Processed (SKUs)      : 50" in table
    assert "Net Financial Savings" in table

