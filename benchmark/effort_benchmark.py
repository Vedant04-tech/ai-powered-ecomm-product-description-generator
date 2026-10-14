"""Empirical effort and financial ROI benchmark comparing manual copywriting vs pipeline generation."""

from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

DEFAULT_MANUAL_MINUTES_PER_SKU = 12.0   # Industry baseline: 7 min drafting + 5 min SEO/meta/formatting
DEFAULT_REVIEW_MINUTES_PER_SKU = 1.0    # Human-in-the-loop review time per AI generation
DEFAULT_HOURLY_LABOR_RATE_USD = 30.0    # Standard freelance copywriter hourly rate
DEFAULT_API_COST_PER_SKU_USD = 0.002    # Token inference cost on Gemini 3.6 Flash per SKU


@dataclass
class BenchmarkResult:
    """Quantitative effort and financial ROI reduction metrics."""
    sku_count: int
    manual_minutes_per_sku: float
    review_minutes_per_sku: float
    manual_total_hours: float
    pipeline_wall_clock_seconds: float
    human_review_hours: float
    pipeline_total_hours: float
    hours_saved: float
    effort_reduction_pct: float
    # Financial metrics
    hourly_labor_rate_usd: float = DEFAULT_HOURLY_LABOR_RATE_USD
    api_cost_per_sku_usd: float = DEFAULT_API_COST_PER_SKU_USD
    manual_cost_usd: float = 0.0
    ai_token_cost_usd: float = 0.0
    review_labor_cost_usd: float = 0.0
    pipeline_total_cost_usd: float = 0.0
    dollars_saved: float = 0.0
    cost_reduction_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_table(self) -> str:
        return (
            f"============================================================\n"
            f"                EFFORT & ROI BENCHMARK                      \n"
            f"============================================================\n"
            f"Products Processed (SKUs)      : {self.sku_count}\n"
            f"Manual Baseline (per SKU)      : {self.manual_minutes_per_sku:.1f} mins\n"
            f"Estimated Manual Effort        : {self.manual_total_hours:.2f} hours\n"
            f"Pipeline Wall-Clock Run Time   : {self.pipeline_wall_clock_seconds:.1f} seconds\n"
            f"Human Review Time (per SKU)    : {self.review_minutes_per_sku:.1f} mins\n"
            f"Total Pipeline + Review Effort : {self.pipeline_total_hours:.2f} hours\n"
            f"Net Hours Saved                : {self.hours_saved:.2f} hours\n"
            f"------------------------------------------------------------\n"
            f"EFFORT REDUCTION               : {self.effort_reduction_pct:.1f}%\n"
            f"============================================================\n"
            f"FINANCIAL ROI (at ${self.hourly_labor_rate_usd:.2f}/hr labor rate)\n"
            f"------------------------------------------------------------\n"
            f"Estimated Manual Labor Cost    : ${self.manual_cost_usd:.2f}\n"
            f"AI Pipeline + Review Cost      : ${self.pipeline_total_cost_usd:.2f} (API: ${self.ai_token_cost_usd:.2f})\n"
            f"Net Financial Savings          : ${self.dollars_saved:.2f}\n"
            f"COST REDUCTION                 : {self.cost_reduction_pct:.1f}%\n"
            f"============================================================"
        )


def calculate_effort_reduction(
    sku_count: int,
    pipeline_wall_clock_seconds: float,
    manual_minutes_per_sku: float = DEFAULT_MANUAL_MINUTES_PER_SKU,
    review_minutes_per_sku: float = DEFAULT_REVIEW_MINUTES_PER_SKU,
    hourly_labor_rate: float = DEFAULT_HOURLY_LABOR_RATE_USD,
    api_cost_per_sku: float = DEFAULT_API_COST_PER_SKU_USD
) -> BenchmarkResult:
    """Compute empirical effort reduction and dollar ROI comparing manual baseline vs AI pipeline."""
    if sku_count <= 0:
        return BenchmarkResult(
            sku_count=0,
            manual_minutes_per_sku=manual_minutes_per_sku,
            review_minutes_per_sku=review_minutes_per_sku,
            manual_total_hours=0.0,
            pipeline_wall_clock_seconds=0.0,
            human_review_hours=0.0,
            pipeline_total_hours=0.0,
            hours_saved=0.0,
            effort_reduction_pct=0.0
        )

    # Time calculations
    manual_total_hours = (sku_count * manual_minutes_per_sku) / 60.0
    human_review_hours = (sku_count * review_minutes_per_sku) / 60.0
    pipeline_hours = pipeline_wall_clock_seconds / 3600.0
    total_pipeline_hours = pipeline_hours + human_review_hours

    hours_saved = max(0.0, manual_total_hours - total_pipeline_hours)
    reduction_pct = (hours_saved / manual_total_hours) * 100.0 if manual_total_hours > 0 else 0.0

    # Financial ROI calculations
    manual_cost = manual_total_hours * hourly_labor_rate
    ai_token_cost = sku_count * api_cost_per_sku
    review_labor_cost = human_review_hours * hourly_labor_rate
    total_pipeline_cost = ai_token_cost + review_labor_cost

    dollars_saved = max(0.0, manual_cost - total_pipeline_cost)
    cost_reduction = (dollars_saved / manual_cost) * 100.0 if manual_cost > 0 else 0.0

    return BenchmarkResult(
        sku_count=sku_count,
        manual_minutes_per_sku=manual_minutes_per_sku,
        review_minutes_per_sku=review_minutes_per_sku,
        manual_total_hours=round(manual_total_hours, 2),
        pipeline_wall_clock_seconds=round(pipeline_wall_clock_seconds, 2),
        human_review_hours=round(human_review_hours, 2),
        pipeline_total_hours=round(total_pipeline_hours, 2),
        hours_saved=round(hours_saved, 2),
        effort_reduction_pct=round(reduction_pct, 1),
        hourly_labor_rate_usd=round(hourly_labor_rate, 2),
        api_cost_per_sku_usd=round(api_cost_per_sku, 4),
        manual_cost_usd=round(manual_cost, 2),
        ai_token_cost_usd=round(ai_token_cost, 2),
        review_labor_cost_usd=round(review_labor_cost, 2),
        pipeline_total_cost_usd=round(total_pipeline_cost, 2),
        dollars_saved=round(dollars_saved, 2),
        cost_reduction_pct=round(cost_reduction, 1)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run effort and financial ROI benchmark")
    parser.add_argument("--skus", type=int, default=100, help="Number of SKUs to simulate")
    parser.add_argument("--wallclock", type=float, default=120.0, help="Wall-clock seconds for batch run")
    parser.add_argument("--manual-min", type=float, default=DEFAULT_MANUAL_MINUTES_PER_SKU, help="Manual writing mins per SKU")
    parser.add_argument("--review-min", type=float, default=DEFAULT_REVIEW_MINUTES_PER_SKU, help="Review mins per SKU")
    parser.add_argument("--hourly-rate", type=float, default=DEFAULT_HOURLY_LABOR_RATE_USD, help="Copywriter hourly rate ($)")
    parser.add_argument("--api-cost", type=float, default=DEFAULT_API_COST_PER_SKU_USD, help="API token cost per SKU ($)")
    args = parser.parse_args()

    res = calculate_effort_reduction(
        sku_count=args.skus,
        pipeline_wall_clock_seconds=args.wallclock,
        manual_minutes_per_sku=args.manual_min,
        review_minutes_per_sku=args.review_min,
        hourly_labor_rate=args.hourly_rate,
        api_cost_per_sku=args.api_cost
    )
    print(res.summary_table())
