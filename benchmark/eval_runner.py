"""Prompt-version evaluation runner for the Product Description Generator.

Compares three prompt strategies on a shared evaluation dataset:
  V1 — Baseline:   bare product facts, no structure instructions.
  V2 — Structured: explicit output schema, tone, and SEO requirements added.
  V3 — Full pipeline (default): category YAML template + deterministic validation + repair.

Usage (offline/mocked — no API key required):
    python -m benchmark.eval_runner --mock

Usage (live — requires GEMINI_API_KEY):
    python -m benchmark.eval_runner --dataset data/sample_products.csv --out output/eval_results.json

All results are saved as structured JSON so repeated runs can be compared.
Mocked results are clearly tagged "source": "mock" in the output.

Note on cost estimates:
  Token counts are not available from the langchain-google-genai wrapper without
  a separate billing API call. Where marked ESTIMATED, the figure uses a rough
  1 000 tokens/request assumption at the current Gemini 2.5 Flash pricing page
  ($0.075 / 1M input, $0.30 / 1M output as of 2025-10). These are illustrative
  only and should not be cited as measured costs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.generator import DescriptionGenerator, normalize_product_dict
from src.schema import ProductDescription
from src.validate import validate_product_description, ValidationResult

# ---------------------------------------------------------------------------
# Prompt-version definitions
# ---------------------------------------------------------------------------

PROMPT_VERSIONS: Dict[str, str] = {
    "v1_baseline": (
        "Baseline prompt — bare facts only. "
        "No schema, tone, SEO, or structure instructions given to the model."
    ),
    "v2_structured": (
        "Structured prompt — explicit JSON schema fields, tone, "
        "SEO keyword rules, and character limits. No category-specific context."
    ),
    "v3_full_pipeline": (
        "Full pipeline — category YAML template, tone guidance, keyword injection, "
        "Pydantic validation, and bounded repair retries. This is the production path."
    ),
}

# V1 system prompt (simple)
_V1_SYSTEM = "You are a helpful eCommerce copywriter."

# V1 user template — bare facts, no schema instructions
_V1_USER_TEMPLATE = """\
Write a product description for the following product.

Product Name: {name}
Brand: {brand}
Category: {category}
Specifications: {specs}
Price: {price}

Respond with a valid JSON object with these exact keys:
sku, title, short_description, long_description, bullet_points (list),
meta_description, keywords_used (list), tone, category.
"""

# V2 system prompt (structured)
_V2_SYSTEM = (
    "You are an expert eCommerce SEO copywriter. "
    "Always respond with a single valid JSON object matching the schema provided."
)

# V2 user template — explicit constraints added
_V2_USER_TEMPLATE = """\
Generate a product listing for the following product.

[PRODUCT FACTS]
- SKU: {sku}
- Name: {name}
- Brand: {brand}
- Category: {category}
- Specifications: {specs}
- Price: {price}
- Target Keywords: {keywords}
- Tone: {tone}

[CONSTRAINTS]
- title: SEO-friendly, max 70 characters, include primary keyword
- short_description: max 160 characters, compelling hook
- long_description: 2-3 paragraphs, include all keywords naturally
- bullet_points: exactly 3 to 5 items
- meta_description: max 160 characters, include keyword, end with call-to-action
- keywords_used: list the keywords you successfully used
- tone: use the requested tone throughout
- category: copy the category value exactly

Respond with raw JSON only — no markdown fences.
"""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    sku: str
    name: str
    category: str
    prompt_version: str
    schema_valid: bool
    seo_score: int
    keyword_coverage_pct: float  # % of target keywords found in output
    initial_valid: bool          # valid on first attempt (V3 only; True for V1/V2 single-shot)
    repair_attempts: int         # 0 for V1/V2; 1+ means a repair loop ran (V3 only)
    latency_seconds: float
    errors: List[str] = field(default_factory=list)
    source: str = "live"         # "live" | "mock"


def _keyword_coverage(desc: Optional[ProductDescription], keywords: List[str]) -> float:
    """Return fraction of target keywords found anywhere in the output (0.0–1.0)."""
    if not desc or not keywords:
        return 0.0
    full_text = (
        f"{desc.title} {desc.short_description} {desc.long_description} {desc.meta_description}"
    ).lower()
    matched = sum(1 for kw in keywords if kw.lower() in full_text)
    return round(matched / len(keywords), 3)


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from a potentially markdown-wrapped response."""
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Per-version evaluators
# ---------------------------------------------------------------------------

def _eval_v1_or_v2(
    product: Dict[str, Any],
    system_prompt: str,
    user_template: str,
    version: str,
    generator: DescriptionGenerator,
    mock: bool,
) -> EvalRecord:
    """Run V1 or V2 single-shot generation (no repair loop)."""
    norm = normalize_product_dict(product)
    sku = norm["sku"]
    keywords = [k.strip() for k in str(norm["keywords"]).split(",") if k.strip()]
    user_msg = user_template.format(
        sku=sku,
        name=norm["name"],
        brand=norm["brand"] or "N/A",
        category=norm["category"],
        specs=norm["specs"] or "N/A",
        price=norm["price"] or "N/A",
        keywords=", ".join(keywords) or "N/A",
        tone=norm["tone"] or "professional",
    )
    messages = [("system", system_prompt), ("human", user_msg)]

    t0 = time.time()
    try:
        if mock:
            raw = _make_mock_response(sku, norm, version)
        else:
            raw = generator._call_llm_with_backoff(messages)
        parsed_dict = _parse_json_response(raw)
        latency = round(time.time() - t0, 2)

        if parsed_dict is None:
            return EvalRecord(
                sku=sku, name=norm["name"], category=norm["category"],
                prompt_version=version, schema_valid=False,
                seo_score=0, keyword_coverage_pct=0.0,
                initial_valid=False, repair_attempts=0,
                latency_seconds=latency,
                errors=["JSON parse failure"], source="mock" if mock else "live"
            )

        parsed_dict.setdefault("sku", sku)
        parsed_dict.setdefault("tone", norm["tone"] or "professional")
        parsed_dict.setdefault("category", norm["category"])
        parsed_dict.setdefault("keywords_used", [])

        try:
            desc = ProductDescription(**parsed_dict)
            val = validate_product_description(desc, target_keywords=keywords)
            return EvalRecord(
                sku=sku, name=norm["name"], category=norm["category"],
                prompt_version=version, schema_valid=True,
                seo_score=val.seo_score,
                keyword_coverage_pct=_keyword_coverage(desc, keywords),
                initial_valid=val.is_valid, repair_attempts=0,
                latency_seconds=latency, errors=val.errors,
                source="mock" if mock else "live"
            )
        except Exception as e:
            return EvalRecord(
                sku=sku, name=norm["name"], category=norm["category"],
                prompt_version=version, schema_valid=False,
                seo_score=0, keyword_coverage_pct=0.0,
                initial_valid=False, repair_attempts=0,
                latency_seconds=latency, errors=[str(e)],
                source="mock" if mock else "live"
            )
    except Exception as e:
        return EvalRecord(
            sku=sku, name=norm["name"], category=norm["category"],
            prompt_version=version, schema_valid=False,
            seo_score=0, keyword_coverage_pct=0.0,
            initial_valid=False, repair_attempts=0,
            latency_seconds=round(time.time() - t0, 2),
            errors=[str(e)], source="mock" if mock else "live"
        )


def _eval_v3(
    product: Dict[str, Any],
    generator: DescriptionGenerator,
    mock: bool,
) -> EvalRecord:
    """Run V3 — full pipeline with category templates and repair retries."""
    norm = normalize_product_dict(product)
    sku = norm["sku"]
    keywords = [k.strip() for k in str(norm["keywords"]).split(",") if k.strip()]

    t0 = time.time()
    if mock:
        # Return a representative mocked success to avoid touching the LLM
        desc, val, attempts = _make_v3_mock_result(sku, norm, keywords)
    else:
        desc, val, attempts = generator.generate_single(product)

    latency = round(time.time() - t0, 2)
    return EvalRecord(
        sku=sku, name=norm["name"], category=norm["category"],
        prompt_version="v3_full_pipeline",
        schema_valid=desc is not None,
        seo_score=val.seo_score,
        keyword_coverage_pct=_keyword_coverage(desc, keywords),
        initial_valid=(desc is not None and attempts == 1 and val.is_valid),
        repair_attempts=max(0, attempts - 1),
        latency_seconds=latency,
        errors=val.errors,
        source="mock" if mock else "live"
    )


# ---------------------------------------------------------------------------
# Mock helpers (offline evaluation without API calls)
# ---------------------------------------------------------------------------

def _make_mock_response(sku: str, norm: Dict[str, Any], version: str) -> str:
    """Return a plausible-but-synthetic JSON response for offline testing.

    V1 mock intentionally omits some SEO requirements to show baseline weakness.
    V2 mock includes structure but may miss keyword placement.
    """
    name = norm["name"]
    category = norm["category"]
    tone = norm["tone"] or "professional"
    kw_str = str(norm["keywords"])
    kws = [k.strip() for k in kw_str.split(",") if k.strip()]
    primary = kws[0] if kws else name

    if version == "v1_baseline":
        # Baseline: short title, no keyword in title, thin meta
        payload = {
            "sku": sku,
            "title": f"{name}"[:68],
            "short_description": f"A quality {category} product.",
            "long_description": (
                f"Introducing the {name}. This product offers great features and reliable performance "
                f"for everyday use. Customers love its quality and design."
            ),
            "bullet_points": ["Quality construction", "Easy to use", "Great value"],
            "meta_description": f"Buy {name} online today.",
            "keywords_used": [],
            "tone": tone,
            "category": category,
        }
    elif version == "v2_structured":
        # Structured: includes primary keyword in title and meta, better coverage
        title = f"{primary.title()} — {name}"[:70]
        payload = {
            "sku": sku,
            "title": title,
            "short_description": f"Experience the best {primary} with {name}. Shop now.",
            "long_description": (
                f"The {name} is designed for discerning customers seeking the finest {primary}. "
                f"Engineered for performance and built to last, it meets the demands of everyday life. "
                f"{'Also featuring ' + kws[1] + ' capabilities.' if len(kws) > 1 else ''}"
            ),
            "bullet_points": [
                f"Premium {primary} quality",
                "Engineered for daily performance",
                "Backed by quality assurance",
            ],
            "meta_description": (
                f"Shop the best {primary} — {name}. Free delivery available. Order today."
            )[:160],
            "keywords_used": kws[:2],
            "tone": tone,
            "category": category,
        }
    else:
        payload = {}  # Should not be called for V3

    return json.dumps(payload)


def _make_v3_mock_result(
    sku: str, norm: Dict[str, Any], keywords: List[str]
) -> tuple:
    """Build a mock V3 result: schema-valid, keyword-rich, passes validation."""
    name = norm["name"]
    category = norm["category"]
    tone = norm["tone"] or "premium"
    primary = keywords[0] if keywords else name
    kw_str = " and ".join(keywords[:2]) if len(keywords) >= 2 else primary

    title = f"{primary.title()} — {name}"[:70]
    short = f"Shop the best {primary} — {name}. Premium quality, fast delivery."[:160]
    long_desc = (
        f"The {name} delivers exceptional {kw_str} performance built for modern lifestyles. "
        f"Crafted with premium materials and tested under real-world conditions, it sets a new "
        f"benchmark in the {category.replace('_', ' ')} category. "
        f"{'Featuring ' + (keywords[1] if len(keywords) > 1 else primary) + ' technology, ' if len(keywords) > 1 else ''}"
        f"this product is designed to exceed your expectations every day."
    )
    meta = (
        f"Discover the {primary} — {name}. Premium {category.replace('_', ' ')} quality. Shop now."
    )[:160]

    try:
        desc = ProductDescription(
            sku=sku, title=title, short_description=short,
            long_description=long_desc,
            bullet_points=[
                f"Premium {primary} quality",
                f"Built for everyday {category.replace('_', ' ')} use",
                "Durable, reliable construction",
            ],
            meta_description=meta,
            keywords_used=keywords[:3],
            tone=tone, category=category,
        )
        val = validate_product_description(desc, target_keywords=keywords)
        return desc, val, 1
    except Exception:
        return None, ValidationResult(is_valid=False, errors=["Mock build failed"]), 1


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(records: List[EvalRecord]) -> Dict[str, Any]:
    total = len(records)
    if total == 0:
        return {}
    valid = [r for r in records if r.schema_valid]
    seo_compliant = [r for r in valid if r.seo_score >= 70]
    repaired = [r for r in records if r.repair_attempts > 0]
    repaired_success = [r for r in repaired if r.schema_valid]
    return {
        "total_products": total,
        "schema_valid_count": len(valid),
        "schema_valid_rate_pct": round(100 * len(valid) / total, 1),
        "seo_score_avg": round(sum(r.seo_score for r in valid) / max(len(valid), 1), 1),
        "seo_compliant_rate_pct": round(100 * len(seo_compliant) / max(len(valid), 1), 1),
        "keyword_coverage_avg_pct": round(
            100 * sum(r.keyword_coverage_pct for r in valid) / max(len(valid), 1), 1
        ),
        "initial_valid_count": sum(1 for r in records if r.initial_valid),
        "repair_attempts_total": sum(r.repair_attempts for r in records),
        "repair_success_rate_pct": (
            round(100 * len(repaired_success) / len(repaired), 1) if repaired else None
        ),
        "avg_latency_seconds": round(sum(r.latency_seconds for r in records) / total, 2),
        "source": records[0].source if records else "unknown",
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_path: Path,
    output_path: Optional[Path],
    mock: bool,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Run all three prompt versions over the dataset and return structured results."""
    # Load dataset
    products: List[Dict[str, Any]] = []
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        products = [dict(row) for row in reader]

    print(f"Loaded {len(products)} products from {dataset_path}")
    print(f"Mode: {'OFFLINE MOCK' if mock else 'LIVE API'}")
    print(f"Prompt versions: {list(PROMPT_VERSIONS.keys())}\n")

    api_key = os.getenv("GEMINI_API_KEY") if not mock else "fake-key"
    generator = DescriptionGenerator(api_key=api_key, model_name=model, max_validation_retries=2)

    all_records: List[EvalRecord] = []
    version_records: Dict[str, List[EvalRecord]] = {v: [] for v in PROMPT_VERSIONS}

    for i, product in enumerate(products, 1):
        norm = normalize_product_dict(product)
        print(f"  [{i:02d}/{len(products)}] SKU={norm['sku']} name='{norm['name'][:30]}'")

        for version in PROMPT_VERSIONS:
            if version == "v1_baseline":
                rec = _eval_v1_or_v2(
                    product, _V1_SYSTEM, _V1_USER_TEMPLATE, version, generator, mock
                )
            elif version == "v2_structured":
                rec = _eval_v1_or_v2(
                    product, _V2_SYSTEM, _V2_USER_TEMPLATE, version, generator, mock
                )
            else:
                rec = _eval_v3(product, generator, mock)

            all_records.append(rec)
            version_records[version].append(rec)

        if not mock:
            time.sleep(1.0)  # Rate-limit pause between products (not versions)

    # Build output
    results = {
        "meta": {
            "dataset": str(dataset_path),
            "dataset_size": len(products),
            "model": model,
            "mock": mock,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_versions": PROMPT_VERSIONS,
            "cost_note": (
                "ESTIMATED: Token costs not directly measurable via this wrapper. "
                "See benchmark/eval_runner.py module docstring for assumptions."
            ),
        },
        "version_summary": {
            v: _aggregate(version_records[v]) for v in PROMPT_VERSIONS
        },
        "records": [asdict(r) for r in all_records],
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 66)
    print(f"{'Metric':<40} {'V1 Base':>8} {'V2 Struct':>9} {'V3 Full':>8}")
    print("-" * 66)
    for key, label in [
        ("schema_valid_rate_pct", "Schema Valid Rate (%)"),
        ("seo_score_avg", "Avg SEO Score"),
        ("seo_compliant_rate_pct", "SEO >=70 Compliant (%)"),
        ("keyword_coverage_avg_pct", "Keyword Coverage (%)"),
        ("avg_latency_seconds", "Avg Latency (s)"),
    ]:
        row = []
        for v in PROMPT_VERSIONS:
            val = results["version_summary"][v].get(key, "N/A")
            row.append(f"{val:>8}" if isinstance(val, (int, float)) else f"{'N/A':>8}")
        print(f"{label:<40} {row[0]} {row[1]} {row[2]}")
    print("=" * 66)
    print("\nNote: mocked results show structural differences between prompt")
    print("versions but do not represent real LLM output quality.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate prompt versions V1/V2/V3 on the product dataset."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/sample_products.csv"),
        help="Path to evaluation CSV dataset"
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Path to save JSON results (optional)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run in offline mock mode (no API calls, no key required)"
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Gemini model name (live mode only)"
    )
    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.out,
        mock=args.mock,
        model=args.model,
    )
