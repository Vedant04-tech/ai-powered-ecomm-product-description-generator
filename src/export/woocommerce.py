"""WooCommerce REST API payload export mapper for ProductDescription objects."""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from src.schema import ProductDescription
from src.export.shopify import format_body_html, format_currency


def to_woocommerce_payload(
    desc: ProductDescription,
    extra_attrs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert a ProductDescription into a WooCommerce REST API POST /wp-json/wc/v3/products payload.
    
    Adheres strictly to the WooCommerce REST API schema:
    https://woocommerce.github.io/woocommerce-rest-api-docs/#create-a-product
    """
    extras = extra_attrs or {}
    raw_price = extras.get("price", "")
    price = format_currency(raw_price)
    primary_kw = desc.keywords_used[0] if desc.keywords_used else ""

    tags = [{"name": kw} for kw in desc.keywords_used]
    categories = [{"name": desc.category.replace("_", " ").title()}]

    meta_data = [
        # Yoast SEO fields
        {"key": "_yoast_wpseo_title", "value": desc.title},
        {"key": "_yoast_wpseo_metadesc", "value": desc.meta_description},
        {"key": "_yoast_wpseo_focuskw", "value": primary_kw},
        # RankMath SEO fields
        {"key": "rank_math_title", "value": desc.title},
        {"key": "rank_math_description", "value": desc.meta_description},
        {"key": "rank_math_focus_keyword", "value": primary_kw},
        # Generation metadata
        {"key": "_ai_generated_tone", "value": desc.tone},
        {"key": "_ai_bullet_points", "value": json.dumps(desc.bullet_points)}
    ]

    payload: Dict[str, Any] = {
        "name": desc.title,
        "type": "simple",
        "sku": desc.sku,
        "status": "publish",
        "catalog_visibility": "visible",
        "manage_stock": True,
        "stock_quantity": 100,
        "stock_status": "instock",
        "description": format_body_html(desc),
        "short_description": f"<p>{desc.short_description}</p>",
        "categories": categories,
        "tags": tags,
        "meta_data": meta_data
    }

    if price:
        payload["regular_price"] = price

    return payload


def to_woocommerce_payload_json(
    descriptions: List[ProductDescription],
    extra_attrs_list: Optional[List[Dict[str, Any]]] = None,
    indent: int = 2
) -> str:
    """Generate pretty-printed JSON array of WooCommerce product payloads."""
    extras = extra_attrs_list or [{} for _ in descriptions]
    payloads = [
        to_woocommerce_payload(desc, extras[i] if i < len(extras) else {})
        for i, desc in enumerate(descriptions)
    ]
    return json.dumps(payloads, indent=indent)
