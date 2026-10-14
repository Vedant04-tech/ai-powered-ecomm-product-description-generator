"""Shopify CSV export mapper for ProductDescription objects."""

from __future__ import annotations
import io
import re
import csv
from typing import Dict, Any, List, Optional
from src.schema import ProductDescription


def format_currency(price_val: Any) -> str:
    """Format numerical or string price into standard 2-decimal currency string (e.g. '79.90')."""
    if price_val is None or str(price_val).strip() == "":
        return ""
    clean = re.sub(r"[^\d.]", "", str(price_val).strip())
    try:
        flt = float(clean)
        return f"{flt:.2f}"
    except (ValueError, TypeError):
        return str(price_val).strip()


def generate_handle(title: str, sku: str = "", unique_suffix: bool = False) -> str:
    """Generate URL-safe Shopify product handle with optional SKU collision avoidance."""
    clean = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", clean)
    base_slug = slug if slug else "product"

    if unique_suffix and sku:
        sku_clean = re.sub(r"[^\w-]", "", sku.lower()).strip()
        return f"{base_slug}-{sku_clean}"
    return base_slug if slug else f"product-{sku.lower()}"


def format_body_html(desc: ProductDescription) -> str:
    """Format long description and bullet points as rich HTML for Shopify Body (HTML)."""
    paragraphs = [f"<p>{p.strip()}</p>" for p in desc.long_description.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [f"<p>{desc.long_description.strip()}</p>"]
    
    bullets_html = ""
    if desc.bullet_points:
        bullets_items = "".join(f"<li>{b.strip()}</li>" for b in desc.bullet_points)
        bullets_html = f"<ul>{bullets_items}</ul>"
    
    return f"{''.join(paragraphs)}\n{bullets_html}"


def to_shopify_csv_row(
    desc: ProductDescription,
    extra_attrs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert a ProductDescription instance into a Shopify product import CSV row dict.
    
    Adheres strictly to the official Shopify CSV format requirements:
    https://help.shopify.com/en/manual/products/import-export/using-csv
    """
    extras = extra_attrs or {}
    vendor = str(extras.get("brand", "") or "Default Brand")
    raw_price = extras.get("price", "")
    price = format_currency(raw_price)

    tags_list = list(dict.fromkeys(desc.keywords_used + [desc.category, desc.tone]))
    tags = ", ".join(t for t in tags_list if t)

    return {
        "Handle": generate_handle(desc.title, desc.sku),
        "Title": desc.title,
        "Body (HTML)": format_body_html(desc),
        "Vendor": vendor,
        "Product Category": desc.category.replace("_", " ").title(),
        "Type": desc.category.replace("_", " ").title(),
        "Tags": tags,
        "Published": "TRUE",
        "Option1 Name": "Title",
        "Option1 Value": "Default Title",
        "Variant SKU": desc.sku,
        "Variant Grams": "0",
        "Variant Inventory Tracker": "shopify",
        "Variant Inventory Qty": "100",
        "Variant Inventory Policy": "deny",
        "Variant Fulfillment Service": "manual",
        "Variant Price": price,
        "Variant Requires Shipping": "TRUE",
        "Variant Taxable": "TRUE",
        "SEO Title": desc.title,
        "SEO Description": desc.meta_description,
        "Status": "active"
    }


def to_shopify_csv(
    descriptions: List[ProductDescription],
    extra_attrs_list: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Generate complete Shopify-compatible CSV string from descriptions."""
    if not descriptions:
        return ""
    
    output = io.StringIO()
    extras = extra_attrs_list or [{} for _ in descriptions]
    rows = [to_shopify_csv_row(desc, extras[i] if i < len(extras) else {}) for i, desc in enumerate(descriptions)]
    
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
