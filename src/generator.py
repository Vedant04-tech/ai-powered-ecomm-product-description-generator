"""LangChain-powered generation pipeline with intelligent column mapping, category aliasing, and Gemini API integration."""

from __future__ import annotations
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import yaml

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.schema import ProductDescription
from src.validate import validate_product_description, ValidationResult

load_dotenv()
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

# Common eCommerce category aliases mapped to the core template families
CATEGORY_ALIASES: Dict[str, str] = {
    # Electronics
    "tech": "electronics",
    "gadgets": "electronics",
    "gadget": "electronics",
    "hardware": "electronics",
    "audio": "electronics",
    "sound": "electronics",
    "headphones": "electronics",
    "earbuds": "electronics",
    "computers": "electronics",
    "computer": "electronics",
    "pc": "electronics",
    "laptop": "electronics",
    "mobile": "electronics",
    "phone": "electronics",
    "peripherals": "electronics",
    "gaming": "electronics",
    "cameras": "electronics",
    "camera": "electronics",
    # Apparel
    "clothing": "apparel",
    "fashion": "apparel",
    "clothes": "apparel",
    "shoes": "apparel",
    "shoe": "apparel",
    "footwear": "apparel",
    "garments": "apparel",
    "garment": "apparel",
    "wear": "apparel",
    "activewear": "apparel",
    "sportswear": "apparel",
    "outerwear": "apparel",
    "streetwear": "apparel",
    "accessories": "apparel",
    "t-shirt": "apparel",
    "shirts": "apparel",
    "pants": "apparel",
    # Home Goods
    "home": "home_goods",
    "living": "home_goods",
    "home_decor": "home_goods",
    "decor": "home_goods",
    "furniture": "home_goods",
    "kitchen": "home_goods",
    "kitchenware": "home_goods",
    "dining": "home_goods",
    "cookware": "home_goods",
    "houseware": "home_goods",
    "office": "home_goods",
    "workspace": "home_goods",
    "desk": "home_goods",
    "bath": "home_goods",
    "bathroom": "home_goods",
    "bedding": "home_goods",
    "lighting": "home_goods"
}


def normalize_key(k: Any) -> str:
    cleaned = re.sub(r"[#]", "no", str(k))
    cleaned = re.sub(r"[^\w\s]", " ", cleaned).strip().lower()
    return re.sub(r"\s+", "_", cleaned)


def normalize_product_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligently map real-world merchant headers to standard attributes.
    
    Accommodates variations like 'Title' -> 'name', 'Product Description' -> 'specs',
    'Item #' -> 'sku', 'Retail Price' -> 'price', etc.
    """
    # Create case-insensitive, symbol-normalized lookup
    lowered = {normalize_key(k): v for k, v in raw.items()}

    def find_val(candidates: List[str], default: Any = "") -> Any:
        for c in candidates:
            if c in lowered and lowered[c] is not None and str(lowered[c]).strip():
                return lowered[c]
        return default

    sku = str(find_val(["sku", "id", "item_id", "product_id", "item_no", "item_number", "item", "handle", "code", "model"], "SKU-UNKNOWN"))
    name = str(find_val(["name", "title", "product_name", "item_name", "product_title", "headline"], "Unnamed Product"))
    specs = str(find_val(["specs", "description", "details", "features", "specifications", "attributes", "body", "product_details", "bullets"], ""))
    category = str(find_val(["category", "type", "product_type", "department", "collection"], "electronics"))
    price = str(find_val(["price", "variant_price", "retail_price", "unit_price", "cost", "msrp", "regular_price"], ""))
    brand = str(find_val(["brand", "vendor", "manufacturer", "make"], ""))
    tone = str(find_val(["tone", "voice", "style", "brand_voice"], ""))
    keywords = find_val(["keywords", "tags", "target_keywords", "seo_keywords", "search_terms"], "")

    return {
        "sku": sku,
        "name": name,
        "specs": specs,
        "category": category,
        "price": price,
        "brand": brand,
        "tone": tone,
        "keywords": keywords,
        "_raw": raw
    }


def resolve_category(category_input: str, templates_dir: Path = TEMPLATES_DIR) -> str:
    """Resolve raw category to an existing template file using stems, aliases, and substring match."""
    clean = category_input.strip().lower().replace(" ", "_").replace("-", "_")

    available_templates = {f.stem for f in templates_dir.glob("*.yaml")}
    if clean in available_templates:
        return clean

    # Check direct aliases
    if clean in CATEGORY_ALIASES and CATEGORY_ALIASES[clean] in available_templates:
        return CATEGORY_ALIASES[clean]

    # Check substring tokens (e.g. "men's clothing" -> "clothing" -> "apparel")
    tokens = re.split(r"[\s\-_/]+", clean)
    for tok in tokens:
        if tok in available_templates:
            return tok
        if tok in CATEGORY_ALIASES and CATEGORY_ALIASES[tok] in available_templates:
            return CATEGORY_ALIASES[tok]

    # Graceful fallback: default to home_goods (broad lifestyle template) with warning
    default_cat = "home_goods" if "home_goods" in available_templates else (list(available_templates)[0] if available_templates else "electronics")
    logger.warning(
        f"Category '{category_input}' could not be matched to available templates {available_templates}. "
        f"Gracefully falling back to '{default_cat}' template."
    )
    return default_cat


def load_category_template(category: str, templates_dir: Path = TEMPLATES_DIR) -> Dict[str, Any]:
    """Load template configuration YAML dynamically for a resolved category."""
    resolved_cat = resolve_category(category, templates_dir)
    template_file = templates_dir / f"{resolved_cat}.yaml"

    if not template_file.exists():
        available = [f.stem for f in templates_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"No prompt template found for category '{resolved_cat}' at {template_file}. "
            f"Available templates: {', '.join(available)}"
        )

    with open(template_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def list_available_categories(templates_dir: Path = TEMPLATES_DIR) -> List[str]:
    """Return list of supported categories discovered in templates/ directory."""
    if not templates_dir.exists():
        return []
    return sorted([f.stem for f in templates_dir.glob("*.yaml")])


class DescriptionGenerator:
    """Manages the LangChain pipeline, template resolution, rate limits, and validation retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        temperature: float = 0.7,
        max_validation_retries: int = 2
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_validation_retries = max_validation_retries
        self.parser = PydanticOutputParser(pydantic_object=ProductDescription)
        self._llm = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-initialize ChatGoogleGenerativeAI instance."""
        if self._llm is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please set it in your .env file or pass it to DescriptionGenerator."
                )
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=self.temperature
            )
        return self._llm

    def _call_llm_with_backoff(self, prompt_messages: Any, max_attempts: int = 4) -> str:
        """Execute LLM call with exponential backoff for Gemini free-tier rate limits (HTTP 429)."""
        backoff = 2.0
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.llm.invoke(prompt_messages)
                content = response.content if hasattr(response, "content") else str(response)
                # If content is a list of parts or blocks, extract text
                if isinstance(content, list):
                    content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
                return content
            except Exception as e:
                err_str = str(e).lower()
                last_error = e
                # Check for rate limit or quota signals
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    sleep_time = backoff * (2 ** (attempt - 1))
                    logger.warning(
                        f"Gemini API rate limit encountered (attempt {attempt}/{max_attempts}). "
                        f"Backing off for {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
                elif "503" in err_str or "unavailable" in err_str:
                    sleep_time = backoff * (2 ** (attempt - 1))
                    logger.warning(
                        f"Gemini API 503 transient spike (attempt {attempt}/{max_attempts}). "
                        f"Backing off for {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"LLM call failed with non-rate-limit error: {e}")
                    raise e

        raise RuntimeError(f"Exceeded maximum retries ({max_attempts}) on Gemini API. Last error: {last_error}")

    def generate_single(
        self,
        product: Dict[str, Any],
        tone_override: Optional[str] = None,
        keywords_override: Optional[str] = None
    ) -> Tuple[Optional[ProductDescription], ValidationResult, int]:
        """Generate, validate, and retry-on-failure for a single product dictionary."""
        # Normalize headers and attributes
        norm_product = normalize_product_dict(product)
        sku = norm_product["sku"]
        name = norm_product["name"]
        raw_cat = norm_product["category"]
        brand = norm_product["brand"]
        specs = norm_product["specs"]
        price = norm_product["price"]

        # Pre-flight: reject products with no usable content before any API call
        if name == "Unnamed Product" and not specs.strip():
            logger.warning(f"SKU {sku}: skipping — both name and specs are missing. No API call made.")
            return None, ValidationResult(
                is_valid=False,
                errors=["Insufficient product data: name and specs are both missing. Provide at least a product name or specifications."]
            ), 0

        # Resolve category and template
        resolved_category = resolve_category(raw_cat)
        template_cfg = load_category_template(resolved_category)

        # Tone resolution
        selected_tone = tone_override or norm_product["tone"] or "premium"
        tone_options = template_cfg.get("tone_options", {})
        tone_guidance = tone_options.get(selected_tone, f"Write with an engaging, {selected_tone} tone.")

        # Keywords resolution
        raw_keywords = keywords_override if keywords_override is not None else norm_product["keywords"]
        if isinstance(raw_keywords, list):
            target_keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]
        else:
            target_keywords = [k.strip() for k in str(raw_keywords).split(",") if k.strip()]

        system_instruction = template_cfg.get("system_instruction", "You are an eCommerce copywriter.")
        user_template_str = template_cfg.get("user_prompt_template", "")
        format_instructions = self.parser.get_format_instructions()

        variables = {
            "sku": sku,
            "name": name,
            "category": resolved_category,
            "brand": brand,
            "specs": specs if specs else "Standard quality product specifications",
            "price": price if price else "Inquire for pricing",
            "tone": selected_tone,
            "tone_guidance": tone_guidance,
            "keywords": ", ".join(target_keywords) if target_keywords else "N/A"
        }

        # Format initial prompt
        formatted_user_prompt = user_template_str.format(**variables)
        initial_prompt = (
            f"{formatted_user_prompt}\n\n"
            f"[SCHEMA FORMAT INSTRUCTIONS]\n{format_instructions}\n"
            f"Respond with raw JSON only. Do not enclose in markdown ticks if possible, but valid JSON is required."
        )

        history_messages = [
            ("system", system_instruction),
            ("human", initial_prompt)
        ]

        last_val_result = ValidationResult(is_valid=False, errors=["Generation not attempted"])
        attempts = 0

        for attempt in range(1, self.max_validation_retries + 2):
            attempts += 1
            raw_text = self._call_llm_with_backoff(history_messages)

            # Clean markdown formatting if present
            cleaned_json = raw_text.strip()
            if "```json" in cleaned_json:
                cleaned_json = cleaned_json.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_json:
                cleaned_json = cleaned_json.split("```")[1].split("```")[0].strip()

            try:
                parsed_desc = self.parser.parse(cleaned_json)
                # Ensure category matches resolved category
                parsed_desc.category = resolved_category
                parsed_desc.sku = sku
            except Exception as parse_err:
                logger.warning(f"Schema parsing failed on SKU {sku} (attempt {attempt}): {parse_err}")
                last_val_result = ValidationResult(
                    is_valid=False,
                    errors=[f"JSON Schema Parse Error: {parse_err}"],
                    retry_prompt_feedback=(
                        f"Your output failed JSON parsing with error: {parse_err}. "
                        f"Output strictly valid JSON conforming to the schema."
                    )
                )
                if attempt <= self.max_validation_retries:
                    history_messages.append(("ai", raw_text))
                    history_messages.append(("human", last_val_result.retry_prompt_feedback or "Fix JSON"))
                continue

            # Validate quality, SEO, and readability
            val_result = validate_product_description(parsed_desc, target_keywords=target_keywords)
            last_val_result = val_result

            if val_result.is_valid:
                return parsed_desc, val_result, attempts

            # Validation failed, inject feedback and retry if under retry cap
            if attempt <= self.max_validation_retries:
                logger.info(
                    f"SKU {sku} failed validation on attempt {attempt}: {val_result.errors}. Retrying with feedback..."
                )
                history_messages.append(("ai", raw_text))
                history_messages.append(("human", val_result.retry_prompt_feedback or "Fix validation errors"))

        # Exceeded retry cap, fail gracefully
        logger.warning(f"SKU {sku} failed after {attempts} attempt(s). Logging and skipping per PRD convention.")
        return None, last_val_result, attempts

    def generate_batch(
        self,
        products: List[Dict[str, Any]],
        tone_override: Optional[str] = None,
        keywords_override: Optional[str] = None,
        delay_seconds: float = 1.0,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Iterate over products and execute batch generation with rate-limit pauses and progress tracking."""
        start_time = time.time()
        successful: List[ProductDescription] = []
        failed: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []

        total = len(products)
        for idx, product in enumerate(products, 1):
            norm_prod = normalize_product_dict(product)
            sku = norm_prod["sku"]
            try:
                desc, val_res, attempts = self.generate_single(
                    product=product,
                    tone_override=tone_override,
                    keywords_override=keywords_override
                )

                status = "PASS" if val_res.is_valid else ("RETRY_FAILED" if attempts > 1 else "FAIL")

                record = {
                    "sku": sku,
                    "name": norm_prod["name"],
                    "category": desc.category if desc else norm_prod["category"],
                    "status": status,
                    "attempts": attempts,
                    "seo_score": val_res.seo_score,
                    "readability_score": val_res.readability_score,
                    "readability_level": val_res.readability_level,
                    "seo_breakdown": val_res.seo_breakdown,
                    "errors": val_res.errors,
                    "warnings": val_res.warnings,
                    "description": desc.model_dump() if desc else None,
                    "extra_attrs": norm_prod
                }
                records.append(record)

                if desc and val_res.is_valid:
                    successful.append(desc)
                else:
                    failed.append(record)

            except Exception as e:
                logger.error(f"Unexpected error processing SKU {sku}: {e}")
                fail_record = {
                    "sku": sku,
                    "name": norm_prod["name"],
                    "category": norm_prod["category"],
                    "status": "CRASH_SKIPPED",
                    "attempts": 1,
                    "seo_score": 0,
                    "readability_score": 0.0,
                    "readability_level": "N/A",
                    "seo_breakdown": {},
                    "errors": [str(e)],
                    "warnings": [],
                    "description": None,
                    "extra_attrs": norm_prod
                }
                records.append(fail_record)
                failed.append(fail_record)

            if progress_callback:
                progress_callback(idx, total, records[-1])

            # Free-tier rate-limit pause between items
            if idx < total and delay_seconds > 0:
                time.sleep(delay_seconds)

        wall_clock = time.time() - start_time
        return {
            "total_processed": total,
            "success_count": len(successful),
            "fail_count": len(failed),
            "wall_clock_seconds": round(wall_clock, 2),
            "records": records,
            "successful_descriptions": successful
        }
