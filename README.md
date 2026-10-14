# AI-Powered eCommerce Product Description Generator

An SEO-optimized eCommerce product description generation pipeline built with **LangChain**, **Google Gemini**, **Pydantic**, and **Streamlit**. It transforms raw merchant product attributes (CSV/JSON) into schema-validated, conversion-optimized copy and maps them directly to **Shopify** and **WooCommerce** catalog specifications.

---

## 🌟 Resume Claim → Engineering Proof Mapping

| Claim | Engineering Implementation | Proof / Verification |
|---|---|---|
| **"Cutting manual writing effort by over 70%"** | Quantitative benchmark (`benchmark/effort_benchmark.py`) comparing pipeline wall-clock execution + human review against timed copywriting baselines and labor rates ($30/hr). | **Simulated 91.5% effort reduction** at default parameters (100 SKUs, 12 min/SKU manual baseline, 1 min review). This is a model, not a measured field result. |
| **"SEO-optimized listings"** | Programmatic validator (`src/validate.py`) calculating a **100-point SEO Scorecard** and **Flesch Reading Ease score** while enforcing keyword placement in titles, first 100 words, and meta descriptions. | Automated scoring and word-boundary regex containment checks (`\bkeyword\b`). |
| **"Centralized reusable prompt template library"** | Category-agnostic architecture (`templates/*.yaml`) with **intelligent category aliasing** (`tech`, `audio`, `footwear`, `kitchen` automatically resolved). | YAML-driven engine with zero hardcoded Python branching. |
| **"Tunable tone and keyword control"** | Runtime parameter injection (`{tone}`, `{keywords}`) decoupled from template schemas. | Configurable via CLI arguments and interactive Streamlit UI selectors. |
| **"Structured output & schema validation"** | Pydantic model (`src/schema.py`) with bounded 1–2 attempt self-healing retry logic and real-world column normalization. | 100% schema enforcement; invalid JSON or out-of-spec lengths are re-prompted or gracefully flagged. |
| **"Direct Shopify & WooCommerce integrations"** | Format mappers (`src/export/shopify.py`, `src/export/woocommerce.py`) matching official schemas with currency formatting and inventory readiness. | Generates 1-click importable Shopify CSVs and WooCommerce REST API payloads. |

---

## 🏗️ System Architecture

```
.
├── templates/                 # Reusable category prompt templates (YAML)
│   ├── electronics.yaml
│   ├── apparel.yaml
│   └── home_goods.yaml
├── src/
│   ├── schema.py              # Pydantic ProductDescription data contract
│   ├── validate.py            # Flesch readability, 100-pt SEO scoring, regex matching
│   ├── generator.py           # LangChain pipeline, category aliasing, column normalizer
│   ├── export/
│   │   ├── shopify.py         # Currency-formatted Shopify product import CSV mapper
│   │   └── woocommerce.py     # Inventory-ready WooCommerce REST API JSON mapper
│   └── cli.py                 # Rich-formatted CLI batch execution entrypoint
├── dashboard/
│   └── app.py                 # Streamlit UI with column mapper & inline copy-editor
├── data/
│   └── sample_products.csv    # 18 synthetic multi-category sample products (Electronics, Apparel, Home Goods)
├── benchmark/
│   ├── effort_benchmark.py    # Time savings and financial ROI engine (simulated model)
│   └── eval_runner.py         # Prompt-version evaluation (V1/V2/V3 comparison, offline-capable)
├── tests/                     # 38+ pytest unit tests
│   ├── conftest.py            # API key isolation fixture
│   ├── test_schema.py
│   ├── test_validate.py
│   ├── test_export.py
│   ├── test_benchmark.py
│   ├── test_generator.py
│   └── test_eval.py
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

---

## ⚠️ Known Limitations

- **Cost estimates are illustrative.** The `effort_benchmark.py` uses hardcoded parameters, not measured production data. The `eval_runner.py` marks all cost figures as ESTIMATED.
- **Hallucination prevention is prompt-based.** The pipeline instructs the model to use only supplied facts but cannot programmatically verify every generated claim. Treat AI output as a draft requiring human review.
- **No live publishing.** Export files are ready to import into Shopify/WooCommerce but the project does not connect to live stores.
- **Gemini free tier limits.** The default delay between API calls (1–1.5 s) is tuned for free-tier quotas. Paid accounts can reduce this.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Navigate into the project folder
cd "Product Discription Generator"

# Create virtual environment with Python 3.11+
py -3.11 -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Gemini API Key
Obtain a free-tier API key from [Google AI Studio](https://aistudio.google.com/):
```bash
cp .env.example .env
```
Open `.env` and paste your key (no leading spaces):
```ini
GEMINI_API_KEY=AIzaSy...
```

---

## 💻 CLI Usage

Run batch generation over any CSV or JSON file (supports custom headers like `Title`, `Item #`, `Details`, `Retail Price`):

```bash
# Basic batch run saving results to JSON
python -m src.cli --input data/sample_products.csv --out output/results.json

# Batch run with direct Shopify CSV and WooCommerce JSON exports
python -m src.cli \
  --input data/sample_products.csv \
  --out output/results.json \
  --export-shopify output/shopify_import.csv \
  --export-woo output/woocommerce_payload.json \
  --tone premium \
  --model gemini-2.5-flash
```

---

## 🖥️ Streamlit Interactive Dashboard

Launch the web dashboard:
```bash
streamlit run dashboard/app.py
```

### Dashboard Features:
1. **Catalog Upload & Column Normalization**: Automatically maps real-world merchant headers (`Product Name`, `Item #`, `Description`, `Retail Price`).
2. **Prompt Transparency Inspector**: Inspect YAML templates and subcategory aliasing (`audio`, `tech`, `shoes`, `kitchen`).
3. **Batch Generation with Progress**: Real-time batch progress tracking with automatic rate-limit pacing.
4. **Validation Table with SEO Score Badges**: Displays 100-point SEO Score and Flesch Reading Ease level for every SKU.
5. **Interactive Inline Copy-Editor**: Modify the title, bullets, or long description directly in the UI before export.
6. **1-Click Export Downloads**: Export to Validated JSON, Shopify CSV, or WooCommerce JSON.
7. **Dual Time & Financial ROI Benchmark Panel**: Side-by-side charts showing simulated hours saved and cost reduction.

---

## 🧪 Running Unit Tests

Run the full pytest suite (all tests run offline — no API key required):
```bash
pytest -v tests/
```

The suite covers:
- Schema constraints & Pydantic validators
- Flesch Reading Ease algorithm & 100-point SEO Scorecard
- Word-boundary regex keyword containment (`\bkeyword\b`)
- Intelligent column normalizer & category aliasing
- Pre-flight input validation (empty products rejected before API calls)
- Shopify CSV currency formatting & collision-proof handles
- WooCommerce stock parameters (`manage_stock: True`, `stock_status: "instock"`)
- Financial ROI calculations
- Evaluation runner offline mode (mock V1/V2/V3 comparison)

---

## 📊 Prompt Engineering Evaluation (V1 / V2 / V3)

Run the offline evaluation (no API key needed):
```bash
# Offline mode — mocked responses, no API calls
python -m benchmark.eval_runner --mock --out output/eval_results.json

# Live mode — requires GEMINI_API_KEY
python -m benchmark.eval_runner --dataset data/sample_products.csv --out output/eval_results.json
```

Three prompt strategies are compared on the same 18-product dataset:

| Version | Description |
|---------|-------------|
| **V1 Baseline** | Bare product facts only; no schema, tone, or SEO instructions |
| **V2 Structured** | Explicit JSON schema fields, tone, SEO keyword rules, and character limits |
| **V3 Full Pipeline** | Category YAML template + tone guidance + keyword injection + Pydantic validation + bounded repair |

Metrics reported per version:
1. Schema-valid output rate
2. Average SEO score (out of 100)
3. SEO ≥70 compliance rate
4. Keyword coverage %
5. Initial validation pass rate (V3 only)
6. Repair success rate (V3 only)
7. Average latency per product

> **Note:** Mocked results demonstrate structural differences between prompt versions but do not reflect real LLM output quality. Label all results with `source: mock` or `source: live`.

---

## 📊 Effort & Financial ROI Benchmark

Run the standalone effort model:
```bash
python -m benchmark.effort_benchmark --skus 100 --wallclock 120 --hourly-rate 30.0
```

```text
============================================================
                EFFORT & ROI BENCHMARK                      
============================================================
Products Processed (SKUs)      : 100
Manual Baseline (per SKU)      : 12.0 mins
Estimated Manual Effort        : 20.00 hours
Pipeline Wall-Clock Run Time   : 120.0 seconds
Human Review Time (per SKU)    : 1.0 mins
Total Pipeline + Review Effort : 1.70 hours
Net Hours Saved                : 18.30 hours
------------------------------------------------------------
EFFORT REDUCTION               : 91.5%  [SIMULATED — see note below]
============================================================
FINANCIAL ROI (at $30.00/hr labor rate)
------------------------------------------------------------
Estimated Manual Labor Cost    : $600.00
AI Pipeline + Review Cost      : $50.20 (API: $0.20)
Net Financial Savings          : $549.80
COST REDUCTION                 : 91.6%  [SIMULATED — see note below]
============================================================
```

> **Important:** These numbers are the output of an arithmetic model using the inputs shown. The 12 min/SKU manual baseline is an industry estimate, not a measured time study. The $0.002/SKU API cost is an estimate based on approximate token usage and public Gemini pricing as of Oct 2025. Do not cite these as measured production results.

---

## Supported Categories & Templates

| Category | Template File | Supported Tones |
|----------|--------------|-----------------|
| Electronics & Gadgets | `electronics.yaml` | premium, technical, casual |
| Apparel & Fashion | `apparel.yaml` | premium, technical, casual |
| Home Goods & Living | `home_goods.yaml` | premium, technical, casual |

Category aliases (e.g., `tech`, `audio`, `footwear`, `kitchen`) automatically resolve to the correct template.

---

## Shopify & WooCommerce Export

**Shopify CSV:** Conforms to [Shopify product import format](https://help.shopify.com/en/manual/products/import-export/using-csv). Fields include `Handle`, `Title`, `Body (HTML)`, `Vendor`, `Tags`, `Variant SKU`, `Variant Price`, `SEO Title`, `SEO Description`.

**WooCommerce JSON:** Conforms to [WooCommerce REST API v3](https://woocommerce.github.io/woocommerce-rest-api-docs/#create-a-product). Includes Yoast SEO and RankMath meta fields. Does **not** POST to a live store — it generates ready-to-use import payloads.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details. Built for portfolio demonstration and production eCommerce catalog generation.
