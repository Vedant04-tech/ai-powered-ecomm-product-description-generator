"""Streamlit frontend dashboard for interactive eCommerce product description generation.

Adheres strictly to PRD.md §6.7 and CLAUDE.md conventions:
- Thin UI layer: imports from src/ and benchmark/ with zero business logic duplicated here.
- Covers: upload with intelligent column normalizer, prompt template transparency, batch run with progress,
  results review table with SEO & Readability badges, inline copy-editor, export buttons, and dual Time/Dollar benchmark ROI metrics.
"""

from __future__ import annotations
import io
import json
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path when launched via `streamlit run dashboard/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.generator import (
    DescriptionGenerator,
    list_available_categories,
    load_category_template,
    normalize_product_dict,
    CATEGORY_ALIASES
)
from src.schema import ProductDescription
from src.validate import calculate_seo_score, calculate_flesch_reading_ease
from src.export.shopify import to_shopify_csv
from src.export.woocommerce import to_woocommerce_payload_json
from benchmark.effort_benchmark import (
    calculate_effort_reduction,
    DEFAULT_MANUAL_MINUTES_PER_SKU,
    DEFAULT_REVIEW_MINUTES_PER_SKU,
    DEFAULT_HOURLY_LABOR_RATE_USD
)

# Configure Streamlit page
st.set_page_config(
    page_title="AI eCommerce Product Description Generator",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .score-badge-high {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
    .score-badge-med {
        background-color: #fef9c3;
        color: #a16207;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
    .score-badge-low {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "products_df" not in st.session_state:
    st.session_state.products_df = None
if "normalized_products" not in st.session_state:
    st.session_state.normalized_products = None

# Sidebar: Configuration & API Settings
st.sidebar.title("⚙️ Generator Settings")

api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Provided via .env or entered directly here for session use."
)

model_choice = st.sidebar.selectbox(
    "Model Selection",
    options=["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"],
    index=0,
    help="Fast and rate-limit friendly for batch eCommerce generation."
)

global_tone = st.sidebar.selectbox(
    "Default Tone (if not in CSV)",
    options=["premium", "casual", "technical"],
    index=0,
    help="Applied to products lacking a specific tone attribute."
)

api_delay = st.sidebar.slider(
    "Rate Limit Pause (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=1.5,
    step=0.5,
    help="Pacing between requests to prevent free-tier quota exhaustion."
)

# Header Section
st.markdown('<div class="main-header">🛍️ AI Product Description Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enterprise SEO-optimized copy pipeline with category prompt templates, '
    'Flesch readability scoring, 100-pt SEO grading, and 1-click Shopify/WooCommerce exports.</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["1. Upload & Normalize", "2. Prompt Transparency", "3. Batch Generate & Review", "4. Effort & ROI Benchmark"])

# TAB 1: Upload & Normalize
with tabs[0]:
    st.subheader("1. Catalog Upload & Column Normalization")
    st.write(
        "Upload any merchant CSV or JSON file. The intelligent column normalizer automatically detects headers "
        "like `Title`, `Product Name`, `Item #`, `Description`, `Retail Price`, and `Vendor`."
    )

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Product Attributes (CSV or JSON)",
            type=["csv", "json"],
            help="Supports standard and custom eCommerce catalog exports."
        )
    with col_sample:
        st.write("Quick start?")
        if st.button("Load Built-in Sample Catalog"):
            sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_products.csv")
            if os.path.exists(sample_path):
                raw_df = pd.read_csv(sample_path)
                st.session_state.products_df = raw_df
                # Normalize
                records = raw_df.to_dict(orient="records")
                st.session_state.normalized_products = [normalize_product_dict(r) for r in records]
                st.success("Loaded and normalized 6 sample products across Electronics, Apparel, and Home Goods!")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                data = json.load(uploaded_file)
                raw_df = pd.DataFrame(data if isinstance(data, list) else [data])
            st.session_state.products_df = raw_df
            records = raw_df.to_dict(orient="records")
            st.session_state.normalized_products = [normalize_product_dict(r) for r in records]
            st.success(f"Successfully loaded and normalized {len(st.session_state.products_df)} products!")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

    if st.session_state.products_df is not None and st.session_state.normalized_products is not None:
        st.write("### Normalized Attribute Mapping Preview")
        norm_preview_df = pd.DataFrame([
            {
                "SKU": p["sku"],
                "Product Name": p["name"],
                "Category": p["category"],
                "Brand": p["brand"],
                "Price": p["price"],
                "Tone": p["tone"] if p["tone"] else f"(Default: {global_tone})",
                "Specs Snippet": p["specs"][:40] + "..." if len(p["specs"]) > 40 else p["specs"],
                "Target Keywords": p["keywords"]
            }
            for p in st.session_state.normalized_products
        ])
        st.dataframe(norm_preview_df, use_container_width=True)

# TAB 2: Prompt Template Transparency Panel
with tabs[1]:
    st.subheader("🔍 Prompt Template Library & Category Aliases")
    st.info(
        "PRD §6.1 & §6.7 Requirement: Inspect the centralized prompt template library. "
        "Zero hardcoded category logic in Python; templates are versioned YAML files with runtime tone and keyword slots."
    )

    available_cats = list_available_categories()
    c_cat, c_alias = st.columns([1, 1])

    with c_cat:
        if available_cats:
            selected_cat = st.selectbox("Select Template to Inspect", options=available_cats)
            template_info = load_category_template(selected_cat)

            st.markdown(f"**Display Name:** {template_info.get('display_name', selected_cat)}")
            st.markdown(f"**Description:** {template_info.get('description', '')}")
            st.markdown(f"**Required Attributes:** `{', '.join(template_info.get('required_variables', []))}`")

            st.markdown("**Tone Profiles:**")
            tones = template_info.get("tone_options", {})
            for t_name, t_desc in tones.items():
                st.markdown(f"- **{t_name.title()}:** {t_desc}")

            with st.expander("View System Instructions", expanded=False):
                st.text(template_info.get("system_instruction", ""))

            with st.expander("View User Prompt Template", expanded=False):
                st.code(template_info.get("user_prompt_template", ""), language="yaml")

    with c_alias:
        st.markdown("### Intelligent Category Aliases")
        st.write(
            "When products contain granular merchant subcategories, they automatically map to the matching prompt template:"
        )
        alias_df = pd.DataFrame([
            {"Subcategory / Keyword": k, "Maps to Template": v.title()}
            for k, v in sorted(CATEGORY_ALIASES.items())
        ])
        st.dataframe(alias_df, height=320, use_container_width=True)

# TAB 3: Batch Generate & Review
with tabs[2]:
    st.subheader("⚡ Batch Description Generation")

    if st.session_state.normalized_products is None:
        st.info("Please upload a file or load the sample catalog in Tab 1 first.")
    else:
        products_list = st.session_state.normalized_products
        st.write(f"Ready to generate descriptions for **{len(products_list)} items**.")

        if st.button("🚀 Start Batch Generation", type="primary"):
            if not api_key:
                st.error("Please supply a Gemini API Key in the sidebar to proceed.")
            else:
                generator = DescriptionGenerator(
                    api_key=api_key,
                    model_name=model_choice
                )
                progress_bar = st.progress(0.0)
                status_box = st.empty()

                def streamlit_progress(curr: int, tot: int, rec: dict):
                    pct = curr / tot
                    progress_bar.progress(pct)
                    status_box.markdown(
                        f"**Processing ({curr}/{tot}):** SKU `{rec['sku']}` — "
                        f"Status: **{rec['status']}** | SEO Score: **{rec.get('seo_score', 'N/A')}/100**"
                    )

                with st.spinner("Generating SEO-optimized product listings..."):
                    batch_res = generator.generate_batch(
                        products=products_list,
                        tone_override=global_tone,
                        delay_seconds=api_delay,
                        progress_callback=streamlit_progress
                    )
                    st.session_state.batch_results = batch_res
                    progress_bar.progress(1.0)
                    status_box.success(
                        f"Batch complete! Generated {batch_res['success_count']}/{batch_res['total_processed']} "
                        f"successful listings in {batch_res['wall_clock_seconds']} seconds."
                    )

    # Review Table and Detail Inspector
    if st.session_state.batch_results is not None:
        st.divider()
        st.subheader("📋 Results Review & Validation Table")

        records = st.session_state.batch_results["records"]

        # Filter by status
        filter_status = st.selectbox("Filter by Status", ["All", "PASS", "RETRY_FAILED", "FAIL"])
        filtered_records = records if filter_status == "All" else [r for r in records if r["status"] == filter_status]

        # Build summary table for display
        table_rows = []
        for r in filtered_records:
            desc = r.get("description") or {}
            table_rows.append({
                "SKU": r["sku"],
                "Product Name": r["name"],
                "Category": r["category"],
                "Status": r["status"],
                "SEO Score": f"{r.get('seo_score', 0)}/100",
                "Readability": f"{r.get('readability_score', 0.0)} ({r.get('readability_level', 'Standard')})",
                "SEO Title": desc.get("title", ""),
                "Short Description": desc.get("short_description", ""),
                "Meta Description": desc.get("meta_description", ""),
                "Keywords Used": ", ".join(desc.get("keywords_used", []))
            })

        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df, use_container_width=True)

        # Row Detail Inspector & Inline Copy-Editor
        st.write("### 🔍 Row Detail Inspector & Inline Editor")
        sku_options = [r["sku"] for r in records]
        selected_sku = st.selectbox("Select SKU to inspect and edit:", options=sku_options)

        selected_record = next((r for r in records if r["sku"] == selected_sku), None)
        if selected_record and selected_record.get("description"):
            desc_data = selected_record["description"]

            col_view, col_edit = st.columns([1, 1])

            with col_view:
                st.markdown(f"#### {desc_data.get('title', '')}")
                st.caption(f"Category: **{desc_data.get('category')}** | Tone: **{desc_data.get('tone')}** | SKU: `{desc_data.get('sku')}`")

                # Score metrics
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    score = selected_record.get("seo_score", 100)
                    badge_class = "score-badge-high" if score >= 85 else ("score-badge-med" if score >= 70 else "score-badge-low")
                    st.markdown(f"**SEO Score:** <span class='{badge_class}'>{score}/100</span>", unsafe_allow_html=True)
                with s_col2:
                    st.markdown(f"**Readability:** `{selected_record.get('readability_score', 70.0)}` — *{selected_record.get('readability_level', 'Standard')}*")

                st.markdown(f"**Short Hook:** {desc_data.get('short_description')}")
                st.markdown("**Bullet Points:**")
                for bp in desc_data.get("bullet_points", []):
                    st.markdown(f"- {bp}")
                st.markdown("**Long Narrative:**")
                st.write(desc_data.get("long_description"))
                st.info(f"**Meta Description:** {desc_data.get('meta_description')}")

            with col_edit:
                st.markdown("#### ✏️ Edit Listing Before Export")
                with st.form(key=f"edit_form_{selected_sku}"):
                    edit_title = st.text_input("Title (max 70 chars)", value=desc_data.get("title", ""), max_chars=70)
                    edit_short = st.text_area("Short Description (max 160 chars)", value=desc_data.get("short_description", ""), max_chars=160, height=80)
                    bullets_str = "\n".join(desc_data.get("bullet_points", []))
                    edit_bullets = st.text_area("Bullet Points (one per line, 3-5 items)", value=bullets_str, height=120)
                    edit_long = st.text_area("Long Description", value=desc_data.get("long_description", ""), height=150)
                    edit_meta = st.text_area("Meta Description (max 160 chars)", value=desc_data.get("meta_description", ""), max_chars=160, height=80)

                    if st.form_submit_button("💾 Save Changes to Session"):
                        # Parse bullets
                        new_bullets = [b.strip() for b in edit_bullets.split("\n") if b.strip()]
                        if len(new_bullets) < 3:
                            st.error("Please provide at least 3 bullet points.")
                        else:
                            # Update record
                            desc_data["title"] = edit_title
                            desc_data["short_description"] = edit_short
                            desc_data["bullet_points"] = new_bullets
                            desc_data["long_description"] = edit_long
                            desc_data["meta_description"] = edit_meta

                            # Recalculate SEO & Readability
                            target_kws = selected_record.get("extra_attrs", {}).get("keywords", "")
                            kw_list = [k.strip() for k in target_kws.split(",") if k.strip()] if isinstance(target_kws, str) else target_kws
                            temp_obj = ProductDescription(**desc_data)
                            new_score, _ = calculate_seo_score(temp_obj, kw_list)
                            new_flesch, new_tier = calculate_flesch_reading_ease(edit_long)

                            selected_record["seo_score"] = new_score
                            selected_record["readability_score"] = new_flesch
                            selected_record["readability_level"] = new_tier

                            # Update in successful descriptions list
                            for i, s_desc in enumerate(st.session_state.batch_results["successful_descriptions"]):
                                if s_desc.sku == selected_sku:
                                    st.session_state.batch_results["successful_descriptions"][i] = temp_obj
                                    break

                            st.success("Successfully saved changes! Exports will include your updated copy.")
                            st.rerun()

        # Platform Export Section
        st.divider()
        st.subheader("📦 1-Click Platform Export (Shopify & WooCommerce)")
        st.write("Download formatted files matching official platform specifications.")

        success_descs = st.session_state.batch_results["successful_descriptions"]
        extra_attrs = [r["extra_attrs"] for r in records if r["description"] is not None]

        col_json, col_shop, col_woo = st.columns(3)
        with col_json:
            raw_json_str = json.dumps([d.model_dump() for d in success_descs], indent=2)
            st.download_button(
                label="📥 Download Validated JSON",
                data=raw_json_str,
                file_name="product_descriptions.json",
                mime="application/json",
                use_container_width=True
            )

        with col_shop:
            shopify_csv_str = to_shopify_csv(success_descs, extra_attrs)
            st.download_button(
                label="🛍️ Download Shopify CSV",
                data=shopify_csv_str,
                file_name="shopify_products_import.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_woo:
            woo_json_str = to_woocommerce_payload_json(success_descs, extra_attrs)
            st.download_button(
                label="🟣 Download WooCommerce JSON",
                data=woo_json_str,
                file_name="woocommerce_products.json",
                mime="application/json",
                use_container_width=True
            )

# TAB 4: Effort & ROI Benchmark Panel
with tabs[3]:
    st.subheader("⏱️ Empirical Effort & Financial ROI Benchmark")
    st.markdown(
        "Substantiating the **'>70% manual effort reduction'** claim with empirical labor time and financial ROI."
    )

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        manual_min = st.number_input(
            "Manual Copywriting Baseline (Minutes / SKU)",
            min_value=5.0,
            max_value=30.0,
            value=DEFAULT_MANUAL_MINUTES_PER_SKU,
            step=1.0,
            help="Time required for a human copywriter to research keywords, draft title/body/bullets, and format SEO tags."
        )
    with col_b2:
        review_min = st.number_input(
            "Human Review Time (Minutes / SKU)",
            min_value=0.2,
            max_value=5.0,
            value=DEFAULT_REVIEW_MINUTES_PER_SKU,
            step=0.2,
            help="Human-in-the-loop review and proofreading time per AI generation in the dashboard."
        )
    with col_b3:
        hourly_rate = st.number_input(
            "Copywriter Hourly Rate ($ / hour)",
            min_value=15.0,
            max_value=100.0,
            value=DEFAULT_HOURLY_LABOR_RATE_USD,
            step=5.0,
            help="Standard market rate for professional eCommerce copywriters."
        )

    # Use current batch run or simulation
    if st.session_state.batch_results is not None:
        sku_count = st.session_state.batch_results["total_processed"]
        wall_clock = st.session_state.batch_results["wall_clock_seconds"]
        source_label = f"Current Live Batch Run ({sku_count} SKUs)"
    else:
        sku_count = 100
        wall_clock = 120.0
        source_label = "Simulated Catalog (100 SKUs)"

    bm = calculate_effort_reduction(
        sku_count=sku_count,
        pipeline_wall_clock_seconds=wall_clock,
        manual_minutes_per_sku=manual_min,
        review_minutes_per_sku=review_min,
        hourly_labor_rate=hourly_rate
    )

    st.markdown(f"### Performance & ROI Metrics: *{source_label}*")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Manual Effort", f"{bm.manual_total_hours:.1f} hrs", f"${bm.manual_cost_usd:.0f} labor")
    with m2:
        st.metric("AI Pipeline + Review", f"{bm.pipeline_total_hours:.2f} hrs", f"${bm.pipeline_total_cost_usd:.1f} total")
    with m3:
        st.metric("Net Hours Saved", f"{bm.hours_saved:.1f} hrs", f"${bm.dollars_saved:.0f} saved")
    with m4:
        st.metric("Effort & Cost Reduction", f"{bm.effort_reduction_pct:.1f}%", f"{bm.cost_reduction_pct:.1f}% cost saved")

    # Visual comparison charts
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.markdown("#### ⏳ Time Required (Hours)")
        chart_df = pd.DataFrame({
            "Approach": ["Manual Copywriter", "AI Pipeline + Review"],
            "Hours": [bm.manual_total_hours, bm.pipeline_total_hours]
        })
        st.bar_chart(chart_df.set_index("Approach"))

    with c_chart2:
        st.markdown("#### 💰 Financial Cost Comparison ($ USD)")
        cost_df = pd.DataFrame({
            "Approach": ["Manual Copywriter", "AI Pipeline + Review"],
            "Cost ($)": [bm.manual_cost_usd, bm.pipeline_total_cost_usd]
        })
        st.bar_chart(cost_df.set_index("Approach"))

    with st.expander("View Full Benchmark Report", expanded=False):
        st.code(bm.summary_table(), language="text")
