"""CLI entrypoint for batch product description generation."""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from src.generator import DescriptionGenerator
from src.export.shopify import to_shopify_csv
from src.export.woocommerce import to_woocommerce_payload_json

app = typer.Typer(help="AI-Powered eCommerce Product Description Generator CLI")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def read_products_from_file(file_path: Path) -> list[dict]:
    """Load product rows from CSV or JSON file."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    elif suffix == ".json":
        with open(file_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError("JSON file must contain an array of product objects or a single product object.")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Must be .csv or .json")


@app.command()
def generate(
    input_file: Path = typer.Option(..., "--input", "-i", help="Path to input CSV or JSON products file"),
    output_file: Optional[Path] = typer.Option(None, "--out", "-o", help="Path to save output JSON results"),
    shopify_out: Optional[Path] = typer.Option(None, "--export-shopify", help="Path to export Shopify import CSV"),
    woo_out: Optional[Path] = typer.Option(None, "--export-woo", help="Path to export WooCommerce JSON payload"),
    tone: Optional[str] = typer.Option(None, "--tone", "-t", help="Global tone override (e.g. premium, casual, technical)"),
    model: str = typer.Option("gemini-3.8-flash", "--model", "-m", help="Gemini model name"),
    delay: float = typer.Option(1.5, "--delay", help="Seconds delay between API calls for rate limiting")
):
    """Run batch product description generation over an input file."""
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(code=1)

    products = read_products_from_file(input_file)
    console.print(f"[bold cyan]Loaded {len(products)} products from {input_file}[/bold cyan]")

    generator = DescriptionGenerator(model_name=model)

    with console.status("[bold green]Generating product descriptions...") as status:
        def on_progress(current: int, total: int, record: dict):
            status.update(f"[bold green]Processing SKU {record['sku']} ({current}/{total})... Status: {record['status']}")

        result = generator.generate_batch(
            products=products,
            tone_override=tone,
            delay_seconds=delay,
            progress_callback=on_progress
        )

    # Summary table
    table = Table(title="Generation Results Summary")
    table.add_column("SKU", style="cyan")
    table.add_column("Product Name", style="white")
    table.add_column("Status", style="bold")
    table.add_column("SEO Score", justify="center")
    table.add_column("Readability", style="magenta", justify="center")
    table.add_column("Attempts", justify="right")
    table.add_column("Title", style="italic")

    for rec in result["records"]:
        st_color = "green" if rec["status"] == "PASS" else ("yellow" if rec["status"] == "RETRY_FAILED" else "red")
        seo_val = f"{rec.get('seo_score', 0)}/100" if rec.get("seo_score") else "N/A"
        read_val = f"{rec.get('readability_score', 0):.0f}" if rec.get("readability_score") else "N/A"
        title_snippet = (rec["description"]["title"] if rec["description"] else (rec["errors"][0] if rec["errors"] else "N/A"))[:40]
        table.add_row(
            rec["sku"],
            rec["name"][:22],
            f"[{st_color}]{rec['status']}[/{st_color}]",
            seo_val,
            read_val,
            str(rec["attempts"]),
            title_snippet
        )

    console.print(table)
    console.print(f"[bold green]Complete:[/bold green] {result['success_count']}/{result['total_processed']} succeeded in {result['wall_clock_seconds']}s")

    # Save outputs
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        serializable_result = {
            **result,
            "successful_descriptions": [d.model_dump() for d in result["successful_descriptions"]]
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_result, f, indent=2)
        console.print(f"[cyan]Results saved to: {output_file}[/cyan]")

    if shopify_out and result["successful_descriptions"]:
        shopify_out.parent.mkdir(parents=True, exist_ok=True)
        extras = [r["extra_attrs"] for r in result["records"] if r["description"] is not None]
        csv_text = to_shopify_csv(result["successful_descriptions"], extras)
        with open(shopify_out, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
        console.print(f"[cyan]Shopify CSV exported to: {shopify_out}[/cyan]")

    if woo_out and result["successful_descriptions"]:
        woo_out.parent.mkdir(parents=True, exist_ok=True)
        extras = [r["extra_attrs"] for r in result["records"] if r["description"] is not None]
        woo_text = to_woocommerce_payload_json(result["successful_descriptions"], extras)
        with open(woo_out, "w", encoding="utf-8") as f:
            f.write(woo_text)
        console.print(f"[cyan]WooCommerce JSON exported to: {woo_out}[/cyan]")


if __name__ == "__main__":
    app()
