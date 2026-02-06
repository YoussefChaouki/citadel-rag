#!/usr/bin/env python3
"""
CITADEL RAG Evaluation Harness v2.0

Comprehensive evaluation script with:
- Hit Rate @ multiple k values
- MRR (Mean Reciprocal Rank)
- Breakdown by category and difficulty
- Negative test support
- Markdown report generation
- Rich console output

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --k 10
    python scripts/evaluate_rag.py --output reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Rich Console (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None  # type: ignore
    RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8001"
DEFAULT_DATASET = Path("tests/data/golden_dataset.json")
DEFAULT_K = 5
TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Single search result from API."""

    source: str
    content: str
    score: float
    chunk_index: int


@dataclass
class QueryResult:
    """Evaluation result for a single query."""

    query_id: str
    query: str
    category: str
    difficulty: str
    expected_source: str
    expected_text: str

    # Retrieved results
    results: list[SearchResult] = field(default_factory=list)

    # Evaluation
    hit: bool = False
    rank: int | None = None  # 1-indexed, None if not found
    top_score: float = 0.0
    error: str | None = None

    @property
    def is_negative(self) -> bool:
        return self.expected_source.upper() == "NONE"

    @property
    def reciprocal_rank(self) -> float:
        if self.rank is None:
            return 0.0
        return 1.0 / self.rank

    @property
    def success(self) -> bool:
        """For positive: hit found. For negative: no relevant hit."""
        if self.is_negative:
            return not self.hit
        return self.hit


@dataclass
class CategoryStats:
    """Statistics for a category."""

    total: int = 0
    hits: int = 0
    rr_sum: float = 0.0

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total * 100

    @property
    def mrr(self) -> float:
        if self.total == 0:
            return 0.0
        return self.rr_sum / self.total


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    total: int = 0
    positive_total: int = 0
    negative_total: int = 0
    negative_correct: int = 0
    errors: int = 0

    # Hit counts at different k
    hits_at: dict[int, int] = field(default_factory=lambda: {1: 0, 3: 0, 5: 0, 10: 0})
    rr_sum: float = 0.0

    # Breakdowns
    by_category: dict[str, CategoryStats] = field(default_factory=dict)
    by_difficulty: dict[str, CategoryStats] = field(default_factory=dict)

    def hit_rate(self, k: int) -> float:
        if self.positive_total == 0:
            return 0.0
        return self.hits_at.get(k, 0) / self.positive_total * 100

    @property
    def mrr(self) -> float:
        if self.positive_total == 0:
            return 0.0
        return self.rr_sum / self.positive_total

    @property
    def negative_accuracy(self) -> float:
        if self.negative_total == 0:
            return 100.0
        return self.negative_correct / self.negative_total * 100


# ---------------------------------------------------------------------------
# Console Helpers
# ---------------------------------------------------------------------------


def log_info(msg: str) -> None:
    if RICH_AVAILABLE and console:
        console.print(f"[blue]ℹ[/blue] {msg}")
    else:
        print(f"ℹ {msg}")


def log_success(msg: str) -> None:
    if RICH_AVAILABLE and console:
        console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"✓ {msg}")


def log_error(msg: str) -> None:
    if RICH_AVAILABLE and console:
        console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"✗ {msg}")


def log_warning(msg: str) -> None:
    if RICH_AVAILABLE and console:
        console.print(f"[yellow]⚠[/yellow] {msg}")
    else:
        print(f"⚠ {msg}")


# ---------------------------------------------------------------------------
# API Functions
# ---------------------------------------------------------------------------


def check_health(api_url: str) -> bool:
    """Check if API is reachable."""
    try:
        r = httpx.get(f"{api_url}/health", timeout=5.0)
        return r.status_code == 200
    except httpx.RequestError:
        return False


def search(api_url: str, query: str, k: int) -> tuple[list[SearchResult], str | None]:
    """Execute search query."""
    try:
        r = httpx.post(
            f"{api_url}/api/v1/rag/search",
            json={"query": query, "k": k},
            timeout=TIMEOUT,
        )
        r.raise_for_status()

        results = []
        for item in r.json():
            results.append(
                SearchResult(
                    source=item.get("source", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    chunk_index=item.get("chunk_index", 0),
                )
            )
        return results, None

    except httpx.HTTPStatusError as e:
        return [], f"HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        return [], str(e)


# ---------------------------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------------------------


def evaluate_query(api_url: str, entry: dict[str, Any], k: int) -> QueryResult:
    """Evaluate a single query."""
    qr = QueryResult(
        query_id=entry.get("id", "?"),
        query=entry["query"],
        category=entry.get("category", "unknown"),
        difficulty=entry.get("difficulty", "unknown"),
        expected_source=entry["expected_source"],
        expected_text=entry.get("expected_text_content", ""),
    )

    results, error = search(api_url, qr.query, k)
    if error:
        qr.error = error
        return qr

    qr.results = results
    if results:
        qr.top_score = results[0].score

    # Check for hit
    expected_src = qr.expected_source.lower()
    expected_txt = qr.expected_text.lower()

    for i, res in enumerate(results, 1):
        src_lower = res.source.lower()
        content_lower = res.content.lower()

        # Match source name OR expected text content
        if expected_src != "none":
            source_match = expected_src in src_lower
            text_match = expected_txt and expected_txt in content_lower

            if source_match or text_match:
                qr.hit = True
                if qr.rank is None:
                    qr.rank = i
                break
        else:
            # Negative test: check if we incorrectly find relevant content
            if expected_txt and expected_txt in content_lower:
                qr.hit = True
                break

    return qr


def run_evaluation(
    api_url: str, dataset_path: Path, k: int
) -> tuple[list[QueryResult], EvalMetrics]:
    """Run full evaluation."""
    # Load dataset
    with dataset_path.open() as f:
        dataset = json.load(f)

    entries = dataset.get("entries", [])
    if not entries:
        raise ValueError("No entries in dataset")

    results: list[QueryResult] = []
    metrics = EvalMetrics()

    # Progress
    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating queries...", total=len(entries))

            for entry in entries:
                qr = evaluate_query(api_url, entry, k)
                results.append(qr)
                progress.advance(task)
    else:
        for i, entry in enumerate(entries, 1):
            print(f"\rEvaluating {i}/{len(entries)}...", end="")
            qr = evaluate_query(api_url, entry, k)
            results.append(qr)
        print()

    # Aggregate metrics
    for qr in results:
        metrics.total += 1

        if qr.error:
            metrics.errors += 1
            continue

        # Initialize category/difficulty stats
        if qr.category not in metrics.by_category:
            metrics.by_category[qr.category] = CategoryStats()
        if qr.difficulty not in metrics.by_difficulty:
            metrics.by_difficulty[qr.difficulty] = CategoryStats()

        if qr.is_negative:
            metrics.negative_total += 1
            if qr.success:
                metrics.negative_correct += 1
        else:
            metrics.positive_total += 1

            cat_stats = metrics.by_category[qr.category]
            diff_stats = metrics.by_difficulty[qr.difficulty]

            cat_stats.total += 1
            diff_stats.total += 1

            if qr.hit and qr.rank:
                # Update hit counts at various k
                for k_val in [1, 3, 5, 10]:
                    if qr.rank <= k_val:
                        metrics.hits_at[k_val] += 1

                metrics.rr_sum += qr.reciprocal_rank
                cat_stats.hits += 1
                cat_stats.rr_sum += qr.reciprocal_rank
                diff_stats.hits += 1
                diff_stats.rr_sum += qr.reciprocal_rank

    return results, metrics


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def print_console_report(
    results: list[QueryResult], metrics: EvalMetrics, k: int
) -> None:
    """Print rich console report."""
    if RICH_AVAILABLE and console:
        console.print()
        console.print(Panel("📊 [bold]RAG EVALUATION REPORT[/bold]", style="blue"))

        # Summary table
        summary = Table(title="Overall Metrics", box=box.ROUNDED)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")

        summary.add_row("Total Queries", str(metrics.total))
        summary.add_row("Positive Tests", str(metrics.positive_total))
        summary.add_row("Negative Tests", str(metrics.negative_total))
        summary.add_row("Errors", str(metrics.errors))
        summary.add_row("", "")

        # Hit rates
        for k_val in [1, 3, 5, 10]:
            hr = metrics.hit_rate(k_val)
            color = "green" if hr >= 80 else "yellow" if hr >= 60 else "red"
            summary.add_row(f"Hit Rate @{k_val}", f"[{color}]{hr:.1f}%[/{color}]")

        summary.add_row("", "")
        summary.add_row("[bold]MRR[/bold]", f"[bold]{metrics.mrr:.4f}[/bold]")
        summary.add_row("Negative Accuracy", f"{metrics.negative_accuracy:.1f}%")

        console.print(summary)

        # Category breakdown
        if metrics.by_category:
            console.print()
            cat_table = Table(title="📂 By Category", box=box.ROUNDED)
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Queries", justify="right")
            cat_table.add_column("Hit Rate", justify="right")
            cat_table.add_column("MRR", justify="right")

            for cat, stats in sorted(metrics.by_category.items()):
                if cat == "negative":
                    continue
                hr = stats.hit_rate
                color = "green" if hr >= 80 else "yellow" if hr >= 60 else "red"
                cat_table.add_row(
                    cat,
                    str(stats.total),
                    f"[{color}]{hr:.1f}%[/{color}]",
                    f"{stats.mrr:.4f}",
                )

            console.print(cat_table)

        # Difficulty breakdown
        if metrics.by_difficulty:
            console.print()
            diff_table = Table(title="🎯 By Difficulty", box=box.ROUNDED)
            diff_table.add_column("Difficulty", style="cyan")
            diff_table.add_column("Queries", justify="right")
            diff_table.add_column("Hit Rate", justify="right")
            diff_table.add_column("MRR", justify="right")

            for diff in ["easy", "medium", "hard"]:
                if diff in metrics.by_difficulty:
                    stats = metrics.by_difficulty[diff]
                    hr = stats.hit_rate
                    color = "green" if hr >= 80 else "yellow" if hr >= 60 else "red"
                    diff_table.add_row(
                        diff.capitalize(),
                        str(stats.total),
                        f"[{color}]{hr:.1f}%[/{color}]",
                        f"{stats.mrr:.4f}",
                    )

            console.print(diff_table)

        # Failed queries
        failed = [
            r for r in results if not r.success and not r.is_negative and not r.error
        ]
        if failed:
            console.print()
            console.print("[bold red]❌ Failed Queries:[/bold red]")
            for qr in failed[:5]:
                console.print(f"  • [{qr.query_id}] {qr.query[:60]}...")
                sources = [r.source for r in qr.results[:3]]
                console.print(
                    f"    Expected: [cyan]{qr.expected_source}[/cyan] | Got: {sources}"
                )

    else:
        # Plain text
        print(f"\n{'=' * 60}")
        print("RAG EVALUATION REPORT")
        print(f"{'=' * 60}")
        print(
            f"Total: {metrics.total} | Positive: {metrics.positive_total} | Negative: {metrics.negative_total}"
        )
        print(f"Hit Rate @5: {metrics.hit_rate(5):.1f}%")
        print(f"MRR: {metrics.mrr:.4f}")


def generate_markdown_report(
    results: list[QueryResult], metrics: EvalMetrics, k: int, output_path: Path
) -> None:
    """Generate Markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# CITADEL RAG Evaluation Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Retrieval k:** {k}",
        f"**Total Queries:** {metrics.total}",
        "",
        "---",
        "",
        "## 📊 Executive Summary",
        "",
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
    ]

    # Summary metrics
    hr5 = metrics.hit_rate(5)
    hr_emoji = "🟢" if hr5 >= 80 else "🟡" if hr5 >= 60 else "🔴"
    lines.append(f"| **Hit Rate @5** | {hr5:.1f}% | {hr_emoji} |")

    mrr_emoji = "🟢" if metrics.mrr >= 0.7 else "🟡" if metrics.mrr >= 0.5 else "🔴"
    lines.append(f"| **MRR** | {metrics.mrr:.4f} | {mrr_emoji} |")

    lines.append(f"| Negative Accuracy | {metrics.negative_accuracy:.1f}% | - |")
    lines.append(f"| Errors | {metrics.errors} | - |")

    # Hit rate at k
    lines.extend(
        [
            "",
            "## 📈 Hit Rate @ k",
            "",
            "| k | Hit Rate | Found |",
            "|---|----------|-------|",
        ]
    )

    for k_val in [1, 3, 5, 10]:
        hr = metrics.hit_rate(k_val)
        hits = metrics.hits_at.get(k_val, 0)
        lines.append(f"| {k_val} | {hr:.1f}% | {hits}/{metrics.positive_total} |")

    # Category breakdown
    if metrics.by_category:
        lines.extend(
            [
                "",
                "## 📂 Performance by Category",
                "",
                "| Category | Queries | Hit Rate | MRR |",
                "|----------|---------|----------|-----|",
            ]
        )

        for cat, stats in sorted(metrics.by_category.items()):
            if cat == "negative":
                continue
            lines.append(
                f"| {cat} | {stats.total} | {stats.hit_rate:.1f}% | {stats.mrr:.4f} |"
            )

    # Difficulty breakdown
    if metrics.by_difficulty:
        lines.extend(
            [
                "",
                "## 🎯 Performance by Difficulty",
                "",
                "| Difficulty | Queries | Hit Rate | MRR |",
                "|------------|---------|----------|-----|",
            ]
        )

        for diff in ["easy", "medium", "hard"]:
            if diff in metrics.by_difficulty:
                stats = metrics.by_difficulty[diff]
                lines.append(
                    f"| {diff.capitalize()} | {stats.total} | {stats.hit_rate:.1f}% | {stats.mrr:.4f} |"
                )

    # Failed queries
    failed = [r for r in results if not r.success and not r.is_negative and not r.error]
    if failed:
        lines.extend(
            [
                "",
                "## ❌ Failed Queries",
                "",
            ]
        )
        for qr in failed:
            sources = [r.source for r in qr.results[:3]]
            lines.append(f"- **[{qr.query_id}]** {qr.query}")
            lines.append(f"  - Expected: `{qr.expected_source}`")
            lines.append(f"  - Got: `{sources}`")
            lines.append(f"  - Top Score: {qr.top_score:.4f}")
            lines.append("")

    # Methodology
    lines.extend(
        [
            "",
            "---",
            "",
            "## 📖 Methodology",
            "",
            "### Metrics",
            "",
            "- **Hit Rate @k**: % of queries where relevant doc is in top-k",
            "- **MRR**: Mean Reciprocal Rank (1/position of first hit)",
            "- **Negative Accuracy**: % of negative tests correctly rejected",
            "",
            "### Evaluation Process",
            "",
            "1. Load golden dataset with labeled queries",
            "2. Execute each query against RAG API",
            "3. Check if expected source/content appears in results",
            "4. Aggregate metrics by category and difficulty",
            "",
            "---",
            "",
            "*Generated by CITADEL Evaluation Harness v2.0*",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log_success(f"Markdown report: {output_path}")


def save_json_results(
    results: list[QueryResult], metrics: EvalMetrics, output_path: Path
) -> None:
    """Save detailed JSON results."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "metrics": {
            "total": metrics.total,
            "positive_total": metrics.positive_total,
            "negative_total": metrics.negative_total,
            "negative_correct": metrics.negative_correct,
            "errors": metrics.errors,
            "hit_rate_at_1": metrics.hit_rate(1),
            "hit_rate_at_3": metrics.hit_rate(3),
            "hit_rate_at_5": metrics.hit_rate(5),
            "hit_rate_at_10": metrics.hit_rate(10),
            "mrr": metrics.mrr,
            "negative_accuracy": metrics.negative_accuracy,
            "by_category": {
                k: {
                    "total": v.total,
                    "hits": v.hits,
                    "hit_rate": v.hit_rate,
                    "mrr": v.mrr,
                }
                for k, v in metrics.by_category.items()
            },
            "by_difficulty": {
                k: {
                    "total": v.total,
                    "hits": v.hits,
                    "hit_rate": v.hit_rate,
                    "mrr": v.mrr,
                }
                for k, v in metrics.by_difficulty.items()
            },
        },
        "results": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "category": r.category,
                "difficulty": r.difficulty,
                "expected_source": r.expected_source,
                "hit": r.hit,
                "rank": r.rank,
                "top_score": r.top_score,
                "success": r.success,
                "sources": [res.source for res in r.results],
                "scores": [res.score for res in r.results],
                "error": r.error,
            }
            for r in results
        ],
    }

    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_success(f"JSON results: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="CITADEL RAG Evaluation Harness v2.0")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Golden dataset path"
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument("--no-report", action="store_true", help="Skip markdown report")

    args = parser.parse_args()

    if RICH_AVAILABLE and console:
        console.print(
            Panel("🏰 [bold]CITADEL RAG Evaluation v2.0[/bold]", style="blue")
        )
    else:
        print("\n=== CITADEL RAG Evaluation v2.0 ===\n")

    log_info(f"API: {args.api_url}")
    log_info(f"Dataset: {args.dataset}")
    log_info(f"k: {args.k}")
    print()

    # Health check
    if not check_health(args.api_url):
        log_error(f"Cannot reach API at {args.api_url}")
        return 1
    log_success("API connected")

    # Check dataset
    if not args.dataset.exists():
        log_error(f"Dataset not found: {args.dataset}")
        return 1
    log_success("Dataset loaded")
    print()

    # Run evaluation
    try:
        results, metrics = run_evaluation(args.api_url, args.dataset, args.k)
    except Exception as e:
        log_error(f"Evaluation failed: {e}")
        return 1

    # Console report
    print_console_report(results, metrics, args.k)

    # Save outputs
    print()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output.mkdir(parents=True, exist_ok=True)

    json_path = args.output / f"evaluation_results_{timestamp}.json"
    save_json_results(results, metrics, json_path)

    if not args.no_report:
        md_path = args.output / f"evaluation_report_{timestamp}.md"
        generate_markdown_report(results, metrics, args.k, md_path)

    # Exit code based on performance
    if metrics.hit_rate(5) >= 70 and metrics.mrr >= 0.5:
        log_success("✅ Evaluation PASSED")
        return 0
    else:
        log_warning("⚠️ Performance below threshold (HR@5 >= 70%, MRR >= 0.5)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
