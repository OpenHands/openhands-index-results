#!/usr/bin/env python3
"""
Measure progress towards the 3D array goal (benchmarks x models x metrics).

This script analyzes the results directory and calculates progress percentage
based on coverage of benchmarks, models, and metrics.
"""

import json
import re
import sys
from pathlib import Path


def load_json_with_trailing_commas(file_path: Path) -> dict | list:
    """Load JSON file, handling trailing commas which are technically invalid JSON."""
    with open(file_path) as f:
        content = f.read()
    # Remove trailing commas before ] or }
    content = re.sub(r",\s*([}\]])", r"\1", content)
    return json.loads(content)


# Expected benchmarks from README
EXPECTED_BENCHMARKS = [
    "swe-bench",
    "swe-bench-multimodal",
    "commit0",
    "multi-swe-bench",
    "swt-bench",
    "gaia",
]

# Expected metrics based on scores.json schema
EXPECTED_METRICS = [
    "resolve_rate",
    "success_rate",
]


def load_results(results_dir: Path) -> dict:
    """Load all results from the results directory.

    Returns a dict with structure:
    {
        "models": set of model names,
        "benchmarks": set of benchmark names,
        "metrics": set of metric names,
        "coverage": dict mapping (model, benchmark, metric) -> bool
    }
    """
    models = set()
    benchmarks = set()
    metrics = set()
    coverage = {}

    if not results_dir.exists():
        return {
            "models": models,
            "benchmarks": benchmarks,
            "metrics": metrics,
            "coverage": coverage,
        }

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        metadata_file = model_dir / "metadata.json"
        scores_file = model_dir / "scores.json"

        if not metadata_file.exists() or not scores_file.exists():
            continue

        try:
            metadata = load_json_with_trailing_commas(metadata_file)
            scores = load_json_with_trailing_commas(scores_file)
        except (json.JSONDecodeError, IOError):
            continue

        model_name = metadata.get("model", model_dir.name)
        models.add(model_name)

        for score_entry in scores:
            benchmark = score_entry.get("benchmark")
            metric = score_entry.get("metric")

            if benchmark:
                benchmarks.add(benchmark)
            if metric:
                metrics.add(metric)

            if benchmark and metric:
                coverage[(model_name, benchmark, metric)] = True

    return {
        "models": models,
        "benchmarks": benchmarks,
        "metrics": metrics,
        "coverage": coverage,
    }


def calculate_progress(results: dict) -> dict:
    """Calculate progress percentage towards the 3D array goal.

    Progress is calculated as:
    - Benchmark coverage: % of expected benchmarks that have at least one result
    - Metric coverage: % of expected metrics that have at least one result
    - 3D coverage: % of (model x benchmark x metric) combinations filled

    Returns a dict with progress details.
    """
    models = results["models"]
    benchmarks = results["benchmarks"]
    metrics = results["metrics"]
    coverage = results["coverage"]

    # Benchmark coverage
    covered_benchmarks = benchmarks.intersection(set(EXPECTED_BENCHMARKS))
    benchmark_coverage = (
        len(covered_benchmarks) / len(EXPECTED_BENCHMARKS) * 100
        if EXPECTED_BENCHMARKS
        else 0
    )

    # Metric coverage
    covered_metrics = metrics.intersection(set(EXPECTED_METRICS))
    metric_coverage = (
        len(covered_metrics) / len(EXPECTED_METRICS) * 100 if EXPECTED_METRICS else 0
    )

    # 3D array coverage (models x benchmarks x metrics)
    # Calculate what percentage of the expected 3D array is filled
    if models and EXPECTED_BENCHMARKS and EXPECTED_METRICS:
        total_expected_cells = (
            len(models) * len(EXPECTED_BENCHMARKS) * len(EXPECTED_METRICS)
        )
        filled_cells = 0
        for model in models:
            for benchmark in EXPECTED_BENCHMARKS:
                for metric in EXPECTED_METRICS:
                    if coverage.get((model, benchmark, metric)):
                        filled_cells += 1
        array_coverage = filled_cells / total_expected_cells * 100
    else:
        total_expected_cells = 0
        filled_cells = 0
        array_coverage = 0

    # Overall progress (weighted average)
    overall_progress = (benchmark_coverage + metric_coverage + array_coverage) / 3

    return {
        "models_count": len(models),
        "models": sorted(models),
        "benchmarks_found": sorted(benchmarks),
        "benchmarks_expected": EXPECTED_BENCHMARKS,
        "benchmark_coverage_pct": round(benchmark_coverage, 2),
        "metrics_found": sorted(metrics),
        "metrics_expected": EXPECTED_METRICS,
        "metric_coverage_pct": round(metric_coverage, 2),
        "array_total_cells": total_expected_cells,
        "array_filled_cells": filled_cells,
        "array_coverage_pct": round(array_coverage, 2),
        "overall_progress_pct": round(overall_progress, 2),
    }


def print_progress_report(progress: dict) -> None:
    """Print a formatted progress report."""
    print("=" * 60)
    print("OpenHands Index Results - Progress Report")
    print("=" * 60)
    print()

    print(f"Models found: {progress['models_count']}")
    for model in progress["models"]:
        print(f"  - {model}")
    print()

    print(f"Benchmark Coverage: {progress['benchmark_coverage_pct']}%")
    print(f"  Found: {progress['benchmarks_found']}")
    print(f"  Expected: {progress['benchmarks_expected']}")
    print()

    print(f"Metric Coverage: {progress['metric_coverage_pct']}%")
    print(f"  Found: {progress['metrics_found']}")
    print(f"  Expected: {progress['metrics_expected']}")
    print()

    print(f"3D Array Coverage: {progress['array_coverage_pct']}%")
    print(
        f"  Filled cells: {progress['array_filled_cells']} / {progress['array_total_cells']}"
    )
    print()

    print("=" * 60)
    print(f"OVERALL PROGRESS: {progress['overall_progress_pct']}%")
    print("=" * 60)


def main():
    """Main entry point."""
    # Determine results directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"

    # Allow override via command line argument
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    results = load_results(results_dir)
    progress = calculate_progress(results)
    print_progress_report(progress)

    # Return non-zero exit code if progress is below threshold (optional)
    # This could be used to fail CI if progress regresses
    return 0


if __name__ == "__main__":
    sys.exit(main())
