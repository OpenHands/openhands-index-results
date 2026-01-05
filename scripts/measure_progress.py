#!/usr/bin/env python3
"""
Measure progress towards the 3D array goal (benchmarks x models x metrics).

This script analyzes the results directory and calculates progress percentage
based on coverage of benchmarks, models, and metrics.
"""

import json
import sys
from pathlib import Path


def load_json(file_path: Path) -> dict | list:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


# Expected benchmarks from issue #2
# https://github.com/OpenHands/openhands-index-results/issues/2
EXPECTED_BENCHMARKS = [
    "swe-bench",           # SWE-Bench
    "swe-bench-multimodal",  # SWE-Bench multimodal (#6)
    "multi-swe-bench",     # multi-swe-bench (#7)
    "swt-bench",           # SWT-bench (#5)
    "commit0",             # commit-0 (#8)
    "gaia",                # GAIA (#9)
]

# Expected metrics from issue #2
EXPECTED_METRICS = [
    "accuracy",            # accuracy (resolve_rate)
    "total_cost",          # monetary cost (#3)
    # Prefer average_runtime, fall back to total_runtime for legacy results
    "average_runtime",
]

# Expected models from issue #2
EXPECTED_MODELS = [
    "claude-4.5-opus",
    "claude-4.5-sonnet",
    "gemini-3-pro",
    "gemini-3-flash",
    "gpt-5.2-high-reasoning",
    "gpt-5.2",
    "kimi-k2-thinking",
    "minimax-m2",
    "deepseek-v3.2-reasoner",
    "qwen-3-coder",
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
            metadata = load_json(metadata_file)
            scores = load_json(scores_file)
        except (json.JSONDecodeError, IOError):
            continue

        model_name = metadata.get("model", model_dir.name)
        models.add(model_name)

        for score_entry in scores:
            benchmark = score_entry.get("benchmark")
            metric = score_entry.get("metric")
            has_total_cost = "total_cost" in score_entry
            has_average_runtime = "average_runtime" in score_entry
            has_total_runtime = "total_runtime" in score_entry

            if benchmark:
                benchmarks.add(benchmark)
            if metric:
                metrics.add(metric)
                if benchmark:
                    coverage[(model_name, benchmark, metric)] = True

            # Track total_cost and total_runtime as separate metrics
            if has_total_cost:
                metrics.add("total_cost")
                if benchmark:
                    coverage[(model_name, benchmark, "total_cost")] = True
            # Track runtime metrics (prefer average_runtime, include legacy total_runtime)
            if has_average_runtime:
                metrics.add("average_runtime")
                if benchmark:
                    coverage[(model_name, benchmark, "average_runtime")] = True
            if has_total_runtime:
                metrics.add("average_runtime")
                if benchmark:
                    coverage[(model_name, benchmark, "average_runtime")] = True

    return {
        "models": models,
        "benchmarks": benchmarks,
        "metrics": metrics,
        "coverage": coverage,
    }


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for matching against expected models."""
    return model_name.lower().replace("_", "-").replace(" ", "-")


# Mapping from found model names to expected model names
# This handles variations in naming conventions
MODEL_NAME_MAPPING = {
    "claude-opus-4-5-20251101": "claude-4.5-opus",
    "claude-sonnet-4-5-20250929": "claude-4.5-sonnet",
    "gemini-3-pro-preview": "gemini-3-pro",
    "gpt-5": "gpt-5.2",
    "qwen3-coder-480b-a35b-instruct-fp8": "qwen-3-coder",
}


def calculate_progress(results: dict) -> dict:
    """Calculate progress percentage towards the 3D array goal.

    Progress is calculated based on the expected dimensions from issue #2:
    - 6 benchmarks
    - 3 metrics
    - 10 models

    The 3D array has 6 * 3 * 10 = 180 total cells.

    Returns a dict with progress details.
    """
    found_models = results["models"]
    found_benchmarks = results["benchmarks"]
    found_metrics = results["metrics"]
    coverage = results["coverage"]

    # Map found models to expected models
    found_to_expected = {}
    for found_model in found_models:
        normalized = normalize_model_name(found_model)
        if normalized in MODEL_NAME_MAPPING:
            found_to_expected[found_model] = MODEL_NAME_MAPPING[normalized]

    # Model coverage - check which expected models have any results
    covered_models = list(set(found_to_expected.values()))
    model_coverage = len(covered_models) / len(EXPECTED_MODELS) * 100

    # Benchmark coverage
    covered_benchmarks = found_benchmarks.intersection(set(EXPECTED_BENCHMARKS))
    benchmark_coverage = len(covered_benchmarks) / len(EXPECTED_BENCHMARKS) * 100

    # Metric coverage - direct match against expected metrics
    covered_metrics = found_metrics.intersection(set(EXPECTED_METRICS))
    metric_coverage = len(covered_metrics) / len(EXPECTED_METRICS) * 100

    # 3D array coverage (expected models x expected benchmarks x expected metrics)
    # Total cells = 10 models * 6 benchmarks * 3 metrics = 180
    total_expected_cells = (
        len(EXPECTED_MODELS) * len(EXPECTED_BENCHMARKS) * len(EXPECTED_METRICS)
    )

    # Count filled cells by checking coverage
    # Build reverse mapping: expected model -> found model
    expected_to_found = {v: k for k, v in found_to_expected.items()}

    filled_cells = 0
    for expected_model in EXPECTED_MODELS:
        found_model = expected_to_found.get(expected_model)
        if found_model:
            for benchmark in EXPECTED_BENCHMARKS:
                for metric in EXPECTED_METRICS:
                    if coverage.get((found_model, benchmark, metric)):
                        filled_cells += 1

    array_coverage = filled_cells / total_expected_cells * 100 if total_expected_cells else 0

    # Overall progress is the 3D array coverage (the main goal)
    overall_progress = array_coverage

    return {
        "models_found": sorted(found_models),
        "models_expected": EXPECTED_MODELS,
        "models_covered": covered_models,
        "model_coverage_pct": round(model_coverage, 2),
        "benchmarks_found": sorted(found_benchmarks),
        "benchmarks_expected": EXPECTED_BENCHMARKS,
        "benchmarks_covered": sorted(covered_benchmarks),
        "benchmark_coverage_pct": round(benchmark_coverage, 2),
        "metrics_found": sorted(found_metrics),
        "metrics_expected": EXPECTED_METRICS,
        "metrics_covered": sorted(covered_metrics),
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
    print("Target: 3D array of benchmarks × models × metrics")
    print(f"  {len(EXPECTED_BENCHMARKS)} benchmarks × {len(EXPECTED_MODELS)} models × {len(EXPECTED_METRICS)} metrics = {progress['array_total_cells']} cells")
    print()

    print(f"Model Coverage: {progress['model_coverage_pct']}% ({len(progress['models_covered'])}/{len(EXPECTED_MODELS)})")
    print(f"  Expected: {progress['models_expected']}")
    print(f"  Found: {progress['models_found']}")
    print(f"  Covered: {progress['models_covered']}")
    print()

    print(f"Benchmark Coverage: {progress['benchmark_coverage_pct']}% ({len(progress['benchmarks_covered'])}/{len(EXPECTED_BENCHMARKS)})")
    print(f"  Expected: {progress['benchmarks_expected']}")
    print(f"  Found: {progress['benchmarks_found']}")
    print(f"  Covered: {list(progress['benchmarks_covered'])}")
    print()

    print(f"Metric Coverage: {progress['metric_coverage_pct']}% ({len(progress['metrics_covered'])}/{len(EXPECTED_METRICS)})")
    print(f"  Expected: {progress['metrics_expected']}")
    print(f"  Found: {progress['metrics_found']}")
    print(f"  Covered: {list(progress['metrics_covered'])}")
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
