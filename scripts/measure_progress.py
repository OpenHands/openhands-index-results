#!/usr/bin/env python3
"""
Measure progress towards the 3D array goal (benchmarks x models x metrics).

This script analyzes the results directory and reports missing combinations
of models, benchmarks, and metrics.
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
    "swt-bench",           # SWT-bench (#5)
    "commit0",             # commit-0 (#8)
    "gaia",                # GAIA (#9)
]

# Expected metrics from issue #2
# Note: "score" represents any primary score metric (accuracy, solveable_accuracy, etc.)
# The actual metric name is stored in the "metric" field of each score entry
EXPECTED_METRICS = [
    "score",               # primary score (accuracy, solveable_accuracy, etc.)
    "cost_per_instance",   # monetary cost per problem (#3)
    "average_runtime",     # wall clock time (#4)
]

# Expected models from issue #2
EXPECTED_MODELS = [
    "claude-4.5-opus",
    "claude-4.5-sonnet",
    "gemini-3-pro",
    "gemini-3-flash",
    "gpt-5.2",
    "gpt-5.2-codex",
    "kimi-k2-thinking",
    "minimax-m2.1",
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
            has_score = "score" in score_entry and "metric" in score_entry
            has_cost_per_instance = "cost_per_instance" in score_entry
            has_average_runtime = "average_runtime" in score_entry

            if benchmark:
                benchmarks.add(benchmark)

            # Track "score" if both score and metric fields exist
            if has_score:
                metrics.add("score")
                if benchmark:
                    coverage[(model_name, benchmark, "score")] = True

            # Track cost_per_instance and average_runtime as separate metrics
            if has_cost_per_instance:
                metrics.add("cost_per_instance")
                if benchmark:
                    coverage[(model_name, benchmark, "cost_per_instance")] = True
            if has_average_runtime:
                metrics.add("average_runtime")
                if benchmark:
                    coverage[(model_name, benchmark, "average_runtime")] = True

    return {
        "models": models,
        "benchmarks": benchmarks,
        "metrics": metrics,
        "coverage": coverage,
    }


def calculate_progress(results: dict) -> dict:
    """Calculate progress metrics for the 3D array coverage.

    Returns a dict with progress percentages and counts.
    """
    models_found = results["models"] & set(EXPECTED_MODELS)
    benchmarks_found = results["benchmarks"] & set(EXPECTED_BENCHMARKS)
    metrics_found = results["metrics"] & set(EXPECTED_METRICS)
    coverage = results["coverage"]

    # Calculate coverage percentages
    model_coverage_pct = round(len(models_found) / len(EXPECTED_MODELS) * 100, 2) if EXPECTED_MODELS else 0.0
    benchmark_coverage_pct = round(len(benchmarks_found) / len(EXPECTED_BENCHMARKS) * 100, 2) if EXPECTED_BENCHMARKS else 0.0
    metric_coverage_pct = round(len(metrics_found) / len(EXPECTED_METRICS) * 100, 2) if EXPECTED_METRICS else 0.0

    # Calculate 3D array coverage
    array_total_cells = len(EXPECTED_MODELS) * len(EXPECTED_BENCHMARKS) * len(EXPECTED_METRICS)
    array_filled_cells = 0
    for model in EXPECTED_MODELS:
        for benchmark in EXPECTED_BENCHMARKS:
            for metric in EXPECTED_METRICS:
                if coverage.get((model, benchmark, metric)):
                    array_filled_cells += 1

    array_coverage_pct = round(array_filled_cells / array_total_cells * 100, 2) if array_total_cells else 0.0

    # Overall progress is the array coverage
    overall_progress_pct = array_coverage_pct

    return {
        "overall_progress_pct": overall_progress_pct,
        "model_coverage_pct": model_coverage_pct,
        "benchmark_coverage_pct": benchmark_coverage_pct,
        "metric_coverage_pct": metric_coverage_pct,
        "array_coverage_pct": array_coverage_pct,
        "array_filled_cells": array_filled_cells,
        "array_total_cells": array_total_cells,
        "models_found": len(models_found),
        "models_expected": len(EXPECTED_MODELS),
        "benchmarks_found": len(benchmarks_found),
        "benchmarks_expected": len(EXPECTED_BENCHMARKS),
        "metrics_found": len(metrics_found),
        "metrics_expected": len(EXPECTED_METRICS),
    }


def find_missing_combinations(results: dict) -> dict:
    """Find missing model+benchmark pairs.

    A model+benchmark pair is only considered complete if ALL metrics are present.

    Returns a dict with:
    - missing_pairs: dict mapping model -> list of (benchmark, missing_metrics) tuples
    - total_pairs: total expected model+benchmark pairs
    - complete_pairs: number of pairs with all metrics present
    """
    coverage = results["coverage"]

    missing_pairs = {}
    complete_pairs = 0
    
    for model in EXPECTED_MODELS:
        for benchmark in EXPECTED_BENCHMARKS:
            # Check which metrics are missing for this model+benchmark pair
            missing_metrics = []
            for metric in EXPECTED_METRICS:
                if not coverage.get((model, benchmark, metric)):
                    missing_metrics.append(metric)
            
            if not missing_metrics:
                # All metrics present - pair is complete
                complete_pairs += 1
            else:
                # Some metrics missing - track what's missing
                if model not in missing_pairs:
                    missing_pairs[model] = []
                missing_pairs[model].append((benchmark, missing_metrics))

    total_pairs = len(EXPECTED_MODELS) * len(EXPECTED_BENCHMARKS)

    return {
        "missing_pairs": missing_pairs,
        "total_pairs": total_pairs,
        "complete_pairs": complete_pairs,
    }


def generate_progress_bar(percentage: float, width: int = 11) -> str:
    """Generate an ASCII progress bar using block characters."""
    filled = int(round(percentage / 100 * width))
    empty = width - filled
    bar = "⬛" * filled + "⬜" * empty
    return f"{bar} {percentage}%"


def print_progress_report(missing: dict) -> None:
    """Print a formatted progress report showing missing model+benchmark pairs."""
    total = missing["total_pairs"]
    complete = missing["complete_pairs"]
    progress_pct = round(complete / total * 100, 2) if total else 0

    print("=" * 60)
    print("OpenHands Index Results - Progress Report")
    print("=" * 60)
    print()
    print("Target: Complete all model × benchmark pairs")
    print(f"  {len(EXPECTED_MODELS)} models × {len(EXPECTED_BENCHMARKS)} benchmarks = {total} pairs")
    print(f"  (each pair requires all {len(EXPECTED_METRICS)} metrics: {', '.join(EXPECTED_METRICS)})")
    print()

    # Missing pairs - grouped by model
    if missing["missing_pairs"]:
        incomplete_count = total - complete
        print(f"Incomplete Pairs ({incomplete_count}):")
        
        for model in EXPECTED_MODELS:
            if model in missing["missing_pairs"]:
                print(f"  {model}:")
                for benchmark, missing_metrics in missing["missing_pairs"][model]:
                    if set(missing_metrics) == set(EXPECTED_METRICS):
                        print(f"    - {benchmark} (all metrics)")
                    else:
                        print(f"    - {benchmark} ({', '.join(missing_metrics)})")
        print()

    print("=" * 60)
    progress_bar = generate_progress_bar(progress_pct)
    print(f"OVERALL PROGRESS: {progress_bar}")
    print(f"  Complete: {complete} / {total} pairs")
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
    missing = find_missing_combinations(results)
    print_progress_report(missing)

    return 0


if __name__ == "__main__":
    sys.exit(main())
