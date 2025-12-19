"""Tests for the measure_progress script."""

import json
import sys
import tempfile
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from measure_progress import (
    EXPECTED_BENCHMARKS,
    EXPECTED_METRICS,
    calculate_progress,
    load_json_with_trailing_commas,
    load_results,
)


class TestLoadJsonWithTrailingCommas:
    """Tests for load_json_with_trailing_commas function."""

    def test_valid_json(self, tmp_path):
        """Test loading valid JSON without trailing commas."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')
        result = load_json_with_trailing_commas(json_file)
        assert result == {"key": "value"}

    def test_json_with_trailing_comma_in_object(self, tmp_path):
        """Test loading JSON with trailing comma in object."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value",}')
        result = load_json_with_trailing_commas(json_file)
        assert result == {"key": "value"}

    def test_json_with_trailing_comma_in_array(self, tmp_path):
        """Test loading JSON with trailing comma in array."""
        json_file = tmp_path / "test.json"
        json_file.write_text('[1, 2, 3,]')
        result = load_json_with_trailing_commas(json_file)
        assert result == [1, 2, 3]

    def test_json_with_multiple_trailing_commas(self, tmp_path):
        """Test loading JSON with multiple trailing commas."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"items": [1, 2,], "nested": {"a": 1,},}')
        result = load_json_with_trailing_commas(json_file)
        assert result == {"items": [1, 2], "nested": {"a": 1}}


class TestLoadResults:
    """Tests for load_results function."""

    def test_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        results = load_results(tmp_path)
        assert results["models"] == set()
        assert results["benchmarks"] == set()
        assert results["metrics"] == set()
        assert results["coverage"] == {}

    def test_nonexistent_directory(self, tmp_path):
        """Test loading from nonexistent directory."""
        results = load_results(tmp_path / "nonexistent")
        assert results["models"] == set()
        assert results["benchmarks"] == set()
        assert results["metrics"] == set()
        assert results["coverage"] == {}

    def test_single_model_single_benchmark(self, tmp_path):
        """Test loading single model with single benchmark."""
        model_dir = tmp_path / "202511_test-model"
        model_dir.mkdir()

        metadata = {"model": "test-model", "agent_name": "Test Agent"}
        scores = [{"benchmark": "swe-bench", "score": 50.0, "metric": "resolve_rate"}]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        results = load_results(tmp_path)
        assert "test-model" in results["models"]
        assert "swe-bench" in results["benchmarks"]
        assert "resolve_rate" in results["metrics"]
        assert results["coverage"][("test-model", "swe-bench", "resolve_rate")] is True

    def test_multiple_models(self, tmp_path):
        """Test loading multiple models."""
        for i, model_name in enumerate(["model-a", "model-b", "model-c"]):
            model_dir = tmp_path / f"20251{i}_{model_name}"
            model_dir.mkdir()

            metadata = {"model": model_name}
            scores = [
                {"benchmark": "swe-bench", "score": 50.0 + i, "metric": "resolve_rate"}
            ]

            (model_dir / "metadata.json").write_text(json.dumps(metadata))
            (model_dir / "scores.json").write_text(json.dumps(scores))

        results = load_results(tmp_path)
        assert len(results["models"]) == 3
        assert "model-a" in results["models"]
        assert "model-b" in results["models"]
        assert "model-c" in results["models"]

    def test_multiple_benchmarks_per_model(self, tmp_path):
        """Test loading model with multiple benchmarks."""
        model_dir = tmp_path / "202511_multi-bench-model"
        model_dir.mkdir()

        metadata = {"model": "multi-bench-model"}
        scores = [
            {"benchmark": "swe-bench", "score": 50.0, "metric": "resolve_rate"},
            {"benchmark": "commit0", "score": 60.0, "metric": "success_rate"},
            {"benchmark": "gaia", "score": 70.0, "metric": "resolve_rate"},
        ]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        results = load_results(tmp_path)
        assert len(results["benchmarks"]) == 3
        assert "swe-bench" in results["benchmarks"]
        assert "commit0" in results["benchmarks"]
        assert "gaia" in results["benchmarks"]

    def test_skips_invalid_json(self, tmp_path):
        """Test that invalid JSON files are skipped."""
        model_dir = tmp_path / "202511_invalid-model"
        model_dir.mkdir()

        (model_dir / "metadata.json").write_text("not valid json {{{")
        (model_dir / "scores.json").write_text('[{"benchmark": "swe-bench"}]')

        results = load_results(tmp_path)
        assert len(results["models"]) == 0

    def test_skips_missing_files(self, tmp_path):
        """Test that directories with missing files are skipped."""
        model_dir = tmp_path / "202511_incomplete-model"
        model_dir.mkdir()

        # Only create metadata, no scores
        (model_dir / "metadata.json").write_text('{"model": "incomplete"}')

        results = load_results(tmp_path)
        assert len(results["models"]) == 0


class TestCalculateProgress:
    """Tests for calculate_progress function."""

    def test_empty_results(self):
        """Test progress calculation with empty results."""
        results = {
            "models": set(),
            "benchmarks": set(),
            "metrics": set(),
            "coverage": {},
        }
        progress = calculate_progress(results)
        assert progress["overall_progress_pct"] == 0.0
        assert progress["benchmark_coverage_pct"] == 0.0
        assert progress["metric_coverage_pct"] == 0.0
        assert progress["array_coverage_pct"] == 0

    def test_single_benchmark_single_metric(self):
        """Test progress with one benchmark and one metric."""
        results = {
            "models": {"model-a"},
            "benchmarks": {"swe-bench"},
            "metrics": {"resolve_rate"},
            "coverage": {("model-a", "swe-bench", "resolve_rate"): True},
        }
        progress = calculate_progress(results)

        # 1/6 benchmarks = 16.67%
        assert progress["benchmark_coverage_pct"] == 16.67
        # 1/2 metrics = 50%
        assert progress["metric_coverage_pct"] == 50.0
        # 1 model * 6 benchmarks * 2 metrics = 12 cells, 1 filled = 8.33%
        assert progress["array_coverage_pct"] == 8.33
        # Average of 16.67, 50, 8.33 = 25%
        assert progress["overall_progress_pct"] == 25.0

    def test_full_coverage(self):
        """Test progress with full coverage."""
        models = {"model-a", "model-b"}
        coverage = {}
        for model in models:
            for benchmark in EXPECTED_BENCHMARKS:
                for metric in EXPECTED_METRICS:
                    coverage[(model, benchmark, metric)] = True

        results = {
            "models": models,
            "benchmarks": set(EXPECTED_BENCHMARKS),
            "metrics": set(EXPECTED_METRICS),
            "coverage": coverage,
        }
        progress = calculate_progress(results)

        assert progress["benchmark_coverage_pct"] == 100.0
        assert progress["metric_coverage_pct"] == 100.0
        assert progress["array_coverage_pct"] == 100.0
        assert progress["overall_progress_pct"] == 100.0

    def test_partial_coverage(self):
        """Test progress with partial coverage."""
        models = {"model-a", "model-b"}
        coverage = {
            ("model-a", "swe-bench", "resolve_rate"): True,
            ("model-a", "commit0", "resolve_rate"): True,
            ("model-b", "swe-bench", "resolve_rate"): True,
        }

        results = {
            "models": models,
            "benchmarks": {"swe-bench", "commit0"},
            "metrics": {"resolve_rate"},
            "coverage": coverage,
        }
        progress = calculate_progress(results)

        # 2/6 benchmarks = 33.33%
        assert progress["benchmark_coverage_pct"] == 33.33
        # 1/2 metrics = 50%
        assert progress["metric_coverage_pct"] == 50.0
        # 2 models * 6 benchmarks * 2 metrics = 24 cells, 3 filled = 12.5%
        assert progress["array_coverage_pct"] == 12.5

    def test_models_count(self):
        """Test that models count is correct."""
        results = {
            "models": {"model-a", "model-b", "model-c"},
            "benchmarks": {"swe-bench"},
            "metrics": {"resolve_rate"},
            "coverage": {},
        }
        progress = calculate_progress(results)
        assert progress["models_count"] == 3

    def test_models_sorted(self):
        """Test that models list is sorted."""
        results = {
            "models": {"zebra-model", "alpha-model", "beta-model"},
            "benchmarks": set(),
            "metrics": set(),
            "coverage": {},
        }
        progress = calculate_progress(results)
        assert progress["models"] == ["alpha-model", "beta-model", "zebra-model"]


class TestIntegration:
    """Integration tests using the actual results directory structure."""

    def test_actual_results_directory(self):
        """Test loading from the actual results directory."""
        repo_root = Path(__file__).parent.parent
        results_dir = repo_root / "results"

        if not results_dir.exists():
            return  # Skip if results directory doesn't exist

        results = load_results(results_dir)
        progress = calculate_progress(results)

        # Basic sanity checks
        assert progress["models_count"] >= 0
        assert 0 <= progress["benchmark_coverage_pct"] <= 100
        assert 0 <= progress["metric_coverage_pct"] <= 100
        assert 0 <= progress["array_coverage_pct"] <= 100
        assert 0 <= progress["overall_progress_pct"] <= 100

        # Verify expected fields exist
        assert "benchmarks_found" in progress
        assert "benchmarks_expected" in progress
        assert "metrics_found" in progress
        assert "metrics_expected" in progress
