"""Tests for the measure_progress script."""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from measure_progress import (
    EXPECTED_BENCHMARKS,
    EXPECTED_METRICS,
    EXPECTED_MODELS,
    MODEL_NAME_MAPPING,
    calculate_progress,
    load_json_with_trailing_commas,
    load_results,
    normalize_model_name,
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
        """Test loading JSON with trailing commas."""
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

    def test_skips_invalid_json(self, tmp_path):
        """Test that invalid JSON files are skipped."""
        model_dir = tmp_path / "202511_invalid-model"
        model_dir.mkdir()

        (model_dir / "metadata.json").write_text("not valid json {{{")
        (model_dir / "scores.json").write_text('[{"benchmark": "swe-bench"}]')

        results = load_results(tmp_path)
        assert len(results["models"]) == 0


class TestExpectedDimensions:
    """Tests for expected dimensions from issue #2."""

    def test_expected_benchmarks_count(self):
        """Test that we have 6 expected benchmarks."""
        assert len(EXPECTED_BENCHMARKS) == 6

    def test_expected_metrics_count(self):
        """Test that we have 3 expected metrics."""
        assert len(EXPECTED_METRICS) == 3

    def test_expected_models_count(self):
        """Test that we have 10 expected models."""
        assert len(EXPECTED_MODELS) == 10

    def test_total_cells(self):
        """Test that total 3D array cells = 6 * 10 * 3 = 180."""
        total = len(EXPECTED_BENCHMARKS) * len(EXPECTED_MODELS) * len(EXPECTED_METRICS)
        assert total == 180


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
        assert progress["model_coverage_pct"] == 0.0
        assert progress["array_coverage_pct"] == 0.0
        assert progress["array_total_cells"] == 180

    def test_known_model_mapping(self):
        """Test progress with a known model that maps to expected."""
        results = {
            "models": {"gpt-5"},
            "benchmarks": {"swe-bench"},
            "metrics": {"resolve_rate"},
            "coverage": {("gpt-5", "swe-bench", "resolve_rate"): True},
        }
        progress = calculate_progress(results)

        # gpt-5 maps to gpt-5.2, so 1/10 models = 10%
        assert progress["model_coverage_pct"] == 10.0
        # 1/6 benchmarks = 16.67%
        assert progress["benchmark_coverage_pct"] == 16.67
        # resolve_rate maps to accuracy, so 1/3 metrics = 33.33%
        assert progress["metric_coverage_pct"] == 33.33
        # 1 cell filled out of 180 = 0.56%
        assert progress["array_filled_cells"] == 1
        assert progress["array_coverage_pct"] == 0.56

    def test_unknown_model_not_counted(self):
        """Test that unknown models do not contribute to coverage."""
        results = {
            "models": {"unknown-model-xyz"},
            "benchmarks": {"swe-bench"},
            "metrics": {"resolve_rate"},
            "coverage": {("unknown-model-xyz", "swe-bench", "resolve_rate"): True},
        }
        progress = calculate_progress(results)

        # Unknown model does not map to any expected model
        assert progress["model_coverage_pct"] == 0.0
        # But benchmark and metric coverage still count
        assert progress["benchmark_coverage_pct"] == 16.67
        assert progress["metric_coverage_pct"] == 33.33
        # No cells filled because model does not map
        assert progress["array_filled_cells"] == 0

    def test_metric_mapping(self):
        """Test that metrics are correctly mapped."""
        results = {
            "models": {"gpt-5"},
            "benchmarks": {"swe-bench"},
            "metrics": {"resolve_rate", "total_cost", "total_runtime"},
            "coverage": {
                ("gpt-5", "swe-bench", "resolve_rate"): True,
                ("gpt-5", "swe-bench", "total_cost"): True,
                ("gpt-5", "swe-bench", "total_runtime"): True,
            },
        }
        progress = calculate_progress(results)

        # All 3 metrics should be covered
        assert progress["metric_coverage_pct"] == 100.0
        assert set(progress["metrics_covered"]) == {"accuracy", "cost", "time"}
        # 3 cells filled (1 model * 1 benchmark * 3 metrics)
        assert progress["array_filled_cells"] == 3


class TestIntegration:
    """Integration tests using the actual results directory structure."""

    def test_actual_results_directory(self):
        """Test loading from the actual results directory."""
        repo_root = Path(__file__).parent.parent
        results_dir = repo_root / "results"

        if not results_dir.exists():
            return  # Skip if results directory does not exist

        results = load_results(results_dir)
        progress = calculate_progress(results)

        # Basic sanity checks
        assert 0 <= progress["model_coverage_pct"] <= 100
        assert 0 <= progress["benchmark_coverage_pct"] <= 100
        assert 0 <= progress["metric_coverage_pct"] <= 100
        assert 0 <= progress["array_coverage_pct"] <= 100
        assert 0 <= progress["overall_progress_pct"] <= 100

        # Verify expected fields exist
        assert "models_found" in progress
        assert "models_expected" in progress
        assert "benchmarks_found" in progress
        assert "benchmarks_expected" in progress
        assert "metrics_found" in progress
        assert "metrics_expected" in progress
        assert progress["array_total_cells"] == 180
