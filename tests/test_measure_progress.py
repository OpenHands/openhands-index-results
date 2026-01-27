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
    calculate_progress,
    load_json,
    load_results,
)


class TestLoadJson:
    """Tests for load_json function."""

    def test_valid_json(self, tmp_path):
        """Test loading valid JSON."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')
        result = load_json(json_file)
        assert result == {"key": "value"}

    def test_valid_array(self, tmp_path):
        """Test loading valid JSON array."""
        json_file = tmp_path / "test.json"
        json_file.write_text('[1, 2, 3]')
        result = load_json(json_file)
        assert result == [1, 2, 3]

    def test_nested_json(self, tmp_path):
        """Test loading nested JSON."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"items": [1, 2], "nested": {"a": 1}}')
        result = load_json(json_file)
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
        model_dir = tmp_path / "v1.0.0_test-model"
        model_dir.mkdir()

        metadata = {"model": "test-model", "agent_name": "Test Agent"}
        scores = [{"benchmark": "swe-bench", "score": 50.0, "metric": "accuracy", "cost_per_instance": 0.2, "average_runtime": 3600}]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        results = load_results(tmp_path)
        assert "test-model" in results["models"]
        assert "swe-bench" in results["benchmarks"]
        assert "score" in results["metrics"]
        assert "cost_per_instance" in results["metrics"]
        assert "average_runtime" in results["metrics"]
        assert results["coverage"][("test-model", "swe-bench", "score")] is True
        assert results["coverage"][("test-model", "swe-bench", "cost_per_instance")] is True
        assert results["coverage"][("test-model", "swe-bench", "average_runtime")] is True

    def test_skips_invalid_json(self, tmp_path):
        """Test that invalid JSON files are skipped."""
        model_dir = tmp_path / "v1.0.0_invalid-model"
        model_dir.mkdir()

        (model_dir / "metadata.json").write_text("not valid json {{{")
        (model_dir / "scores.json").write_text('[{"benchmark": "swe-bench"}]')

        results = load_results(tmp_path)
        assert len(results["models"]) == 0


class TestExpectedDimensions:
    """Tests for expected dimensions from issue #2."""

    def test_expected_benchmarks_count(self):
        """Test that we have 5 expected benchmarks."""
        assert len(EXPECTED_BENCHMARKS) == 5

    def test_expected_metrics_count(self):
        """Test that we have 3 expected metrics."""
        assert len(EXPECTED_METRICS) == 3
        assert "score" in EXPECTED_METRICS
        assert "cost_per_instance" in EXPECTED_METRICS
        assert "average_runtime" in EXPECTED_METRICS

    def test_expected_models_count(self):
        """Test that we have 11 expected models."""
        assert len(EXPECTED_MODELS) == 11
        assert "gpt-5.2-codex" in EXPECTED_MODELS
        assert "nemotron" in EXPECTED_MODELS

    def test_total_cells(self):
        """Test that total 3D array cells = 5 * 11 * 3 = 165."""
        total = len(EXPECTED_BENCHMARKS) * len(EXPECTED_MODELS) * len(EXPECTED_METRICS)
        assert total == 165


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
        assert progress["array_total_cells"] == 165

    def test_known_model_with_all_metrics(self):
        """Test progress with an expected model name and all metrics."""
        # Models must use exact expected names (enforced by schema validation)
        results = {
            "models": {"gpt-5.2"},
            "benchmarks": {"swe-bench"},
            "metrics": {"score", "cost_per_instance", "average_runtime"},
            "coverage": {
                ("gpt-5.2", "swe-bench", "score"): True,
                ("gpt-5.2", "swe-bench", "cost_per_instance"): True,
                ("gpt-5.2", "swe-bench", "average_runtime"): True,
            },
        }
        progress = calculate_progress(results)

        # gpt-5.2 is an expected model, so 1/11 models = 9.09%
        assert progress["model_coverage_pct"] == 9.09
        # 1/5 benchmarks = 20%
        assert progress["benchmark_coverage_pct"] == 20.0
        # All 3 metrics = 100%
        assert progress["metric_coverage_pct"] == 100.0
        # 3 cells filled out of 165 = 1.82%
        assert progress["array_filled_cells"] == 3
        assert progress["array_coverage_pct"] == 1.82

    def test_unknown_model_not_counted(self):
        """Test that unknown models do not contribute to model coverage or filled cells.
        
        Note: In practice, unknown models are rejected by schema validation.
        This test verifies the progress calculation behavior.
        """
        results = {
            "models": {"unknown-model-xyz"},
            "benchmarks": {"swe-bench"},
            "metrics": {"score"},
            "coverage": {("unknown-model-xyz", "swe-bench", "score"): True},
        }
        progress = calculate_progress(results)

        # Unknown model is not in EXPECTED_MODELS
        assert progress["model_coverage_pct"] == 0.0
        # But benchmark and metric coverage still count
        assert progress["benchmark_coverage_pct"] == 20.0
        assert progress["metric_coverage_pct"] == 33.33
        # No cells filled because model is not in expected list
        assert progress["array_filled_cells"] == 0

    def test_partial_metric_coverage(self):
        """Test partial metric coverage."""
        # Models must use exact expected names (enforced by schema validation)
        results = {
            "models": {"gpt-5.2"},
            "benchmarks": {"swe-bench"},
            "metrics": {"score"},  # Only score, missing cost_per_instance and average_runtime
            "coverage": {
                ("gpt-5.2", "swe-bench", "score"): True,
            },
        }
        progress = calculate_progress(results)

        # 1/3 metrics = 33.33%
        assert progress["metric_coverage_pct"] == 33.33
        # 1 cell filled (1 model * 1 benchmark * 1 metric)
        assert progress["array_filled_cells"] == 1


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
        assert progress["array_total_cells"] == 165

        # Verify actual metrics are found
        assert "score" in results["metrics"]
        assert "cost_per_instance" in results["metrics"]
        assert "average_runtime" in results["metrics"]
