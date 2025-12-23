"""Tests for the push_to_index script."""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from push_to_index import (
    BENCHMARK_NAME_MAP,
    VALID_BENCHMARKS,
    calculate_accuracy,
    create_metadata,
    create_score_entry,
    generate_directory_name,
    get_cost_and_duration,
    load_json,
    load_jsonl,
    normalize_benchmark_name,
    update_scores,
)


class TestLoadJson:
    """Tests for load_json function."""

    def test_valid_json(self, tmp_path):
        """Test loading valid JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')
        result = load_json(json_file)
        assert result == {"key": "value"}

    def test_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file returns None."""
        result = load_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_invalid_json(self, tmp_path):
        """Test loading invalid JSON returns None."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json {{{")
        result = load_json(json_file)
        assert result is None


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_valid_jsonl(self, tmp_path):
        """Test loading valid JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n{"b": 2}\n')
        result = load_jsonl(jsonl_file)
        assert result == [{"a": 1}, {"b": 2}]

    def test_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file returns empty list."""
        result = load_jsonl(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"a": 1}\n\n{"b": 2}\n\n')
        result = load_jsonl(jsonl_file)
        assert result == [{"a": 1}, {"b": 2}]

    def test_invalid_jsonl(self, tmp_path):
        """Test loading invalid JSONL returns empty list."""
        jsonl_file = tmp_path / "invalid.jsonl"
        jsonl_file.write_text("not valid json\n")
        result = load_jsonl(jsonl_file)
        assert result == []


class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""

    def test_normal_accuracy(self):
        """Test normal accuracy calculation."""
        report = {"resolved_instances": 50, "submitted_instances": 100}
        assert calculate_accuracy(report) == 50.0

    def test_zero_submitted(self):
        """Test accuracy with zero submitted instances."""
        report = {"resolved_instances": 10, "submitted_instances": 0}
        assert calculate_accuracy(report) == 0.0

    def test_missing_fields(self):
        """Test accuracy with missing fields defaults to 0."""
        assert calculate_accuracy({}) == 0.0
        assert calculate_accuracy({"resolved_instances": 10}) == 0.0
        assert calculate_accuracy({"submitted_instances": 100}) == 0.0

    def test_full_accuracy(self):
        """Test 100% accuracy."""
        report = {"resolved_instances": 100, "submitted_instances": 100}
        assert calculate_accuracy(report) == 100.0

    def test_decimal_accuracy(self):
        """Test accuracy with decimal result."""
        report = {"resolved_instances": 33, "submitted_instances": 100}
        assert calculate_accuracy(report) == 33.0


class TestGetCostAndDuration:
    """Tests for get_cost_and_duration function."""

    def test_single_entry(self):
        """Test with single cost report entry."""
        cost_report = [{"total_cost": 100.5, "total_duration": 3600}]
        cost, duration = get_cost_and_duration(cost_report)
        assert cost == 100.5
        assert duration == 3600

    def test_multiple_entries(self):
        """Test with multiple cost report entries."""
        cost_report = [
            {"total_cost": 50.0, "total_duration": 1800},
            {"total_cost": 75.5, "total_duration": 2400},
        ]
        cost, duration = get_cost_and_duration(cost_report)
        assert cost == 125.5
        assert duration == 4200

    def test_empty_report(self):
        """Test with empty cost report."""
        cost, duration = get_cost_and_duration([])
        assert cost == 0.0
        assert duration == 0.0

    def test_missing_fields(self):
        """Test with missing fields defaults to 0."""
        cost_report = [{"total_cost": 100}, {"total_duration": 1800}]
        cost, duration = get_cost_and_duration(cost_report)
        assert cost == 100.0
        assert duration == 1800.0

    def test_none_values(self):
        """Test with None values defaults to 0."""
        cost_report = [{"total_cost": None, "total_duration": None}]
        cost, duration = get_cost_and_duration(cost_report)
        assert cost == 0.0
        assert duration == 0.0


class TestNormalizeBenchmarkName:
    """Tests for normalize_benchmark_name function."""

    def test_swe_bench_variations(self):
        """Test SWE-bench name variations."""
        assert normalize_benchmark_name("swe-bench") == "swe-bench"
        assert normalize_benchmark_name("swe_bench") == "swe-bench"
        assert normalize_benchmark_name("swebench") == "swe-bench"
        assert normalize_benchmark_name("SWE-BENCH") == "swe-bench"

    def test_swt_bench_variations(self):
        """Test SWT-bench name variations."""
        assert normalize_benchmark_name("swt-bench") == "swt-bench"
        assert normalize_benchmark_name("swt_bench") == "swt-bench"
        assert normalize_benchmark_name("swtbench") == "swt-bench"

    def test_gaia(self):
        """Test GAIA benchmark."""
        assert normalize_benchmark_name("gaia") == "gaia"
        assert normalize_benchmark_name("GAIA") == "gaia"

    def test_commit0(self):
        """Test commit0 benchmark."""
        assert normalize_benchmark_name("commit0") == "commit0"
        assert normalize_benchmark_name("commit-0") == "commit0"

    def test_multi_swe_bench(self):
        """Test multi-swe-bench variations."""
        assert normalize_benchmark_name("multi-swe-bench") == "multi-swe-bench"
        assert normalize_benchmark_name("multi_swe_bench") == "multi-swe-bench"
        assert normalize_benchmark_name("multiswebench") == "multi-swe-bench"

    def test_swe_bench_multimodal(self):
        """Test swe-bench-multimodal variations."""
        assert normalize_benchmark_name("swe-bench-multimodal") == "swe-bench-multimodal"
        assert normalize_benchmark_name("swe_bench_multimodal") == "swe-bench-multimodal"
        assert normalize_benchmark_name("swebench-multimodal") == "swe-bench-multimodal"

    def test_unknown_benchmark(self):
        """Test unknown benchmark returns as-is (lowercase)."""
        assert normalize_benchmark_name("unknown-benchmark") == "unknown-benchmark"


class TestGenerateDirectoryName:
    """Tests for generate_directory_name function."""

    def test_basic_model_name(self):
        """Test basic model name."""
        date = datetime(2025, 11, 15)
        result = generate_directory_name("gpt-4o", date)
        assert result == "202511_gpt-4o"

    def test_model_with_slashes(self):
        """Test model name with slashes."""
        date = datetime(2025, 12, 1)
        result = generate_directory_name("org/model-name", date)
        assert result == "202512_org-model-name"

    def test_model_with_spaces(self):
        """Test model name with spaces."""
        date = datetime(2025, 10, 20)
        result = generate_directory_name("My Model Name", date)
        assert result == "202510_My-Model-Name"

    def test_default_date(self):
        """Test that default date uses current date."""
        result = generate_directory_name("test-model")
        expected_prefix = datetime.now().strftime("%Y%m")
        assert result.startswith(expected_prefix)
        assert result.endswith("_test-model")


class TestCreateMetadata:
    """Tests for create_metadata function."""

    def test_basic_metadata(self):
        """Test basic metadata creation."""
        metadata = create_metadata(
            model_name="gpt-4o",
            directory_name="202511_gpt-4o",
        )
        assert metadata["model"] == "gpt-4o"
        assert metadata["agent_name"] == "OpenHands CodeAct"
        assert metadata["agent_version"] == "unknown"
        assert metadata["openness"] == "closed_api_available"
        assert metadata["tool_usage"] == "standard"
        assert metadata["directory_name"] == "202511_gpt-4o"
        assert "submission_time" in metadata

    def test_custom_metadata(self):
        """Test metadata with custom values."""
        metadata = create_metadata(
            model_name="llama-3",
            agent_name="Custom Agent",
            agent_version="1.2.3",
            openness="open_weights",
            tool_usage="custom",
            directory_name="202511_llama-3",
        )
        assert metadata["model"] == "llama-3"
        assert metadata["agent_name"] == "Custom Agent"
        assert metadata["agent_version"] == "1.2.3"
        assert metadata["openness"] == "open_weights"
        assert metadata["tool_usage"] == "custom"


class TestCreateScoreEntry:
    """Tests for create_score_entry function."""

    def test_basic_score_entry(self):
        """Test basic score entry creation."""
        entry = create_score_entry(
            benchmark="swe-bench",
            score=65.5,
        )
        assert entry["benchmark"] == "swe-bench"
        assert entry["score"] == 65.5
        assert entry["metric"] == "accuracy"
        assert entry["total_cost"] == 0
        assert entry["total_runtime"] == 0
        assert entry["tags"] == ["swe-bench"]

    def test_score_entry_with_cost(self):
        """Test score entry with cost and runtime."""
        entry = create_score_entry(
            benchmark="gaia",
            score=80.0,
            total_cost=150.5,
            total_runtime=7200,
        )
        assert entry["benchmark"] == "gaia"
        assert entry["score"] == 80.0
        assert entry["total_cost"] == 150.5
        assert entry["total_runtime"] == 7200

    def test_score_entry_normalizes_benchmark(self):
        """Test that benchmark name is normalized."""
        entry = create_score_entry(
            benchmark="swe_bench",
            score=50.0,
        )
        assert entry["benchmark"] == "swe-bench"
        assert entry["tags"] == ["swe-bench"]


class TestUpdateScores:
    """Tests for update_scores function."""

    def test_add_new_score(self):
        """Test adding a new score to empty list."""
        existing = []
        new_entry = {"benchmark": "swe-bench", "score": 65.0}
        result = update_scores(existing, new_entry)
        assert len(result) == 1
        assert result[0] == new_entry

    def test_update_existing_score(self):
        """Test updating an existing score."""
        existing = [
            {"benchmark": "swe-bench", "score": 50.0},
            {"benchmark": "gaia", "score": 70.0},
        ]
        new_entry = {"benchmark": "swe-bench", "score": 65.0}
        result = update_scores(existing, new_entry)
        assert len(result) == 2
        assert result[0]["benchmark"] == "swe-bench"
        assert result[0]["score"] == 65.0
        assert result[1]["benchmark"] == "gaia"
        assert result[1]["score"] == 70.0

    def test_add_different_benchmark(self):
        """Test adding a score for a different benchmark."""
        existing = [{"benchmark": "swe-bench", "score": 50.0}]
        new_entry = {"benchmark": "gaia", "score": 75.0}
        result = update_scores(existing, new_entry)
        assert len(result) == 2
        assert result[0]["benchmark"] == "swe-bench"
        assert result[1]["benchmark"] == "gaia"


class TestValidBenchmarks:
    """Tests for valid benchmarks constant."""

    def test_all_benchmarks_present(self):
        """Test that all expected benchmarks are in VALID_BENCHMARKS."""
        expected = {
            "swe-bench",
            "swt-bench",
            "gaia",
            "commit0",
            "multi-swe-bench",
            "swe-bench-multimodal",
        }
        assert VALID_BENCHMARKS == expected

    def test_benchmark_count(self):
        """Test that we have 6 valid benchmarks."""
        assert len(VALID_BENCHMARKS) == 6


class TestBenchmarkNameMap:
    """Tests for benchmark name mapping."""

    def test_all_variations_map_to_valid(self):
        """Test that all mapped names are valid benchmarks."""
        for source, target in BENCHMARK_NAME_MAP.items():
            assert target in VALID_BENCHMARKS, f"{source} maps to invalid benchmark {target}"


class TestIntegration:
    """Integration tests for push_to_index functionality."""

    def test_full_workflow_dry_run(self, tmp_path):
        """Test the full workflow in dry-run mode."""
        # Create test report file
        report_path = tmp_path / "output.report.json"
        report_path.write_text(json.dumps({
            "resolved_instances": 65,
            "submitted_instances": 100,
        }))

        # Create test cost report file
        cost_report_path = tmp_path / "cost_report.jsonl"
        cost_report_path.write_text(
            '{"total_cost": 100.5, "total_duration": 3600}\n'
            '{"total_cost": 50.0, "total_duration": 1800}\n'
        )

        # Load and verify data
        report_data = load_json(report_path)
        assert report_data is not None
        accuracy = calculate_accuracy(report_data)
        assert accuracy == 65.0

        cost_report = load_jsonl(cost_report_path)
        assert len(cost_report) == 2
        total_cost, total_duration = get_cost_and_duration(cost_report)
        assert total_cost == 150.5
        assert total_duration == 5400

        # Generate directory name
        dir_name = generate_directory_name("test-model")
        assert "_test-model" in dir_name

        # Create metadata
        metadata = create_metadata(
            model_name="test-model",
            directory_name=dir_name,
        )
        assert metadata["model"] == "test-model"

        # Create score entry
        score_entry = create_score_entry(
            benchmark="swe-bench",
            score=accuracy,
            total_cost=total_cost,
            total_runtime=total_duration,
        )
        assert score_entry["score"] == 65.0
        assert score_entry["total_cost"] == 150.5
        assert score_entry["total_runtime"] == 5400

    def test_missing_report_file(self, tmp_path):
        """Test handling of missing report file."""
        report_path = tmp_path / "nonexistent.json"
        report_data = load_json(report_path)
        assert report_data is None
        # Should default to 0 accuracy
        accuracy = calculate_accuracy(report_data or {})
        assert accuracy == 0.0

    def test_missing_cost_report_file(self, tmp_path):
        """Test handling of missing cost report file."""
        cost_report_path = tmp_path / "nonexistent.jsonl"
        cost_report = load_jsonl(cost_report_path)
        assert cost_report == []
        # Should default to 0 cost and duration
        total_cost, total_duration = get_cost_and_duration(cost_report)
        assert total_cost == 0.0
        assert total_duration == 0.0
