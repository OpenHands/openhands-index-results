"""Tests for the validate_schema script."""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_schema import (
    Metadata,
    ScoreEntry,
    validate_metadata,
    validate_scores,
    validate_results_directory,
)


class TestMetadataSchema:
    """Tests for Metadata schema validation."""

    def test_valid_metadata(self, tmp_path):
        """Test valid metadata passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_missing_required_field(self, tmp_path):
        """Test metadata with missing required field fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            # Missing agent_version
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "agent_version" in msg

    def test_invalid_openness_value(self, tmp_path):
        """Test metadata with invalid openness value fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",
            "openness": "invalid_value",  # Invalid
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "openness" in msg.lower()

    def test_invalid_model_value(self, tmp_path):
        """Test metadata with invalid model value fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "invalid-model-name",  # Invalid - not in Model enum
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_invalid-model-name"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "model" in msg.lower()

    def test_valid_semantic_version(self, tmp_path):
        """Test metadata with valid semantic version passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.2",
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.8.2_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_semantic_version_commit_hash(self, tmp_path):
        """Test metadata with commit hash instead of semantic version fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "54c5858",  # Invalid - commit hash
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "54c5858_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "agent_version" in msg.lower()

    def test_invalid_semantic_version_no_v_prefix(self, tmp_path):
        """Test metadata with semantic version without 'v' prefix fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "1.0.0",  # Invalid - missing 'v' prefix
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "1.0.0_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "agent_version" in msg.lower()

    def test_invalid_semantic_version_branch_name(self, tmp_path):
        """Test metadata with branch name instead of semantic version fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "main",  # Invalid - branch name
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "main_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "agent_version" in msg.lower()

    def test_valid_directory_name_format(self, tmp_path):
        """Test metadata with valid directory_name format passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "claude-4.5-sonnet",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.8.3_claude-4.5-sonnet"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_directory_name_date_format(self, tmp_path):
        """Test metadata with old date-based directory_name format fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "claude-4.5-sonnet",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "202511_claude-4.5-sonnet"  # Invalid - old date format
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "directory_name" in msg.lower()

    def test_invalid_directory_name_mismatch(self, tmp_path):
        """Test metadata with directory_name not matching version and model fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "claude-4.5-sonnet",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2"  # Invalid - doesn't match version and model
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "directory_name" in msg.lower()


class TestScoreEntrySchema:
    """Tests for ScoreEntry schema validation."""

    def test_valid_score_entry(self, tmp_path):
        """Test valid score entry passes validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.412,  # Cost per problem in USD
            "average_runtime": 3600,
            "tags": ["swe-bench"]
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_benchmark(self, tmp_path):
        """Test score entry with invalid benchmark fails validation."""
        scores = [{
            "benchmark": "invalid-benchmark",  # Invalid
            "score": 68.8,
            "metric": "accuracy",
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "benchmark" in msg.lower()

    def test_score_out_of_range(self, tmp_path):
        """Test score entry with score > 100 fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 150.0,  # Invalid - > 100
            "metric": "accuracy",
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "score" in msg.lower()

    def test_negative_cost(self, tmp_path):
        """Test score entry with negative cost_per_instance fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": -0.5,  # Invalid - negative
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "cost_per_instance" in msg.lower()

    def test_optional_fields(self, tmp_path):
        """Test that optional fields can be omitted."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_full_archive_url(self, tmp_path):
        """Test score entry with valid full_archive URL passes validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench"]
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_full_archive_url_wrong_prefix(self, tmp_path):
        """Test score entry with full_archive URL not starting with CDN prefix fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "full_archive": "https://storage.googleapis.com/openhands-evaluation-results/eval-12345.tar.gz",
            "tags": ["swe-bench"]
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "full_archive" in msg.lower()
        assert "results.eval.all-hands.dev" in msg

    def test_full_archive_url_none_is_valid(self, tmp_path):
        """Test score entry with full_archive set to null passes validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "full_archive": None,
            "tags": ["swe-bench"]
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"


class TestValidateResultsDirectory:
    """Tests for validate_results_directory function."""

    def test_empty_directory(self, tmp_path):
        """Test validation of empty directory."""
        passed, failed, errors = validate_results_directory(tmp_path)
        assert passed == 0
        assert failed == 0
        assert len(errors) == 0

    def test_valid_result_directory(self, tmp_path):
        """Test validation of valid result directory."""
        model_dir = tmp_path / "v1.0.0_gpt-5.2"
        model_dir.mkdir()

        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2"
        }
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.412,  # Cost per problem in USD
            "average_runtime": 0,
            "tags": ["swe-bench"]
        }]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        passed, failed, errors = validate_results_directory(tmp_path)
        assert passed == 2
        assert failed == 0
        assert len(errors) == 0

    def test_missing_metadata(self, tmp_path):
        """Test validation fails when metadata.json is missing."""
        model_dir = tmp_path / "v1.0.0_gpt-5.2"
        model_dir.mkdir()

        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "tags": []
        }]
        (model_dir / "scores.json").write_text(json.dumps(scores))

        passed, failed, errors = validate_results_directory(tmp_path)
        assert failed >= 1
        assert any("missing metadata.json" in e for e in errors)


class TestIntegration:
    """Integration tests using the actual results directory."""

    def test_actual_results_directory(self):
        """Test validation of actual results directory."""
        repo_root = Path(__file__).parent.parent
        results_dir = repo_root / "results"

        if not results_dir.exists():
            return  # Skip if results directory does not exist

        passed, failed, errors = validate_results_directory(results_dir)

        # All files should pass validation
        assert failed == 0, f"Validation errors: {errors}"
        assert passed > 0
