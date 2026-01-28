"""Tests for the validate_schema script."""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_schema import (
    Metadata,
    ScoreEntry,
    format_validation_error,
    validate_metadata,
    validate_scores,
    validate_results_directory,
)
from pydantic import ValidationError


class TestMetadataSchema:
    """Tests for Metadata schema validation."""

    def test_valid_metadata(self, tmp_path):
        """Test valid metadata passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",
            "country": "us",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_gpt_5_2_codex(self, tmp_path):
        """Test valid metadata for gpt-5.2-codex passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2-codex",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2-codex",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_nemotron(self, tmp_path):
        """Test valid metadata for nemotron passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "nemotron",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2026-01-27T20:02:11.332283+00:00",
            "directory_name": "v1.8.3_nemotron",
            "release_date": "2026-01-15"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
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
            "country": "us",
            "openness": "invalid_value",  # Invalid
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "openness" in msg.lower()

    def test_invalid_country_value(self, tmp_path):
        """Test metadata with invalid country value fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",
            "country": "invalid_country",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "country" in msg.lower()

    def test_country_mismatch_for_model(self, tmp_path):
        """Test metadata with incorrect country for model fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",
            "country": "cn",  # Should be "us" for GPT
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "country" in msg.lower()

    def test_invalid_model_value(self, tmp_path):
        """Test metadata with invalid model value fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "invalid-model-name",  # Invalid - not in Model enum
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_invalid-model-name",
            "release_date": "2025-12-11"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.8.2_gpt-5.2",
            "release_date": "2025-12-11"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "54c5858_gpt-5.2",
            "release_date": "2025-12-11"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "main_gpt-5.2",
            "release_date": "2025-12-11"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.8.3_claude-4.5-sonnet",
            "release_date": "2025-09-29"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "202511_claude-4.5-sonnet",  # Invalid - old date format
            "release_date": "2025-09-29"
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
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",  # Invalid - doesn't match version and model
            "release_date": "2025-09-29"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "directory_name" in msg.lower()

    def test_missing_release_date(self, tmp_path):
        """Test metadata with missing release_date fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2"
            # Missing release_date
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "release_date" in msg.lower()

    def test_open_weights_model_requires_parameter_count_b(self, tmp_path):
        """Test that open-weights models require parameter_count_b."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "deepseek-v3.2-reasoner",
            "country": "cn",  # Open-weights model
            "openness": "open_weights",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_deepseek-v3.2-reasoner",
            "release_date": "2025-12-01"
            # Missing parameter_count_b - should fail for open-weights model
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "parameter_count_b" in msg.lower()

    def test_open_weights_model_with_parameter_count_b(self, tmp_path):
        """Test that open-weights models pass validation with parameter_count_b."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "deepseek-v3.2-reasoner",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_deepseek-v3.2-reasoner",
            "release_date": "2025-12-01",
            "parameter_count_b": 685
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_open_weights_model_with_active_parameter_count_b(self, tmp_path):
        """Test that open-weights MoE models can include active_parameter_count_b."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "kimi-k2-thinking",
            "openness": "open_weights",
            "country": "cn",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_kimi-k2-thinking",
            "release_date": "2025-11-06",
            "parameter_count_b": 1000,
            "active_parameter_count_b": 32
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_closed_model_without_parameter_count_b(self, tmp_path):
        """Test that closed models pass validation without parameter_count_b."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "gpt-5.2",
            "country": "us",  # Closed model
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
            # No parameter_count_b - should be OK for closed model
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"


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
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
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
            "cost_per_instance": 0.5,
            "average_runtime": 300,
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
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "score" in msg.lower()

    def test_negative_score(self, tmp_path):
        """Test score entry with negative score fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": -10.0,  # Invalid - negative
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
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
            "average_runtime": 300,
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "cost_per_instance" in msg.lower()

    def test_zero_cost(self, tmp_path):
        """Test score entry with zero cost_per_instance fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0,  # Invalid - must be > 0
            "average_runtime": 300,
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "cost_per_instance" in msg.lower()

    def test_zero_average_runtime(self, tmp_path):
        """Test score entry with zero average_runtime fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 0,  # Invalid - must be > 0
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "average_runtime" in msg.lower()

    def test_negative_average_runtime(self, tmp_path):
        """Test score entry with negative average_runtime fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": -100,  # Invalid - negative
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "average_runtime" in msg.lower()

    def test_optional_fields(self, tmp_path):
        """Test that optional fields (tags) can be omitted."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

    def test_missing_cost_per_instance(self, tmp_path):
        """Test score entry without cost_per_instance fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "average_runtime": 300,
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "cost_per_instance" in msg.lower()

    def test_missing_average_runtime(self, tmp_path):
        """Test score entry without average_runtime fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "tags": []
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "average_runtime" in msg.lower()

    def test_valid_full_archive_url(self, tmp_path):
        """Test score entry with valid full_archive URL passes validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
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
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://storage.googleapis.com/openhands-evaluation-results/eval-12345.tar.gz",
            "tags": ["swe-bench"]
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "full_archive" in msg.lower()
        assert "results.eval.all-hands.dev" in msg

    def test_missing_full_archive(self, tmp_path):
        """Test score entry without full_archive fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "tags": ["swe-bench"]
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "full_archive" in msg.lower()


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
            "model": "gpt-5.2",
            "country": "us",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2",
            "release_date": "2025-12-11"
        }
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.412,  # Cost per problem in USD
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
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
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
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


class TestErrorMessageFormatting:
    """Tests for human-readable error message formatting."""

    def test_missing_field_error_message(self, tmp_path):
        """Test that missing field errors show clear reason."""
        metadata = {
            "agent_name": "Test Agent",
            # Missing agent_version, release_date, etc.
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "country": "us",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_gpt-5.2"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        # Check that error message clearly states the field is missing
        assert "Field 'agent_version' is required but missing" in msg
        assert "Field 'release_date' is required but missing" in msg

    def test_invalid_enum_error_message(self, tmp_path):
        """Test that invalid enum errors show the invalid value."""
        metadata = {
            "agent_name": "Test Agent",
            "agent_version": "v1.0.0",
            "model": "invalid-model",
            "openness": "closed_api_available",
            "country": "us",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.0.0_invalid-model",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        # Check that error message shows the invalid value
        assert "Field 'model'" in msg
        assert "got: 'invalid-model'" in msg

    def test_value_constraint_error_message(self, tmp_path):
        """Test that value constraint errors show the actual value."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 150,  # Invalid: > 100
            "metric": "accuracy",
            "cost_per_instance": -1,  # Invalid: <= 0
            "average_runtime": 0,  # Invalid: <= 0
            "full_archive": "https://results.eval.all-hands.dev/test.tar.gz"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        # Check that error messages show the actual values
        assert "Field 'score'" in msg
        assert "got: 150" in msg
        assert "Field 'cost_per_instance'" in msg
        assert "got: -1" in msg

    def test_custom_validator_error_message(self, tmp_path):
        """Test that custom validator errors show clear reason."""
        metadata = {
            "agent_name": "Test Agent",
            "agent_version": "invalid-version",  # Invalid: not semver
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "country": "us",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "invalid-version_gpt-5.2",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        # Check that error message explains the validation rule
        assert "Field 'agent_version'" in msg
        assert "semantic version" in msg

    def test_invalid_json_error_message(self, tmp_path):
        """Test that invalid JSON errors show line and column."""
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text('{ "invalid": json }')

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "Invalid JSON" in msg
        assert "line" in msg

    def test_scores_entry_error_shows_entry_index(self, tmp_path):
        """Test that scores validation errors show which entry failed."""
        scores = [
            {
                "benchmark": "swe-bench",
                "score": 50,
                "metric": "accuracy",
                "cost_per_instance": 0.5,
                "average_runtime": 300,
                "full_archive": "https://results.eval.all-hands.dev/test.tar.gz"
            },
            {
                "benchmark": "invalid-benchmark",  # Invalid
                "score": 50,
                "metric": "accuracy",
                "cost_per_instance": 0.5,
                "average_runtime": 300,
                "full_archive": "https://results.eval.all-hands.dev/test.tar.gz"
            }
        ]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        # Check that error message shows which entry failed
        assert "Entry 1" in msg

    def test_error_message_no_pydantic_urls(self, tmp_path):
        """Test that error messages don't include Pydantic documentation URLs."""
        metadata = {
            "agent_name": "Test Agent",
            "agent_version": "invalid",
            "model": "invalid-model",
            "openness": "invalid",
            "country": "invalid",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "invalid_invalid-model",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        # Check that error message doesn't include Pydantic URLs
        assert "https://errors.pydantic.dev" not in msg
