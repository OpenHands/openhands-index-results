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
            "model": "GPT-5.2",
            "country": "us",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "GPT-5.2-Codex",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2-Codex",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_nemotron_3_nano_30b(self, tmp_path):
        """Test valid metadata for nemotron-3-nano passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "Nemotron-3-Nano",
            "country": "us",
            "openness": "open_weights",
            "tool_usage": "standard",
            "submission_time": "2026-01-27T20:02:11.332283+00:00",
            "directory_name": "Nemotron-3-Nano",
            "release_date": "2026-01-15",
            "parameter_count_b": 31.6,
            "active_parameter_count_b": 3.2
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_kimi_k2_5(self, tmp_path):
        """Test valid metadata for kimi-k2.5 passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "Kimi-K2.5",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "submission_time": "2026-01-29T10:00:00.000000+00:00",
            "directory_name": "Kimi-K2.5",
            "release_date": "2026-01-20",
            "parameter_count_b": 1000,
            "active_parameter_count_b": 32
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_glm_4_7(self, tmp_path):
        """Test valid metadata for glm-4.7 passes validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "GLM-4.7",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "submission_time": "2026-01-31T10:00:00.000000+00:00",
            "directory_name": "GLM-4.7",
            "release_date": "2026-01-25",
            "parameter_count_b": 9
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
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "GPT-5.2",
            "country": "us",
            "openness": "invalid_value",  # Invalid
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "GPT-5.2",
            "country": "invalid_country",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "GPT-5.2",
            "country": "cn",  # Should be "us" for GPT
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "directory_name": "invalid-model-name",
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
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "model": "claude-sonnet-4-5",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "claude-sonnet-4-5",
            "release_date": "2025-09-29"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_directory_name_old_format(self, tmp_path):
        """Test metadata with old version-prefixed directory_name format fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "claude-sonnet-4-5",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "v1.8.3_claude-sonnet-4-5",  # Invalid - old format with version prefix
            "release_date": "2025-09-29"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "directory_name" in msg.lower()

    def test_invalid_directory_name_mismatch(self, tmp_path):
        """Test metadata with directory_name not matching model fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.8.3",
            "model": "claude-sonnet-4-5",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",  # Invalid - doesn't match model
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
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2"
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
            "model": "DeepSeek-V3.2-Reasoner",
            "country": "cn",  # Open-weights model
            "openness": "open_weights",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "DeepSeek-V3.2-Reasoner",
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
            "model": "DeepSeek-V3.2-Reasoner",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "DeepSeek-V3.2-Reasoner",
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
            "model": "Kimi-K2-Thinking",
            "openness": "open_weights",
            "country": "cn",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "Kimi-K2-Thinking",
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
            "model": "GPT-5.2",
            "country": "us",  # Closed model
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "full_archive" in msg.lower()

    def test_missing_agent_version(self, tmp_path):
        """Test score entry without agent_version fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench"],
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "agent_version" in msg.lower()

    def test_missing_submission_time(self, tmp_path):
        """Test score entry without submission_time fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "submission_time" in msg.lower()

    def test_invalid_agent_version_in_score_entry(self, tmp_path):
        """Test score entry with invalid agent_version fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "invalid-version",  # Invalid - not semver
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "agent_version" in msg.lower()


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
        model_dir = tmp_path / "GPT-5.2"
        model_dir.mkdir()

        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11"
        }
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.412,  # Cost per problem in USD
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        passed, failed, errors = validate_results_directory(tmp_path)
        assert passed == 2
        assert failed == 0
        assert len(errors) == 0

    def test_missing_metadata(self, tmp_path):
        """Test validation fails when metadata.json is missing."""
        model_dir = tmp_path / "GPT-5.2"
        model_dir.mkdir()

        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "model": "GPT-5.2",
            "openness": "closed_api_available",
            "country": "us",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2"
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
            "directory_name": "invalid-model",
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
            "full_archive": "https://results.eval.all-hands.dev/test.tar.gz",
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
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
            "model": "GPT-5.2",
            "openness": "closed_api_available",
            "country": "us",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
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
                "full_archive": "https://results.eval.all-hands.dev/test.tar.gz",
                "agent_version": "v1.0.0",
                "submission_time": "2025-11-24T19:56:00.092865"
            },
            {
                "benchmark": "invalid-benchmark",  # Invalid
                "score": 50,
                "metric": "accuracy",
                "cost_per_instance": 0.5,
                "average_runtime": 300,
                "full_archive": "https://results.eval.all-hands.dev/test.tar.gz",
                "agent_version": "v1.0.0",
                "submission_time": "2025-11-24T19:56:00.092865"
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
            "directory_name": "invalid-model",
            "release_date": "2025-12-11"
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        # Check that error message doesn't include Pydantic URLs
        assert "https://errors.pydantic.dev" not in msg


class TestSweMultimodalValidation:
    """Tests for swe-bench-multimodal specific validation."""

    def test_valid_swe_multimodal_entry(self, tmp_path):
        """Test valid swe-bench-multimodal entry passes validation."""
        scores = [{
            "benchmark": "swe-bench-multimodal",
            "score": 41.2,
            "metric": "solveable_accuracy",
            "cost_per_instance": 2.54,
            "average_runtime": 671.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench-multimodal"],
            "component_scores": {
                "solveable_accuracy": 41.2,
                "unsolveable_accuracy": 0.0,
                "combined_accuracy": 27.5,
                "solveable_resolved": 28,
                "solveable_total": 68,
                "unsolveable_resolved": 0,
                "unsolveable_total": 34
            },
            "agent_version": "v1.8.3",
            "submission_time": "2026-01-27T01:24:15.735789+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_swe_multimodal_entry_minimal_component_scores(self, tmp_path):
        """Test valid swe-bench-multimodal entry with minimal component_scores passes validation."""
        scores = [{
            "benchmark": "swe-bench-multimodal",
            "score": 27.9,
            "metric": "solveable_accuracy",
            "cost_per_instance": 0.19,
            "average_runtime": 1515.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench-multimodal"],
            "component_scores": {
                "solveable_accuracy": 27.9,
                "unsolveable_accuracy": 0.0,
                "combined_accuracy": 18.6
            },
            "agent_version": "v1.8.3",
            "submission_time": "2026-01-27T18:40:51.252521+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

    def test_swe_multimodal_missing_component_scores(self, tmp_path):
        """Test swe-bench-multimodal entry without component_scores fails validation."""
        scores = [{
            "benchmark": "swe-bench-multimodal",
            "score": 28.4,
            "metric": "solveable_accuracy",
            "cost_per_instance": 2.37,
            "average_runtime": 602.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench-multimodal"],
            "agent_version": "v1.11.0",
            "submission_time": "2026-02-07T01:54:03+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "component_scores" in msg.lower()

    def test_swe_multimodal_wrong_metric(self, tmp_path):
        """Test swe-bench-multimodal entry with wrong metric fails validation."""
        scores = [{
            "benchmark": "swe-bench-multimodal",
            "score": 28.4,
            "metric": "accuracy",  # Should be solveable_accuracy
            "cost_per_instance": 2.37,
            "average_runtime": 602.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench-multimodal"],
            "component_scores": {
                "solveable_accuracy": 28.4,
                "unsolveable_accuracy": 0.0,
                "combined_accuracy": 18.9
            },
            "agent_version": "v1.11.0",
            "submission_time": "2026-02-07T01:54:03+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "solveable_accuracy" in msg.lower()

    def test_swe_multimodal_score_mismatch(self, tmp_path):
        """Test swe-bench-multimodal entry with mismatched score fails validation."""
        scores = [{
            "benchmark": "swe-bench-multimodal",
            "score": 50.0,  # Doesn't match component_scores.solveable_accuracy
            "metric": "solveable_accuracy",
            "cost_per_instance": 2.37,
            "average_runtime": 602.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench-multimodal"],
            "component_scores": {
                "solveable_accuracy": 28.4,
                "unsolveable_accuracy": 0.0,
                "combined_accuracy": 18.9
            },
            "agent_version": "v1.11.0",
            "submission_time": "2026-02-07T01:54:03+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "score" in msg.lower()
        assert "component_scores" in msg.lower()

    def test_swe_multimodal_missing_required_component_field(self, tmp_path):
        """Test swe-bench-multimodal entry with missing required component field fails validation."""
        scores = [{
            "benchmark": "swe-bench-multimodal",
            "score": 28.4,
            "metric": "solveable_accuracy",
            "cost_per_instance": 2.37,
            "average_runtime": 602.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench-multimodal"],
            "component_scores": {
                "solveable_accuracy": 28.4,
                # Missing unsolveable_accuracy and combined_accuracy
            },
            "agent_version": "v1.11.0",
            "submission_time": "2026-02-07T01:54:03+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        # Should fail due to missing required fields in component_scores

    def test_swe_multimodal_invalid_component_score_range(self, tmp_path):
        """Test swe-bench-multimodal entry with out-of-range component score fails validation."""
        scores = [{
            "benchmark": "swe-bench-multimodal",
            "score": 28.4,
            "metric": "solveable_accuracy",
            "cost_per_instance": 2.37,
            "average_runtime": 602.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench-multimodal"],
            "component_scores": {
                "solveable_accuracy": 28.4,
                "unsolveable_accuracy": 150.0,  # Invalid - > 100
                "combined_accuracy": 18.9
            },
            "agent_version": "v1.11.0",
            "submission_time": "2026-02-07T01:54:03+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False

    def test_other_benchmarks_dont_require_component_scores(self, tmp_path):
        """Test that other benchmarks don't require component_scores."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 76.6,
            "metric": "accuracy",
            "cost_per_instance": 1.82,
            "average_runtime": 325.0,
            "full_archive": "https://results.eval.all-hands.dev/eval-12345.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.8.3",
            "submission_time": "2026-01-27T01:24:15.735789+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"
