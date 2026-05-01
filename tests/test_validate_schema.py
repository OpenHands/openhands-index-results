"""Tests for the validate_schema script."""

import json
import sys
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_schema import (
    AgentName,
    Metadata,
    ScoreEntry,
    format_validation_error,
    validate_metadata,
    validate_scores,
    validate_results_directory,
    validate_alternative_agents_directory,
    parse_semver,
    is_version_earlier_or_equal,
    validate_metadata_agent_version,
)
from pydantic import ValidationError


class TestMetadataSchema:
    """Tests for Metadata schema validation."""

    def test_valid_metadata(self, tmp_path):
        """Test valid metadata passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_gpt_5_2_codex(self, tmp_path):
        """Test valid metadata for gpt-5.2-codex passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2-Codex",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2-Codex",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_gpt_5_5(self, tmp_path):
        """Test valid metadata for GPT-5.5 passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.18.1",
            "model": "GPT-5.5",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.5",
            "release_date": "2026-04-23",
            "supports_vision": True,
            "input_price": 5.0,
            "output_price": 30.0,
            "cache_read_price": 0.5,
            "cache_write_price": None
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_nemotron_3_nano_30b(self, tmp_path):
        """Test valid metadata for nemotron-3-nano passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "Nemotron-3-Nano",
            "country": "us",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "Nemotron-3-Nano",
            "release_date": "2026-01-15",
            "supports_vision": False,
            "input_price": 0.1,
            "output_price": 0.1,
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
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "Kimi-K2.5",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "Kimi-K2.5",
            "release_date": "2026-01-20",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1,
            "parameter_count_b": 1000,
            "active_parameter_count_b": 32
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_kimi_k2_6(self, tmp_path):
        """Test valid metadata for kimi-k2.6 passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.11.5",
            "model": "Kimi-K2.6",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "Kimi-K2.6",
            "release_date": "2026-04-20",
            "supports_vision": True,
            "input_price": 0.95,
            "output_price": 4.00,
            "parameter_count_b": 1000,
            "active_parameter_count_b": 32,
            "cache_read_price": 0.16,
            "cache_write_price": None
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_glm_4_7(self, tmp_path):
        """Test valid metadata for glm-4.7 passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "GLM-4.7",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "GLM-4.7",
            "release_date": "2026-01-25",
            "supports_vision": False,
            "input_price": 0.1,
            "output_price": 0.1,
            "parameter_count_b": 9
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_glm_5_1(self, tmp_path):
        """Test valid metadata for glm-5.1 passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.11.5",
            "model": "GLM-5.1",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "GLM-5.1",
            "release_date": "2026-04-07",
            "supports_vision": False,
            "input_price": 1.0,
            "output_price": 3.2,
            "parameter_count_b": 754,
            "active_parameter_count_b": 40,
            "cache_read_price": 0.2
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_minimax_m2_5(self, tmp_path):
        """Test valid metadata for MiniMax-M2.5 passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.11.0",
            "model": "MiniMax-M2.5",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "MiniMax-M2.5",
            "release_date": "2026-02-11",
            "supports_vision": False,
            "input_price": 0.1,
            "output_price": 0.1,
            "parameter_count_b": 230,
            "active_parameter_count_b": 10
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_qwen3_6_plus(self, tmp_path):
        """Test valid metadata for Qwen3.6-Plus passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.11.5",
            "model": "Qwen3.6-Plus",
            "country": "cn",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "Qwen3.6-Plus",
            "release_date": "2026-04-01",
            "supports_vision": True,
            "input_price": 0.5,
            "output_price": 3.0,
            "parameter_count_b": None,
            "active_parameter_count_b": None,
            "cache_read_price": None,
            "cache_write_price": None
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_claude_opus_4_7(self, tmp_path):
        """Test valid metadata for claude-opus-4-7 passes validation."""
        metadata = {
            "agent_name": "Claude Code",
            "agent_version": "v1.18.0",
            "model": "claude-opus-4-7",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "claude-opus-4-7",
            "release_date": "2026-04-16",
            "supports_vision": True,
            "input_price": 5.0,
            "output_price": 25.0,
            "cache_read_price": 0.5,
            "cache_write_price": 6.25
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_metadata_trinity_large_thinking(self, tmp_path):
        """Test valid metadata for Trinity-Large-Thinking passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.11.5",
            "model": "Trinity-Large-Thinking",
            "country": "us",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "Trinity-Large-Thinking",
            "release_date": "2026-04-01",
            "supports_vision": False,
            "input_price": 0.25,
            "output_price": 0.9,
            "parameter_count_b": 398,
            "active_parameter_count_b": 13,
            "cache_read_price": None,
            "cache_write_price": None
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_missing_required_field(self, tmp_path):
        """Test metadata with missing required field fails validation."""
        metadata = {
            "agent_name": "OpenHands",
            # Missing agent_version
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "invalid_value",  # Invalid
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "invalid_country",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "cn",  # Should be "us" for GPT
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "invalid-model-name",  # Invalid - not in Model enum
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.8.2",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_semantic_version_commit_hash(self, tmp_path):
        """Test metadata with commit hash instead of semantic version fails validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "54c5858",  # Invalid - commit hash
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "1.0.0",  # Invalid - missing 'v' prefix
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "main",  # Invalid - branch name
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "claude-sonnet-4-5",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "claude-sonnet-4-5",
            "release_date": "2025-09-29",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_directory_name_old_format(self, tmp_path):
        """Test metadata with old version-prefixed directory_name format fails validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "claude-sonnet-4-5",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "claude-sonnet-4-5",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "DeepSeek-V3.2-Reasoner",
            "country": "cn",  # Open-weights model
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "DeepSeek-V3.2-Reasoner",
            "release_date": "2025-12-01",
            "supports_vision": False,
            "input_price": 0.1,
            "output_price": 0.1
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "DeepSeek-V3.2-Reasoner",
            "country": "cn",
            "openness": "open_weights",
            "tool_usage": "standard",
            "directory_name": "DeepSeek-V3.2-Reasoner",
            "release_date": "2025-12-01",
            "supports_vision": False,
            "input_price": 0.1,
            "output_price": 0.1,
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "Kimi-K2-Thinking",
            "openness": "open_weights",
            "country": "cn",
            "tool_usage": "standard",
            "directory_name": "Kimi-K2-Thinking",
            "release_date": "2025-11-06",
            "supports_vision": False,
            "input_price": 0.1,
            "output_price": 0.1,
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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",  # Closed model
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
            # No parameter_count_b - should be OK for closed model
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_missing_supports_vision(self, tmp_path):
        """Test metadata with missing supports_vision fails validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11"
            # Missing supports_vision - should fail
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "supports_vision" in msg.lower()

    def test_valid_agent_name_openhands(self, tmp_path):
        """Test metadata with 'OpenHands' agent_name passes validation."""
        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_agent_name_claude_code(self, tmp_path):
        """Test metadata with 'Claude Code' agent_name passes validation."""
        metadata = {
            "agent_name": "Claude Code",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_agent_name_opencode(self, tmp_path):
        """Test metadata with 'OpenCode' agent_name passes validation."""
        metadata = {
            "agent_name": "OpenCode",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_agent_name_codex(self, tmp_path):
        """Test metadata with 'Codex' agent_name passes validation."""
        metadata = {
            "agent_name": "Codex",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_agent_name_gemini_cli(self, tmp_path):
        """Test metadata with 'Gemini CLI' agent_name passes validation."""
        metadata = {
            "agent_name": "Gemini CLI",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is True
        assert msg == "OK"

    def test_invalid_agent_name(self, tmp_path):
        """Test metadata with invalid agent_name fails validation."""
        metadata = {
            "agent_name": "OpenHands CodeAct",  # Invalid - old name not allowed
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "agent_name" in msg.lower()

    def test_invalid_agent_name_unknown(self, tmp_path):
        """Test metadata with unknown agent_name fails validation."""
        metadata = {
            "agent_name": "Unknown Agent",  # Invalid - not in allowed list
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata))

        valid, msg = validate_metadata(metadata_file)
        assert valid is False
        assert "agent_name" in msg.lower()


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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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

    def test_invalid_full_archive_url_missing_filename(self, tmp_path):
        """Test score entry with full_archive URL missing filename fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "full_archive" in msg.lower()

    def test_invalid_full_archive_url_wrong_format(self, tmp_path):
        """Test score entry with full_archive URL not matching expected patterns fails validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/random-file.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "full_archive" in msg.lower()
        assert "expected format" in msg.lower()

    def test_valid_full_archive_legacy_format(self, tmp_path):
        """Test score entry with valid legacy format full_archive URL passes validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-gpt-5-2-co_litellm_proxy-gpt-5-2-codex_26-01-27-12-57.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

    def test_valid_full_archive_benchmark_format(self, tmp_path):
        """Test score entry with valid benchmark format full_archive URL passes validation."""
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/swtbench/litellm_proxy-anthropic-claude-opus-4-6/21754233398/results.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"

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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "invalid-version",  # Invalid - not semver
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is False
        assert "agent_version" in msg.lower()


class TestScoreEntryProvenanceFields:
    """Tests for the optional ACP provenance fields on ScoreEntry.

    These fields are stamped by OpenHands/benchmarks PR #646 and
    OpenHands/evaluation PR #440 so ACP runs record exactly which ACP
    binary handled the evaluation. Both fields must remain Optional for
    backward compatibility with score entries written before the
    provenance stamping pipeline landed. agent_version continues to
    carry the openhands-sdk version for every run.
    """

    _BASE_ENTRY = {
        "benchmark": "swe-bench",
        "score": 68.8,
        "metric": "accuracy",
        "cost_per_instance": 0.5,
        "average_runtime": 300,
        "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
        "tags": ["swe-bench"],
        "agent_version": "v1.16.1",
        "submission_time": "2025-11-24T19:56:00.092865",
    }

    def _write(self, tmp_path, entry):
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps([entry]))
        return scores_file

    def test_backward_compat_score_entry_without_provenance_fields(self, tmp_path):
        """Existing pre-provenance score entries must still validate."""
        scores_file = self._write(tmp_path, self._BASE_ENTRY)
        valid, msg = validate_scores(scores_file)
        assert valid is True, msg
        assert msg == "OK"

    def test_default_agent_entry_without_acp_fields(self, tmp_path):
        """Default-agent run: only agent_version (SDK version), no ACP fields."""
        valid, msg = validate_scores(self._write(tmp_path, self._BASE_ENTRY))
        assert valid is True, msg

    def test_acp_entry_with_full_provenance(self, tmp_path):
        """ACP run: agent_version is still the SDK version, ACP fields set together."""
        entry = {
            **self._BASE_ENTRY,
            "agent_version": "v1.16.1",  # SDK version, same as default runs
            "acp_agent_name": "@agentclientprotocol/claude-agent-acp",
            "acp_agent_version": "v0.25.3",
        }
        valid, msg = validate_scores(self._write(tmp_path, entry))
        assert valid is True, msg

    def test_invalid_acp_agent_version_format(self, tmp_path):
        """acp_agent_version must match the v-prefixed semver pattern."""
        entry = {
            **self._BASE_ENTRY,
            "acp_agent_name": "@agentclientprotocol/claude-agent-acp",
            "acp_agent_version": "0.25.3",  # missing 'v' prefix
        }
        valid, msg = validate_scores(self._write(tmp_path, entry))
        assert valid is False
        assert "acp_agent_version" in msg

    def test_acp_name_without_version_is_rejected(self, tmp_path):
        """Partial ACP pair: name without version is a capture bug."""
        entry = {
            **self._BASE_ENTRY,
            "acp_agent_name": "@agentclientprotocol/claude-agent-acp",
        }
        valid, msg = validate_scores(self._write(tmp_path, entry))
        assert valid is False
        assert "acp_agent_name" in msg and "acp_agent_version" in msg

    def test_acp_version_without_name_is_rejected(self, tmp_path):
        """Partial ACP pair: version without name is a capture bug."""
        entry = {**self._BASE_ENTRY, "acp_agent_version": "v0.25.3"}
        valid, msg = validate_scores(self._write(tmp_path, entry))
        assert valid is False
        assert "acp_agent_name" in msg and "acp_agent_version" in msg

    def test_unknown_fields_still_ignored(self, tmp_path):
        """Pydantic's extra='ignore' default must remain in effect.

        This is the safety net that lets the producer pipeline add new
        provenance fields in the future without needing a simultaneous
        schema update here.
        """
        entry = {**self._BASE_ENTRY, "some_future_field": "hello"}
        valid, msg = validate_scores(self._write(tmp_path, entry))
        assert valid is True, msg


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
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",  # Must be an expected model name
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2025-11-24T19:56:00.092865",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.412,  # Cost per problem in USD
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
                "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
                "agent_version": "v1.0.0",
                "submission_time": "2025-11-24T19:56:00.092865"
            },
            {
                "benchmark": "invalid-benchmark",  # Invalid
                "score": 50,
                "metric": "accuracy",
                "cost_per_instance": 0.5,
                "average_runtime": 300,
                "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
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
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.8.3",
            "submission_time": "2026-01-27T01:24:15.735789+00:00"
        }]
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(scores))

        valid, msg = validate_scores(scores_file)
        assert valid is True
        assert msg == "OK"


class TestValidateAlternativeAgentsDirectory:
    """Tests for validate_alternative_agents_directory function."""

    def _make_valid_model_dir(self, parent_dir: Path, model_name: str = "GPT-5.2") -> None:
        """Helper to create a valid model directory with metadata.json and scores.json."""
        model_dir = parent_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "agent_name": "Claude Code",
            "agent_version": "v1.0.0",
            "model": model_name,
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": model_name,
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.412,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

    def test_nonexistent_directory(self, tmp_path):
        """Test validation of nonexistent alternative_agents directory returns empty results."""
        passed, failed, errors = validate_alternative_agents_directory(tmp_path / "nonexistent")
        assert passed == 0
        assert failed == 0
        assert len(errors) == 0

    def test_empty_alternative_agents_directory(self, tmp_path):
        """Test validation of empty alternative_agents directory."""
        alt_dir = tmp_path / "alternative_agents"
        alt_dir.mkdir()

        passed, failed, errors = validate_alternative_agents_directory(alt_dir)
        assert passed == 0
        assert failed == 0
        assert len(errors) == 0

    def test_valid_agent_with_model(self, tmp_path):
        """Test validation passes for a valid model under an agent type directory."""
        alt_dir = tmp_path / "alternative_agents"
        agent_dir = alt_dir / "claude_code"
        agent_dir.mkdir(parents=True)

        self._make_valid_model_dir(agent_dir, "GPT-5.2")

        passed, failed, errors = validate_alternative_agents_directory(alt_dir)
        assert passed == 2  # metadata.json + scores.json
        assert failed == 0
        assert len(errors) == 0

    def test_multiple_agents_multiple_models(self, tmp_path):
        """Test validation works for multiple agents each with model directories."""
        alt_dir = tmp_path / "alternative_agents"

        # Create two agent directories, each with a model
        claude_dir = alt_dir / "claude_code"
        claude_dir.mkdir(parents=True)
        self._make_valid_model_dir(claude_dir, "GPT-5.2")

        codex_dir = alt_dir / "codex"
        codex_dir.mkdir(parents=True)
        self._make_valid_model_dir(codex_dir, "GPT-5.2")

        passed, failed, errors = validate_alternative_agents_directory(alt_dir)
        assert passed == 4  # 2 agents * 2 files each
        assert failed == 0
        assert len(errors) == 0

    def test_invalid_metadata_in_alternative_agent(self, tmp_path):
        """Test that invalid metadata in alternative_agents is caught."""
        alt_dir = tmp_path / "alternative_agents"
        agent_dir = alt_dir / "claude_code"
        model_dir = agent_dir / "GPT-5.2"
        model_dir.mkdir(parents=True)

        # Invalid metadata (missing required fields)
        metadata = {"agent_name": "Claude Code"}
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        passed, failed, errors = validate_alternative_agents_directory(alt_dir)
        assert failed >= 1
        assert len(errors) >= 1

    def test_missing_metadata_in_alternative_agent(self, tmp_path):
        """Test that missing metadata.json in alternative_agents is caught."""
        alt_dir = tmp_path / "alternative_agents"
        agent_dir = alt_dir / "claude_code"
        model_dir = agent_dir / "GPT-5.2"
        model_dir.mkdir(parents=True)

        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
            "tags": [],
            "agent_version": "v1.0.0",
            "submission_time": "2025-11-24T19:56:00.092865"
        }]
        (model_dir / "scores.json").write_text(json.dumps(scores))

        passed, failed, errors = validate_alternative_agents_directory(alt_dir)
        assert failed >= 1
        assert any("missing metadata.json" in e for e in errors)

    def test_gitkeep_files_are_ignored(self, tmp_path):
        """Test that .gitkeep files in agent directories don't cause issues."""
        alt_dir = tmp_path / "alternative_agents"
        agent_dir = alt_dir / "claude_code"
        agent_dir.mkdir(parents=True)
        (agent_dir / ".gitkeep").write_text("")

        passed, failed, errors = validate_alternative_agents_directory(alt_dir)
        assert passed == 0
        assert failed == 0
        assert len(errors) == 0


class TestParseSemver:
    """Tests for parse_semver helper function."""

    def test_valid_version(self):
        """Test parsing a valid semver string."""
        major, minor, patch = parse_semver("v1.2.3")
        assert major == 1
        assert minor == 2
        assert patch == 3

    def test_version_with_single_digits(self):
        """Test parsing version with single digits."""
        major, minor, patch = parse_semver("v0.1.0")
        assert major == 0
        assert minor == 1
        assert patch == 0

    def test_version_with_large_numbers(self):
        """Test parsing version with large numbers."""
        major, minor, patch = parse_semver("v123.456.789")
        assert major == 123
        assert minor == 456
        assert patch == 789

    def test_invalid_version_missing_v(self):
        """Test that version without 'v' prefix raises ValueError."""
        with pytest.raises(ValueError):
            parse_semver("1.2.3")

    def test_invalid_version_wrong_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_semver("v1.2")


class TestIsVersionEarlierOrEqual:
    """Tests for is_version_earlier_or_equal helper function."""

    def test_same_version(self):
        """Test same versions are equal."""
        assert is_version_earlier_or_equal("v1.0.0", "v1.0.0") is True

    def test_earlier_major(self):
        """Test earlier major version."""
        assert is_version_earlier_or_equal("v1.0.0", "v2.0.0") is True

    def test_later_major(self):
        """Test later major version."""
        assert is_version_earlier_or_equal("v2.0.0", "v1.0.0") is False

    def test_earlier_minor(self):
        """Test earlier minor version."""
        assert is_version_earlier_or_equal("v1.1.0", "v1.2.0") is True

    def test_later_minor(self):
        """Test later minor version."""
        assert is_version_earlier_or_equal("v1.3.0", "v1.2.0") is False

    def test_earlier_patch(self):
        """Test earlier patch version."""
        assert is_version_earlier_or_equal("v1.0.1", "v1.0.2") is True

    def test_later_patch(self):
        """Test later patch version."""
        assert is_version_earlier_or_equal("v1.0.3", "v1.0.2") is False

    def test_earlier_complex(self):
        """Test earlier complex version."""
        assert is_version_earlier_or_equal("v1.10.0", "v1.11.0") is True


class TestValidateMetadataAgentVersion:
    """Tests for validate_metadata_agent_version function."""

    def _make_model_dir(self, tmp_path, model_name="GPT-5.2"):
        """Helper to create a model directory with metadata.json and scores.json."""
        model_dir = tmp_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _make_scores(self, agent_versions):
        """Helper to create scores entries with given agent_versions."""
        scores = []
        for i, version in enumerate(agent_versions):
            scores.append({
                "benchmark": "swe-bench",
                "score": 68.8,
                "metric": "accuracy",
                "cost_per_instance": 0.5,
                "average_runtime": 300,
                "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
                "tags": ["swe-bench"],
                "agent_version": version,
                "submission_time": "2025-11-24T19:56:00.092865"
            })
        return scores

    def test_metadata_version_is_earliest(self, tmp_path):
        """Test validation passes when metadata version is the earliest."""
        model_dir = self._make_model_dir(tmp_path)

        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        scores = self._make_scores(["v1.0.0", "v1.1.0", "v1.2.0"])

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        valid, msg = validate_metadata_agent_version(
            model_dir / "metadata.json",
            model_dir / "scores.json"
        )
        assert valid is True
        assert msg == "OK"

    def test_metadata_version_matches_earliest_with_duplicates(self, tmp_path):
        """Test validation passes when metadata version matches earliest with duplicates."""
        model_dir = self._make_model_dir(tmp_path)

        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.1.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        # Earliest is v1.1.0 (appears twice)
        scores = self._make_scores(["v1.1.0", "v1.1.0", "v1.2.0"])

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        valid, msg = validate_metadata_agent_version(
            model_dir / "metadata.json",
            model_dir / "scores.json"
        )
        assert valid is True
        assert msg == "OK"

    def test_metadata_version_later_than_earliest_fails(self, tmp_path):
        """Test validation fails when metadata version is later than earliest in scores."""
        model_dir = self._make_model_dir(tmp_path)

        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",  # This is LATER than scores versions
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        scores = self._make_scores(["v1.0.0", "v1.1.0", "v1.2.0"])  # Earliest is v1.0.0

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        valid, msg = validate_metadata_agent_version(
            model_dir / "metadata.json",
            model_dir / "scores.json"
        )
        assert valid is False
        assert "v1.8.3" in msg
        assert "v1.0.0" in msg

    def test_empty_scores_list_is_valid(self, tmp_path):
        """Test validation passes with empty scores list."""
        model_dir = self._make_model_dir(tmp_path)

        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        scores = []  # Empty list

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        valid, msg = validate_metadata_agent_version(
            model_dir / "metadata.json",
            model_dir / "scores.json"
        )
        assert valid is True
        assert msg == "OK"

    def test_no_agent_version_in_scores_is_valid(self, tmp_path):
        """Test validation passes when scores have no agent_version."""
        model_dir = self._make_model_dir(tmp_path)

        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        # Scores without agent_version
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
            "tags": ["swe-bench"],
            "submission_time": "2025-11-24T19:56:00.092865"
        }]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        valid, msg = validate_metadata_agent_version(
            model_dir / "metadata.json",
            model_dir / "scores.json"
        )
        assert valid is True
        assert msg == "OK"

    def test_invalid_json_in_metadata(self, tmp_path):
        """Test validation handles invalid JSON in metadata."""
        model_dir = self._make_model_dir(tmp_path)

        (model_dir / "metadata.json").write_text("not valid json {")
        scores = self._make_scores(["v1.0.0"])

        (model_dir / "scores.json").write_text(json.dumps(scores))

        valid, msg = validate_metadata_agent_version(
            model_dir / "metadata.json",
            model_dir / "scores.json"
        )
        assert valid is False
        assert "Invalid JSON" in msg

    def test_invalid_json_in_scores(self, tmp_path):
        """Test validation handles invalid JSON in scores."""
        model_dir = self._make_model_dir(tmp_path)

        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.0.0",
            "model": "GPT-5.2",
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": "GPT-5.2",
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text("not valid json [")

        valid, msg = validate_metadata_agent_version(
            model_dir / "metadata.json",
            model_dir / "scores.json"
        )
        assert valid is False
        assert "Invalid JSON" in msg


class TestAgentVersionConsistencyInValidation:
    """Tests for agent_version consistency validation in _validate_model_dirs."""

    def _make_model_dir_with_version_mismatch(self, tmp_path, model_name="GPT-5.2"):
        """Helper to create a model directory where metadata version is later than scores versions."""
        model_dir = tmp_path / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "agent_name": "OpenHands",
            "agent_version": "v1.8.3",  # Later than scores versions
            "model": model_name,
            "country": "us",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "directory_name": model_name,
            "release_date": "2025-12-11",
            "supports_vision": True,
            "input_price": 0.1,
            "output_price": 0.1
        }
        scores = [{
            "benchmark": "swe-bench",
            "score": 68.8,
            "metric": "accuracy",
            "cost_per_instance": 0.5,
            "average_runtime": 300,
            "full_archive": "https://results.eval.all-hands.dev/eval-21386738547-test_litellm_proxy-test_26-01-27-12-57.tar.gz",
            "tags": ["swe-bench"],
            "agent_version": "v1.0.0",  # Earlier than metadata version
            "submission_time": "2025-11-24T19:56:00.092865"
        }]

        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        (model_dir / "scores.json").write_text(json.dumps(scores))

        return model_dir

    def test_version_mismatch_caught_in_results_dir(self, tmp_path):
        """Test that version mismatch is caught in results directory validation."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        self._make_model_dir_with_version_mismatch(results_dir, "GPT-5.2")

        passed, failed, errors = validate_results_directory(results_dir)

        # Should have at least 3 validations:
        # 1. metadata.json schema (passed)
        # 2. scores.json schema (passed)
        # 3. agent_version consistency (failed)
        assert failed >= 1
        assert any("not the earliest version" in e for e in errors)

    def test_version_mismatch_caught_in_alternative_agents(self, tmp_path):
        """Test that version mismatch is caught in alternative_agents validation."""
        alt_dir = tmp_path / "alternative_agents"
        agent_dir = alt_dir / "claude_code"
        agent_dir.mkdir(parents=True)
        self._make_model_dir_with_version_mismatch(agent_dir, "GPT-5.2")

        passed, failed, errors = validate_alternative_agents_directory(alt_dir)

        # Should catch the version mismatch error
        assert failed >= 1
        assert any("not the earliest version" in e for e in errors)
