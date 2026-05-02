"""Tests for the update_verified_models script."""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from update_verified_models import (
    add_models_to_list,
    detect_provider,
    extract_completed_model_ids,
    find_missing_models,
    generate_updated_content,
    normalize_model_name,
    parse_all_verified_lists,
    parse_verified_list,
)


class TestNormalizeModelName:
    """Tests for normalize_model_name function."""

    def test_lowercase(self):
        assert normalize_model_name("GPT-5.2") == "gpt-5.2"

    def test_already_lowercase(self):
        assert normalize_model_name("claude-opus-4-5") == "claude-opus-4-5"

    def test_mixed_case(self):
        assert normalize_model_name("MiniMax-M2.1") == "minimax-m2.1"

    def test_complex_name(self):
        assert normalize_model_name("DeepSeek-V3.2-Reasoner") == "deepseek-v3.2-reasoner"

    def test_with_uppercase_suffix(self):
        assert normalize_model_name("Qwen3-Coder-480B") == "qwen3-coder-480b"


class TestDetectProvider:
    """Tests for detect_provider function."""

    def test_openai_gpt(self):
        assert detect_provider("gpt-5.2") == "openai"

    def test_openai_o_series(self):
        assert detect_provider("o4-mini") == "openai"

    def test_anthropic(self):
        assert detect_provider("claude-opus-4-5") == "anthropic"

    def test_gemini(self):
        assert detect_provider("gemini-3-flash") == "gemini"

    def test_deepseek(self):
        assert detect_provider("deepseek-v3.2-reasoner") == "deepseek"

    def test_moonshot(self):
        assert detect_provider("kimi-k2-thinking") == "moonshot"

    def test_minimax(self):
        assert detect_provider("minimax-m2.1") == "minimax"

    def test_glm(self):
        assert detect_provider("glm-4.7") == "glm"

    def test_nvidia(self):
        assert detect_provider("nemotron-3-nano") == "nvidia"

    def test_qwen(self):
        assert detect_provider("qwen3-coder-480b") == "qwen"

    def test_unknown(self):
        assert detect_provider("unknown-model") is None


class TestExtractCompletedModelIds:
    """Tests for extract_completed_model_ids function."""

    def test_extracts_results_only(self, tmp_path):
        """Only models from results/ should be extracted."""
        data = [
            {"timestamp": "2026-04-01T00:00:00.000+00:00", "model-path": "results/GPT-5.2"},
            {"timestamp": "2026-04-01T00:00:00.000+00:00", "model-path": "results/claude-opus-4-5"},
            {"timestamp": "2026-04-01T00:00:00.000+00:00", "model-path": "alternative_agents/acp-claude/claude-opus-4-7"},
        ]
        complete_models = tmp_path / "complete-models.json"
        complete_models.write_text(json.dumps(data))

        result = extract_completed_model_ids(complete_models, tmp_path)

        assert result == {"gpt-5.2", "claude-opus-4-5"}

    def test_deduplicates(self, tmp_path):
        """Duplicate model names should be deduplicated."""
        data = [
            {"timestamp": "2026-04-01T00:00:00.000+00:00", "model-path": "results/GPT-5.2"},
            {"timestamp": "2026-03-01T00:00:00.000+00:00", "model-path": "results/GPT-5.2"},
        ]
        complete_models = tmp_path / "complete-models.json"
        complete_models.write_text(json.dumps(data))

        result = extract_completed_model_ids(complete_models, tmp_path)

        assert result == {"gpt-5.2"}

    def test_empty_list(self, tmp_path):
        complete_models = tmp_path / "complete-models.json"
        complete_models.write_text("[]")

        result = extract_completed_model_ids(complete_models, tmp_path)

        assert result == set()

    def test_normalizes_names(self, tmp_path):
        data = [
            {"timestamp": "2026-04-01T00:00:00.000+00:00", "model-path": "results/MiniMax-M2.5"},
        ]
        complete_models = tmp_path / "complete-models.json"
        complete_models.write_text(json.dumps(data))

        result = extract_completed_model_ids(complete_models, tmp_path)

        assert result == {"minimax-m2.5"}


class TestParseVerifiedLists:
    """Tests for parsing verified model lists."""

    SAMPLE_CONTENT = '''VERIFIED_OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5.4",
]

VERIFIED_ANTHROPIC_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
]

VERIFIED_OPENHANDS_MODELS = [
    "claude-opus-4-5",
    "gpt-5.2",
]

VERIFIED_MODELS = {
    "openhands": VERIFIED_OPENHANDS_MODELS,
    "anthropic": VERIFIED_ANTHROPIC_MODELS,
    "openai": VERIFIED_OPENAI_MODELS,
}
'''

    def test_parse_single_list(self):
        result = parse_verified_list(self.SAMPLE_CONTENT, "VERIFIED_OPENAI_MODELS")
        assert result == ["gpt-5.2", "gpt-5.4"]

    def test_parse_all_lists(self):
        result = parse_all_verified_lists(self.SAMPLE_CONTENT)
        assert "VERIFIED_OPENAI_MODELS" in result
        assert "VERIFIED_ANTHROPIC_MODELS" in result
        assert "VERIFIED_OPENHANDS_MODELS" in result
        assert result["VERIFIED_OPENAI_MODELS"] == ["gpt-5.2", "gpt-5.4"]
        assert result["VERIFIED_ANTHROPIC_MODELS"] == ["claude-opus-4-5", "claude-sonnet-4-5"]

    def test_parse_empty_list(self):
        content = 'VERIFIED_EMPTY_MODELS = [\n]\n'
        result = parse_verified_list(content, "VERIFIED_EMPTY_MODELS")
        assert result == []

    def test_parse_nonexistent_list(self):
        result = parse_verified_list(self.SAMPLE_CONTENT, "VERIFIED_NONEXISTENT_MODELS")
        assert result == []


class TestFindMissingModels:
    """Tests for find_missing_models function."""

    def test_all_present(self):
        completed = {"gpt-5.2", "claude-opus-4-5"}
        verified = {
            "VERIFIED_OPENHANDS_MODELS": ["gpt-5.2", "claude-opus-4-5"],
            "VERIFIED_OPENAI_MODELS": ["gpt-5.2"],
            "VERIFIED_ANTHROPIC_MODELS": ["claude-opus-4-5"],
        }
        missing_oh, missing_prov = find_missing_models(completed, verified)
        assert missing_oh == []
        assert missing_prov == {}

    def test_missing_from_openhands(self):
        completed = {"gpt-5.2", "claude-opus-4-5"}
        verified = {
            "VERIFIED_OPENHANDS_MODELS": ["gpt-5.2"],
            "VERIFIED_OPENAI_MODELS": ["gpt-5.2"],
            "VERIFIED_ANTHROPIC_MODELS": ["claude-opus-4-5"],
        }
        missing_oh, missing_prov = find_missing_models(completed, verified)
        assert missing_oh == ["claude-opus-4-5"]
        assert missing_prov == {}

    def test_missing_from_provider(self):
        completed = {"gpt-5.2", "gpt-5.5"}
        verified = {
            "VERIFIED_OPENHANDS_MODELS": ["gpt-5.2", "gpt-5.5"],
            "VERIFIED_OPENAI_MODELS": ["gpt-5.2"],
        }
        missing_oh, missing_prov = find_missing_models(completed, verified)
        assert missing_oh == []
        assert missing_prov == {"VERIFIED_OPENAI_MODELS": ["gpt-5.5"]}

    def test_missing_from_both(self):
        completed = {"gemini-3-flash"}
        verified = {
            "VERIFIED_OPENHANDS_MODELS": [],
            "VERIFIED_GEMINI_MODELS": [],
        }
        missing_oh, missing_prov = find_missing_models(completed, verified)
        assert missing_oh == ["gemini-3-flash"]
        assert missing_prov == {"VERIFIED_GEMINI_MODELS": ["gemini-3-flash"]}

    def test_unknown_provider(self):
        """Models with unknown provider only appear in openhands missing list."""
        completed = {"unknown-model-x"}
        verified = {
            "VERIFIED_OPENHANDS_MODELS": [],
        }
        missing_oh, missing_prov = find_missing_models(completed, verified)
        assert missing_oh == ["unknown-model-x"]
        assert missing_prov == {}

    def test_sorted_output(self):
        completed = {"glm-5", "claude-opus-4-5", "gpt-5.2"}
        verified = {
            "VERIFIED_OPENHANDS_MODELS": [],
            "VERIFIED_OPENAI_MODELS": [],
            "VERIFIED_ANTHROPIC_MODELS": [],
            "VERIFIED_GLM_MODELS": [],
        }
        missing_oh, _ = find_missing_models(completed, verified)
        assert missing_oh == ["claude-opus-4-5", "glm-5", "gpt-5.2"]


class TestAddModelsToList:
    """Tests for add_models_to_list function."""

    def test_add_to_existing_list(self):
        content = 'VERIFIED_OPENAI_MODELS = [\n    "gpt-5.2",\n]\n'
        result = add_models_to_list(content, "VERIFIED_OPENAI_MODELS", ["gpt-5.5"])
        assert '"gpt-5.2"' in result
        assert '"gpt-5.5"' in result

    def test_add_multiple_models(self):
        content = 'VERIFIED_OPENAI_MODELS = [\n    "gpt-5.2",\n]\n'
        result = add_models_to_list(
            content, "VERIFIED_OPENAI_MODELS", ["gpt-5.4", "gpt-5.5"]
        )
        assert '"gpt-5.4"' in result
        assert '"gpt-5.5"' in result

    def test_preserves_other_content(self):
        content = 'BEFORE = 1\n\nVERIFIED_X_MODELS = [\n    "a",\n]\n\nAFTER = 2\n'
        result = add_models_to_list(content, "VERIFIED_X_MODELS", ["b"])
        assert "BEFORE = 1" in result
        assert "AFTER = 2" in result
        assert '"b"' in result

    def test_nonexistent_list_unchanged(self):
        content = 'VERIFIED_X_MODELS = [\n    "a",\n]\n'
        result = add_models_to_list(content, "VERIFIED_NONEXISTENT_MODELS", ["b"])
        assert result == content


class TestGenerateUpdatedContent:
    """Tests for generate_updated_content function."""

    def test_adds_to_both_lists(self):
        content = '''VERIFIED_OPENAI_MODELS = [
    "gpt-5.2",
]

VERIFIED_OPENHANDS_MODELS = [
    "gpt-5.2",
]
'''
        updated = generate_updated_content(
            content,
            missing_openhands=["gpt-5.5"],
            missing_providers={"VERIFIED_OPENAI_MODELS": ["gpt-5.5"]},
        )
        # gpt-5.5 should appear in both lists
        assert updated.count('"gpt-5.5"') == 2

    def test_no_changes_when_empty(self):
        content = 'VERIFIED_OPENHANDS_MODELS = [\n    "gpt-5.2",\n]\n'
        updated = generate_updated_content(content, [], {})
        assert updated == content

    def test_only_openhands_missing(self):
        content = '''VERIFIED_OPENAI_MODELS = [
    "gpt-5.2",
    "gpt-5.5",
]

VERIFIED_OPENHANDS_MODELS = [
    "gpt-5.2",
]
'''
        updated = generate_updated_content(
            content,
            missing_openhands=["gpt-5.5"],
            missing_providers={},
        )
        # gpt-5.5 should now also be in OPENHANDS list
        lines = updated.split("\n")
        oh_section = False
        found_in_oh = False
        for line in lines:
            if "VERIFIED_OPENHANDS_MODELS" in line:
                oh_section = True
            elif oh_section and "]" in line:
                oh_section = False
            elif oh_section and '"gpt-5.5"' in line:
                found_in_oh = True
        assert found_in_oh
