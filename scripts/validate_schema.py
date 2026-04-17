#!/usr/bin/env python3
"""
Validate JSON files against Pydantic schemas.

This script checks that all metadata.json and scores.json files
in the results directory conform to the expected schema.
"""

import ast
import json
import re
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


def format_validation_error(error: ValidationError) -> str:
    """Format a Pydantic ValidationError into a human-readable message.

    Args:
        error: The Pydantic ValidationError to format.

    Returns:
        A formatted string with clear error messages.
    """
    messages = []
    for err in error.errors():
        field = ".".join(str(loc) for loc in err["loc"])
        error_type = err["type"]
        msg = err["msg"]
        input_value = err.get("input")

        # Build a clear, human-readable message
        if error_type == "missing":
            messages.append(f"  • Field '{field}' is required but missing")
        elif error_type == "enum":
            messages.append(f"  • Field '{field}': {msg} (got: '{input_value}')")
        elif error_type == "value_error":
            # Extract the actual error message without "Value error, " prefix
            clean_msg = msg.replace("Value error, ", "")
            messages.append(f"  • Field '{field}': {clean_msg}")
        elif error_type in ("less_than_equal", "greater_than", "greater_than_equal", "less_than"):
            messages.append(f"  • Field '{field}': {msg} (got: {input_value})")
        else:
            messages.append(f"  • Field '{field}': {msg} (got: '{input_value}')")

    return "\n".join(messages)


def check_for_duplicate_dict_keys() -> None:
    """Check for duplicate keys in MODEL_OPENNESS_MAP and MODEL_COUNTRY_MAP.
    
    This function parses the source code of this script to detect duplicate
    dictionary keys that might have been introduced by git merge. When two PRs
    add the same model with different values, git may auto-merge both entries
    without conflict, but Python will silently use the last value.
    
    Raises:
        SystemExit: If duplicate keys are found, with exit code 1.
    """
    script_path = Path(__file__)
    source_code = script_path.read_text()
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Error: Failed to parse {script_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Track dictionaries we care about and their duplicate keys
    dict_duplicates: dict[str, list[str]] = {}
    
    for node in ast.walk(tree):
        # Look for assignment statements
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            var_name = node.target.id
            # Check if this is one of our mapping dictionaries
            if var_name in ("MODEL_OPENNESS_MAP", "MODEL_COUNTRY_MAP"):
                if isinstance(node.value, ast.Dict):
                    # Extract all keys and check for duplicates
                    keys_seen = {}
                    duplicates = []
                    
                    for i, key_node in enumerate(node.value.keys):
                        # Extract the key string (e.g., "Model.GLM_5")
                        if isinstance(key_node, ast.Attribute):
                            if isinstance(key_node.value, ast.Name):
                                key_str = f"{key_node.value.id}.{key_node.attr}"
                                
                                if key_str in keys_seen:
                                    duplicates.append(key_str)
                                else:
                                    keys_seen[key_str] = i
                    
                    if duplicates:
                        dict_duplicates[var_name] = duplicates
    
    # Report any duplicates found
    if dict_duplicates:
        print("=" * 60, file=sys.stderr)
        print("CRITICAL ERROR: Duplicate Dictionary Keys Detected", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(file=sys.stderr)
        print("The validation script has duplicate keys in dictionary literals.", file=sys.stderr)
        print("This likely occurred due to a git merge where two PRs added the", file=sys.stderr)
        print("same model with different values.", file=sys.stderr)
        print(file=sys.stderr)
        
        for dict_name, duplicates in dict_duplicates.items():
            print(f"Dictionary: {dict_name}", file=sys.stderr)
            for dup in duplicates:
                print(f"  - Duplicate key: {dup}", file=sys.stderr)
        
        print(file=sys.stderr)
        print("Python silently uses the LAST occurrence of duplicate keys,", file=sys.stderr)
        print("which causes inconsistent validation behavior.", file=sys.stderr)
        print(file=sys.stderr)
        print("To fix this:", file=sys.stderr)
        print("1. Review the git history to determine the correct value", file=sys.stderr)
        print("2. Remove the duplicate entries, keeping only one correct entry", file=sys.stderr)
        print("3. Ensure the metadata.json files match the chosen value", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)


SEMVER_PATTERN = re.compile(r'^v\d+\.\d+\.\d+$')

# Pattern for full_archive URLs
# Two formats are supported:
# 1. Legacy format: (eval-)?{run_id}-{model_short}_litellm_proxy-{provider}-{model}_{YY-MM-DD-HH-MM}.tar.gz
# 2. Benchmark format: {benchmark}/litellm_proxy-{model}/{run_id}/results.tar.gz
FULL_ARCHIVE_LEGACY_PATTERN = re.compile(
    r'^(eval-)?\d+-[a-zA-Z0-9_-]+_litellm_proxy-[a-zA-Z0-9_-]+_\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.tar\.gz$'
)
FULL_ARCHIVE_BENCHMARK_PATTERN = re.compile(
    r'^[a-z0-9]+/litellm_proxy-[a-zA-Z0-9_-]+/\d+/results\.tar\.gz$'
)


class AgentName(str, Enum):
    """Supported agent names."""
    OPENHANDS = "OpenHands"
    OPENHANDS_SUB_AGENTS = "OpenHands Sub-agents"
    CLAUDE_CODE = "Claude Code"
    OPENCODE = "OpenCode"
    CODEX = "Codex"
    GEMINI_CLI = "Gemini CLI"


class Openness(str, Enum):
    """Model openness classification."""
    OPEN_WEIGHTS = "open_weights"
    CLOSED_API_AVAILABLE = "closed_api_available"
    CLOSED = "closed"


class Country(str, Enum):
    """Country of origin for the model."""
    US = "us"
    CN = "cn"
    FR = "fr"


class ToolUsage(str, Enum):
    """Tool usage classification."""
    STANDARD = "standard"
    CUSTOM = "custom"
    NONE = "none"


class Model(str, Enum):
    """Expected model names from issue #2."""
    CLAUDE_OPUS_4_7 = "claude-opus-4-7"
    CLAUDE_OPUS_4_6 = "claude-opus-4-6"
    CLAUDE_OPUS_4_5 = "claude-opus-4-5"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"
    GEMINI_3_PRO = "Gemini-3-Pro"
    GEMINI_3_1_PRO = "Gemini-3.1-Pro"
    GEMINI_3_FLASH = "Gemini-3-Flash"
    GLM_5 = "GLM-5"
    GLM_5_1 = "GLM-5.1"
    GLM_4_7 = "GLM-4.7"
    GPT_5_2 = "GPT-5.2"
    GPT_5_2_CODEX = "GPT-5.2-Codex"
    GPT_5_4 = "GPT-5.4"
    KIMI_K2_THINKING = "Kimi-K2-Thinking"
    KIMI_K2_5 = "Kimi-K2.5"
    MINIMAX_M2_1 = "MiniMax-M2.1"
    DEEPSEEK_V3_2_REASONER = "DeepSeek-V3.2-Reasoner"
    QWEN_3_CODER = "Qwen3-Coder-480B"
    QWEN3_5_FLASH = "Qwen3.5-Flash"
    QWEN3_6_PLUS = "Qwen3.6-Plus"
    NEMOTRON_3_NANO = "Nemotron-3-Nano"
    NEMOTRON_3_SUPER = "Nemotron-3-Super"
    QWEN3_CODER_NEXT = "Qwen3-Coder-Next"
    MINIMAX_M2_5 = "MiniMax-M2.5"
    MINIMAX_2_7 = "Minimax-2.7"
    TRINITY_LARGE_THINKING = "Trinity-Large-Thinking"


# Mapping of models to their correct openness classification
# Open-weights models have publicly available model weights
# Closed API models only provide API access without weight availability
MODEL_OPENNESS_MAP: dict[Model, Openness] = {
    # Closed API models
    Model.CLAUDE_OPUS_4_7: Openness.CLOSED_API_AVAILABLE,
    Model.CLAUDE_OPUS_4_6: Openness.CLOSED_API_AVAILABLE,
    Model.CLAUDE_OPUS_4_5: Openness.CLOSED_API_AVAILABLE,
    Model.CLAUDE_SONNET_4_6: Openness.CLOSED_API_AVAILABLE,
    Model.CLAUDE_SONNET_4_5: Openness.CLOSED_API_AVAILABLE,
    Model.GEMINI_3_PRO: Openness.CLOSED_API_AVAILABLE,
    Model.GEMINI_3_1_PRO: Openness.CLOSED_API_AVAILABLE,
    Model.GEMINI_3_FLASH: Openness.CLOSED_API_AVAILABLE,
    Model.GPT_5_2: Openness.CLOSED_API_AVAILABLE,
    Model.GPT_5_2_CODEX: Openness.CLOSED_API_AVAILABLE,
    Model.GPT_5_4: Openness.CLOSED_API_AVAILABLE,
    Model.QWEN3_6_PLUS: Openness.CLOSED_API_AVAILABLE,
    # Open-weights models
    Model.GLM_5: Openness.OPEN_WEIGHTS,
    Model.GLM_5_1: Openness.OPEN_WEIGHTS,
    Model.GLM_4_7: Openness.OPEN_WEIGHTS,
    Model.KIMI_K2_THINKING: Openness.OPEN_WEIGHTS,
    Model.KIMI_K2_5: Openness.OPEN_WEIGHTS,
    Model.MINIMAX_M2_1: Openness.OPEN_WEIGHTS,
    Model.DEEPSEEK_V3_2_REASONER: Openness.OPEN_WEIGHTS,
    Model.QWEN_3_CODER: Openness.OPEN_WEIGHTS,
    Model.QWEN3_5_FLASH: Openness.OPEN_WEIGHTS,
    Model.QWEN3_CODER_NEXT: Openness.OPEN_WEIGHTS,
    Model.NEMOTRON_3_NANO: Openness.OPEN_WEIGHTS,
    Model.NEMOTRON_3_SUPER: Openness.OPEN_WEIGHTS,
    Model.MINIMAX_M2_5: Openness.OPEN_WEIGHTS,
    Model.MINIMAX_2_7: Openness.OPEN_WEIGHTS,
    Model.TRINITY_LARGE_THINKING: Openness.OPEN_WEIGHTS,
}


# Mapping of models to their country of origin
MODEL_COUNTRY_MAP: dict[Model, Country] = {
    # US models
    Model.CLAUDE_OPUS_4_6: Country.US,
    Model.CLAUDE_OPUS_4_5: Country.US,
    Model.CLAUDE_SONNET_4_6: Country.US,
    Model.CLAUDE_SONNET_4_5: Country.US,
    Model.GEMINI_3_PRO: Country.US,
    Model.GEMINI_3_1_PRO: Country.US,
    Model.GEMINI_3_FLASH: Country.US,
    Model.GPT_5_2: Country.US,
    Model.GPT_5_2_CODEX: Country.US,
    Model.GPT_5_4: Country.US,
    Model.NEMOTRON_3_NANO: Country.US,
    Model.NEMOTRON_3_SUPER: Country.US,
    Model.TRINITY_LARGE_THINKING: Country.US,
    # China models
    Model.GLM_5: Country.CN,
    Model.GLM_5_1: Country.CN,
    Model.GLM_4_7: Country.CN,
    Model.KIMI_K2_THINKING: Country.CN,
    Model.KIMI_K2_5: Country.CN,
    Model.MINIMAX_M2_1: Country.CN,
    Model.MINIMAX_M2_5: Country.CN,
    Model.DEEPSEEK_V3_2_REASONER: Country.CN,
    Model.MINIMAX_2_7: Country.CN,
    Model.QWEN_3_CODER: Country.CN,
    Model.QWEN3_5_FLASH: Country.CN,
    Model.QWEN3_6_PLUS: Country.CN,
    Model.QWEN3_CODER_NEXT: Country.CN,
}


class Metadata(BaseModel):
    """Schema for metadata.json files."""
    agent_name: AgentName = Field(..., description="Name of the agent (must be one of: OpenHands, OpenHands Sub-agents, Claude Code, OpenCode, Codex, Gemini CLI)")
    agent_version: str = Field(..., description="Version of the agent (semantic version starting with 'v')")
    model: Model = Field(..., description="Model name (must be one of the expected models)")
    openness: Openness = Field(..., description="Model openness classification")
    country: Country = Field(..., description="Country of origin for the model")
    tool_usage: ToolUsage = Field(..., description="Tool usage classification")
    directory_name: str = Field(..., description="Directory name for this result")
    release_date: date = Field(..., description="Model release date (YYYY-MM-DD)")
    supports_vision: bool = Field(..., description="Whether the model supports vision/image inputs")
    parameter_count_b: Optional[float] = Field(None, description="Total model parameter count in billions. Required for open-weights models.")
    active_parameter_count_b: Optional[float] = Field(None, description="Active parameter count in billions (for MoE models)")
    hide_from_leaderboard: bool = Field(default=False, description="Whether to hide this model from the public leaderboard")
    input_price: float = Field(..., gt=0, description="Input price per million tokens in USD")
    output_price: float = Field(..., gt=0, description="Output price per million tokens in USD")
    cache_read_price: Optional[float] = Field(None, gt=0, description="Cache read price per million tokens in USD (None if not supported)")
    cache_write_price: Optional[float] = Field(None, gt=0, description="Cache write price per million tokens in USD (None if not supported)")

    @field_validator("agent_version")
    @classmethod
    def validate_agent_version(cls, v: str) -> str:
        """Ensure agent_version is a valid semantic version starting with 'v'."""
        if not SEMVER_PATTERN.match(v):
            raise ValueError(
                f"agent_version must be a valid semantic version starting with 'v' "
                f"(e.g., 'v1.0.0'), got '{v}'"
            )
        return v

    @field_validator("openness")
    @classmethod
    def validate_openness_matches_model(cls, v: Openness, info) -> Openness:
        """Ensure openness matches the expected value for the model."""
        model = info.data.get("model")
        if model and model in MODEL_OPENNESS_MAP:
            expected_openness = MODEL_OPENNESS_MAP[model]
            if v != expected_openness:
                raise ValueError(
                    f"Model '{model.value}' should have openness '{expected_openness.value}', "
                    f"but got '{v.value}'"
                )
        return v

    @field_validator("country")
    @classmethod
    def validate_country_matches_model(cls, v: Country, info) -> Country:
        """Ensure country matches the expected value for the model."""
        model = info.data.get("model")
        if model and model in MODEL_COUNTRY_MAP:
            expected_country = MODEL_COUNTRY_MAP[model]
            if v != expected_country:
                raise ValueError(
                    f"Model '{model.value}' should have country '{expected_country.value}', "
                    f"but got '{v.value}'"
                )
        return v

    @field_validator("directory_name")
    @classmethod
    def validate_directory_name(cls, v: str, info) -> str:
        """Ensure directory_name matches the model name."""
        model = info.data.get("model")
        if model:
            expected_dir_name = model.value
            if v != expected_dir_name:
                raise ValueError(
                    f"directory_name '{v}' does not match expected model name '{expected_dir_name}'"
                )
        return v

    @model_validator(mode='after')
    def validate_parameter_count_for_open_models(self):
        """Ensure parameter_count_b is provided for open-weights models."""
        model_openness = MODEL_OPENNESS_MAP.get(self.model)
        if model_openness == Openness.OPEN_WEIGHTS and self.parameter_count_b is None:
            raise ValueError(
                f"parameter_count_b is required for open-weights model '{self.model.value}'"
            )
        return self


class Benchmark(str, Enum):
    """Expected benchmark names."""
    SWE_BENCH = "swe-bench"
    SWE_BENCH_MULTIMODAL = "swe-bench-multimodal"
    SWT_BENCH = "swt-bench"
    COMMIT0 = "commit0"
    GAIA = "gaia"


class Metric(str, Enum):
    """Expected metric names for the score field."""
    ACCURACY = "accuracy"
    SOLVEABLE_ACCURACY = "solveable_accuracy"


FULL_ARCHIVE_URL_PREFIX = "https://results.eval.all-hands.dev/"


class SweMultimodalComponentScores(BaseModel):
    """Schema for swe-bench-multimodal component_scores field."""
    solveable_accuracy: float = Field(..., ge=0, le=100, description="Solveable accuracy percentage")
    unsolveable_accuracy: float = Field(..., ge=0, le=100, description="Unsolveable accuracy percentage")
    combined_accuracy: float = Field(..., ge=0, le=100, description="Combined accuracy percentage")
    solveable_resolved: Optional[int] = Field(None, ge=0, description="Number of solveable instances resolved")
    solveable_total: Optional[int] = Field(None, ge=0, description="Total number of solveable instances")
    unsolveable_resolved: Optional[int] = Field(None, ge=0, description="Number of unsolveable instances resolved")
    unsolveable_total: Optional[int] = Field(None, ge=0, description="Total number of unsolveable instances")


class ScoreEntry(BaseModel):
    """Schema for individual score entries in scores.json."""
    benchmark: Benchmark = Field(..., description="Benchmark name")
    score: float = Field(..., ge=0, le=100, description="Score value (0-100)")
    metric: Metric = Field(..., description="Metric type for the score")
    cost_per_instance: float = Field(..., gt=0, description="Average cost per problem in USD")
    average_runtime: float = Field(..., gt=0, description="Average runtime per instance in seconds")
    full_archive: str = Field(..., description="URL to the full evaluation archive")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    agent_version: str = Field(..., description="Version of the agent (semantic version starting with 'v')")
    submission_time: datetime = Field(..., description="Submission timestamp")
    eval_visualization_page: Optional[str] = Field(None, description="URL to the evaluation visualization page")
    component_scores: Optional[SweMultimodalComponentScores] = Field(None, description="Component scores for swe-bench-multimodal benchmark")
    # Optional ACP binary identity, from the ACP server's initialize
    # handshake. Both must be set together (see validate_acp_fields_paired).
    # agent_version still carries the openhands-sdk version for all runs.
    acp_agent_name: Optional[str] = Field(
        None,
        description=(
            "ACP agent package name (e.g. "
            "'@agentclientprotocol/claude-agent-acp'), reported by the ACP "
            "server during its initialize handshake. Only set for ACP "
            "runs; present iff acp_agent_version is also present."
        ),
    )
    acp_agent_version: Optional[str] = Field(
        None,
        description=(
            "ACP agent version (e.g. 'v0.25.3'), reported by the ACP "
            "server during its initialize handshake. Only set for ACP "
            "runs; semantic version starting with 'v'."
        ),
    )

    @field_validator("agent_version")
    @classmethod
    def validate_agent_version(cls, v: str) -> str:
        """Ensure agent_version is a valid semantic version starting with 'v'."""
        if not SEMVER_PATTERN.match(v):
            raise ValueError(
                f"agent_version must be a valid semantic version starting with 'v' "
                f"(e.g., 'v1.0.0'), got '{v}'"
            )
        return v

    @field_validator("acp_agent_version")
    @classmethod
    def validate_acp_agent_version(cls, v: Optional[str]) -> Optional[str]:
        """Ensure acp_agent_version matches the v-prefixed semver pattern when present."""
        if v is None:
            return v
        if not SEMVER_PATTERN.match(v):
            raise ValueError(
                f"acp_agent_version must be a valid semantic version starting "
                f"with 'v' (e.g., 'v1.0.0'), got '{v}'"
            )
        return v

    @model_validator(mode='after')
    def validate_acp_fields_paired(self):
        """Ensure acp_agent_name and acp_agent_version are set together.

        The ACP protocol initialize handshake returns both atomically, so a
        score entry with one but not the other indicates a capture bug.
        """
        if bool(self.acp_agent_name) != bool(self.acp_agent_version):
            raise ValueError(
                "acp_agent_name and acp_agent_version must either both be "
                "set or both be omitted — they come from the same ACP "
                "initialize handshake. "
                f"Got acp_agent_name={self.acp_agent_name!r}, "
                f"acp_agent_version={self.acp_agent_version!r}."
            )
        return self

    @field_validator("full_archive")
    @classmethod
    def validate_full_archive(cls, v: str) -> str:
        """Ensure full_archive URL matches expected patterns."""
        if not v.startswith(FULL_ARCHIVE_URL_PREFIX):
            raise ValueError(
                f"full_archive must begin with '{FULL_ARCHIVE_URL_PREFIX}', got '{v}'"
            )
        # Extract the path after the prefix
        path = v[len(FULL_ARCHIVE_URL_PREFIX):]
        # Check if path matches one of the expected patterns
        if not (FULL_ARCHIVE_LEGACY_PATTERN.match(path) or FULL_ARCHIVE_BENCHMARK_PATTERN.match(path)):
            raise ValueError(
                f"full_archive path must match expected format. "
                f"Expected either '(eval-){{run_id}}-{{model}}_litellm_proxy-{{provider}}_{{date}}.tar.gz' "
                f"or '{{benchmark}}/litellm_proxy-{{model}}/{{run_id}}/results.tar.gz', got '{path}'"
            )
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Ensure tags are non-empty strings."""
        for tag in v:
            if not tag or not isinstance(tag, str):
                raise ValueError(f"Invalid tag: {tag}")
        return v

    @model_validator(mode='after')
    def validate_swe_bench_multimodal_format(self):
        """Ensure swe-bench-multimodal entries have the correct format."""
        if self.benchmark == Benchmark.SWE_BENCH_MULTIMODAL:
            # Metric must be solveable_accuracy for swe-bench-multimodal
            if self.metric != Metric.SOLVEABLE_ACCURACY:
                raise ValueError(
                    f"swe-bench-multimodal entries must use metric 'solveable_accuracy', "
                    f"got '{self.metric.value}'"
                )
            # component_scores is required for swe-bench-multimodal
            if self.component_scores is None:
                raise ValueError(
                    "swe-bench-multimodal entries must include 'component_scores' field with "
                    "solveable_accuracy, unsolveable_accuracy, and combined_accuracy"
                )
            # The score should match the solveable_accuracy in component_scores
            if abs(self.score - self.component_scores.solveable_accuracy) > 0.01:
                raise ValueError(
                    f"swe-bench-multimodal 'score' ({self.score}) must match "
                    f"'component_scores.solveable_accuracy' ({self.component_scores.solveable_accuracy})"
                )
        return self



def load_json(file_path: Path) -> dict | list:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def validate_metadata(file_path: Path) -> tuple[bool, str]:
    """Validate a metadata.json file against the schema."""
    try:
        data = load_json(file_path)
        Metadata(**data)
        return True, "OK"
    except ValidationError as e:
        return False, format_validation_error(e)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}"
    except Exception as e:
        return False, str(e)


def validate_scores(file_path: Path) -> tuple[bool, str]:
    """Validate a scores.json file against the schema."""
    try:
        data = load_json(file_path)
        if not isinstance(data, list):
            return False, "scores.json must be a list of score entries"
        for i, entry in enumerate(data):
            try:
                ScoreEntry(**entry)
            except ValidationError as e:
                return False, f"Entry {i}:\n{format_validation_error(e)}"
            except Exception as e:
                return False, f"Entry {i}: {e}"
        return True, "OK"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}"
    except Exception as e:
        return False, str(e)


def _validate_model_dirs(parent_dir: Path) -> tuple[int, int, list[str]]:
    """Validate all model directories under a given parent directory.

    Each model directory is expected to contain metadata.json and scores.json.

    Returns:
        Tuple of (passed_count, failed_count, error_messages)
    """
    passed = 0
    failed = 0
    errors = []

    for model_dir in sorted(parent_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        # Validate metadata.json
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            valid, msg = validate_metadata(metadata_file)
            if valid:
                passed += 1
            else:
                failed += 1
                errors.append(f"{metadata_file}: {msg}")
        else:
            failed += 1
            errors.append(f"{model_dir}: missing metadata.json")

        # Validate scores.json
        scores_file = model_dir / "scores.json"
        if scores_file.exists():
            valid, msg = validate_scores(scores_file)
            if valid:
                passed += 1
            else:
                failed += 1
                errors.append(f"{scores_file}: {msg}")
        else:
            failed += 1
            errors.append(f"{model_dir}: missing scores.json")

    return passed, failed, errors


def validate_results_directory(results_dir: Path) -> tuple[int, int, list[str]]:
    """Validate all JSON files in the results directory.

    Returns:
        Tuple of (passed_count, failed_count, error_messages)
    """
    if not results_dir.exists():
        return 0, 0, [f"Results directory not found: {results_dir}"]

    return _validate_model_dirs(results_dir)


def validate_alternative_agents_directory(alt_agents_dir: Path) -> tuple[int, int, list[str]]:
    """Validate all JSON files under the alternative_agents directory.

    Scans alternative_agents/{agent_type}/{model_name}/ directories and
    validates metadata.json and scores.json in each model directory.

    Returns:
        Tuple of (passed_count, failed_count, error_messages)
    """
    if not alt_agents_dir.exists():
        return 0, 0, []

    passed = 0
    failed = 0
    errors = []

    for agent_dir in sorted(alt_agents_dir.iterdir()):
        if not agent_dir.is_dir():
            continue

        p, f, e = _validate_model_dirs(agent_dir)
        passed += p
        failed += f
        errors.extend(e)

    return passed, failed, errors


def main():
    """Main entry point."""
    # First, check for duplicate keys in the validation script itself
    # This catches issues from git merges where duplicate entries might exist
    check_for_duplicate_dict_keys()

    # Determine results directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"
    alt_agents_dir = repo_root / "alternative_agents"

    # Allow override via command line argument
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])

    print("=" * 60)
    print("Schema Validation Report")
    print("=" * 60)
    print()

    # Validate results/ directory
    passed, failed, errors = validate_results_directory(results_dir)

    print(f"Results directory: {results_dir}")
    print(f"Files validated: {passed + failed}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print()

    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        print()

    # Validate alternative_agents/ directory
    alt_passed, alt_failed, alt_errors = validate_alternative_agents_directory(alt_agents_dir)

    if alt_passed + alt_failed > 0:
        print(f"Alternative agents directory: {alt_agents_dir}")
        print(f"Files validated: {alt_passed + alt_failed}")
        print(f"  Passed: {alt_passed}")
        print(f"  Failed: {alt_failed}")
        print()

        if alt_errors:
            print("Errors:")
            for error in alt_errors:
                print(f"  - {error}")
            print()

    total_failed = failed + alt_failed

    print("=" * 60)
    if total_failed == 0:
        print("VALIDATION PASSED")
        print("=" * 60)
        return 0
    else:
        print("VALIDATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
