#!/usr/bin/env python3
"""
Validate JSON files against Pydantic schemas.

This script checks that all metadata.json and scores.json files
in the results directory conform to the expected schema.
"""

import json
import re
import sys
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

SEMVER_PATTERN = re.compile(r'^v\d+\.\d+\.\d+$')
DIRECTORY_NAME_PATTERN = re.compile(r'^v\d+\.\d+\.\d+_.+$')
# Model name pattern: alphanumeric with dots, hyphens, and underscores
# Examples: gpt-5.2, claude-4.5-sonnet, deepseek-v3.2-reasoner, qwen-3-coder
MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9._-]*$')


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


# Mapping of known models to their correct openness classification
# Open-weights models have publicly available model weights
# Closed API models only provide API access without weight availability
# Note: Unknown models are not validated against this mapping
MODEL_OPENNESS_MAP: dict[str, Openness] = {
    # Closed API models
    "claude-4.5-opus": Openness.CLOSED_API_AVAILABLE,
    "claude-4.5-sonnet": Openness.CLOSED_API_AVAILABLE,
    "gemini-3-pro": Openness.CLOSED_API_AVAILABLE,
    "gemini-3-flash": Openness.CLOSED_API_AVAILABLE,
    "gpt-5.2": Openness.CLOSED_API_AVAILABLE,
    "gpt-5.2-codex": Openness.CLOSED_API_AVAILABLE,
    # Open-weights models
    "kimi-k2-thinking": Openness.OPEN_WEIGHTS,
    "minimax-m2.1": Openness.OPEN_WEIGHTS,
    "deepseek-v3.2-reasoner": Openness.OPEN_WEIGHTS,
    "qwen-3-coder": Openness.OPEN_WEIGHTS,
}


# Known closed models where parameter count is not publicly known
# Note: Unknown models default to requiring parameter_count_b
CLOSED_MODELS: set[str] = {
    "claude-4.5-opus",
    "claude-4.5-sonnet",
    "gemini-3-pro",
    "gemini-3-flash",
    "gpt-5.2",
    "gpt-5.2-codex",
}


# Mapping of known models to their country of origin
# Note: Unknown models are not validated against this mapping
MODEL_COUNTRY_MAP: dict[str, Country] = {
    # US models
    "claude-4.5-opus": Country.US,
    "claude-4.5-sonnet": Country.US,
    "gemini-3-pro": Country.US,
    "gemini-3-flash": Country.US,
    "gpt-5.2": Country.US,
    "gpt-5.2-codex": Country.US,
    # China models
    "kimi-k2-thinking": Country.CN,
    "minimax-m2.1": Country.CN,
    "deepseek-v3.2-reasoner": Country.CN,
    "qwen-3-coder": Country.CN,
}


class Metadata(BaseModel):
    """Schema for metadata.json files."""
    agent_name: str = Field(..., description="Name of the agent")
    agent_version: str = Field(..., description="Version of the agent (semantic version starting with 'v')")
    model: str = Field(..., description="Model name (alphanumeric with dots, hyphens, and underscores)")
    openness: Openness = Field(..., description="Model openness classification")
    country: Country = Field(..., description="Country of origin for the model")
    tool_usage: ToolUsage = Field(..., description="Tool usage classification")
    submission_time: datetime = Field(..., description="Submission timestamp")
    directory_name: str = Field(..., description="Directory name for this result")
    release_date: date = Field(..., description="Model release date (YYYY-MM-DD)")
    parameter_count_b: Optional[float] = Field(None, description="Total model parameter count in billions. Required for open-weights models.")
    active_parameter_count_b: Optional[float] = Field(None, description="Active parameter count in billions (for MoE models)")

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

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name follows the expected pattern."""
        if not MODEL_NAME_PATTERN.match(v):
            raise ValueError(
                f"model name must start with a letter and contain only alphanumeric "
                f"characters, dots, hyphens, and underscores, got '{v}'"
            )
        return v

    @field_validator("openness")
    @classmethod
    def validate_openness_matches_model(cls, v: Openness, info) -> Openness:
        """Ensure openness matches the expected value for known models."""
        model = info.data.get("model")
        if model and model in MODEL_OPENNESS_MAP:
            expected_openness = MODEL_OPENNESS_MAP[model]
            if v != expected_openness:
                raise ValueError(
                    f"Model '{model}' should have openness '{expected_openness.value}', "
                    f"but got '{v.value}'"
                )
        return v

    @field_validator("country")
    @classmethod
    def validate_country_matches_model(cls, v: Country, info) -> Country:
        """Ensure country matches the expected value for known models."""
        model = info.data.get("model")
        if model and model in MODEL_COUNTRY_MAP:
            expected_country = MODEL_COUNTRY_MAP[model]
            if v != expected_country:
                raise ValueError(
                    f"Model '{model}' should have country '{expected_country.value}', "
                    f"but got '{v.value}'"
                )
        return v

    @field_validator("directory_name")
    @classmethod
    def validate_directory_name(cls, v: str, info) -> str:
        """Ensure directory_name follows the format {version}_{model_name}."""
        if not DIRECTORY_NAME_PATTERN.match(v):
            raise ValueError(
                f"directory_name must follow the format '{{version}}_{{model_name}}' "
                f"(e.g., 'v1.8.3_claude-4.5-sonnet'), got '{v}'"
            )
        # Validate that directory_name matches agent_version and model
        agent_version = info.data.get("agent_version")
        model = info.data.get("model")
        if agent_version and model:
            expected_dir_name = f"{agent_version}_{model}"
            if v != expected_dir_name:
                raise ValueError(
                    f"directory_name '{v}' does not match expected format "
                    f"'{{agent_version}}_{{model}}' = '{expected_dir_name}'"
                )
        return v

    @model_validator(mode='after')
    def validate_parameter_count_for_open_models(self):
        """Ensure parameter_count_b is provided for open-weights models.
        
        For known models, check against CLOSED_MODELS set.
        For unknown models, check the openness field - if it's 'open_weights',
        parameter_count_b is required.
        """
        is_known_closed = self.model in CLOSED_MODELS
        is_declared_closed = self.openness in (Openness.CLOSED_API_AVAILABLE, Openness.CLOSED)
        
        # Require parameter_count_b only for open-weights models
        if not is_known_closed and not is_declared_closed and self.parameter_count_b is None:
            raise ValueError(
                f"parameter_count_b is required for open-weights model '{self.model}'"
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


class ScoreEntry(BaseModel):
    """Schema for individual score entries in scores.json."""
    benchmark: Benchmark = Field(..., description="Benchmark name")
    score: float = Field(..., ge=0, le=100, description="Score value (0-100)")
    metric: Metric = Field(..., description="Metric type for the score")
    cost_per_instance: float = Field(..., gt=0, description="Average cost per problem in USD")
    average_runtime: float = Field(..., gt=0, description="Average runtime per instance in seconds")
    full_archive: str = Field(..., description="URL to the full evaluation archive")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

    @field_validator("full_archive")
    @classmethod
    def validate_full_archive(cls, v: str) -> str:
        """Ensure full_archive URL starts with the expected CDN prefix."""
        if not v.startswith(FULL_ARCHIVE_URL_PREFIX):
            raise ValueError(
                f"full_archive must begin with '{FULL_ARCHIVE_URL_PREFIX}', got '{v}'"
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
    except Exception as e:
        return False, str(e)


def validate_scores(file_path: Path) -> tuple[bool, str]:
    """Validate a scores.json file against the schema."""
    try:
        data = load_json(file_path)
        if not isinstance(data, list):
            return False, "scores.json must be a list"
        for i, entry in enumerate(data):
            try:
                ScoreEntry(**entry)
            except Exception as e:
                return False, f"Entry {i}: {e}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def validate_results_directory(results_dir: Path) -> tuple[int, int, list[str]]:
    """Validate all JSON files in the results directory.

    Returns:
        Tuple of (passed_count, failed_count, error_messages)
    """
    passed = 0
    failed = 0
    errors = []

    if not results_dir.exists():
        return 0, 0, [f"Results directory not found: {results_dir}"]

    for model_dir in sorted(results_dir.iterdir()):
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


def main():
    """Main entry point."""
    # Determine results directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"

    # Allow override via command line argument
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])

    print("=" * 60)
    print("Schema Validation Report")
    print("=" * 60)
    print()

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

    print("=" * 60)
    if failed == 0:
        print("VALIDATION PASSED")
        print("=" * 60)
        return 0
    else:
        print("VALIDATION FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
