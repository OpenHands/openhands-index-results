#!/usr/bin/env python3
"""
Validate JSON files against Pydantic schemas.

This script checks that all metadata.json and scores.json files
in the results directory conform to the expected schema.
"""

import json
import re
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

SEMVER_PATTERN = re.compile(r'^v\d+\.\d+\.\d+$')
DIRECTORY_NAME_PATTERN = re.compile(r'^v\d+\.\d+\.\d+_.+$')


class Openness(str, Enum):
    """Model openness classification."""
    OPEN_WEIGHTS = "open_weights"
    CLOSED_API_AVAILABLE = "closed_api_available"
    CLOSED = "closed"


class ToolUsage(str, Enum):
    """Tool usage classification."""
    STANDARD = "standard"
    CUSTOM = "custom"
    NONE = "none"


class Model(str, Enum):
    """Expected model names from issue #2."""
    CLAUDE_4_5_OPUS = "claude-4.5-opus"
    CLAUDE_4_5_SONNET = "claude-4.5-sonnet"
    GEMINI_3_PRO = "gemini-3-pro"
    GEMINI_3_FLASH = "gemini-3-flash"
    GPT_5_2 = "gpt-5.2"
    KIMI_K2_THINKING = "kimi-k2-thinking"
    MINIMAX_M2_1 = "minimax-m2.1"
    DEEPSEEK_V3_2_REASONER = "deepseek-v3.2-reasoner"
    QWEN_3_CODER = "qwen-3-coder"


# Mapping of models to their correct openness classification
# Open-weights models have publicly available model weights
# Closed API models only provide API access without weight availability
MODEL_OPENNESS_MAP: dict[Model, Openness] = {
    # Closed API models
    Model.CLAUDE_4_5_OPUS: Openness.CLOSED_API_AVAILABLE,
    Model.CLAUDE_4_5_SONNET: Openness.CLOSED_API_AVAILABLE,
    Model.GEMINI_3_PRO: Openness.CLOSED_API_AVAILABLE,
    Model.GEMINI_3_FLASH: Openness.CLOSED_API_AVAILABLE,
    Model.GPT_5_2: Openness.CLOSED_API_AVAILABLE,
    # Open-weights models
    Model.KIMI_K2_THINKING: Openness.OPEN_WEIGHTS,
    Model.MINIMAX_M2_1: Openness.OPEN_WEIGHTS,
    Model.DEEPSEEK_V3_2_REASONER: Openness.OPEN_WEIGHTS,
    Model.QWEN_3_CODER: Openness.OPEN_WEIGHTS,
}


class Metadata(BaseModel):
    """Schema for metadata.json files."""
    agent_name: str = Field(..., description="Name of the agent")
    agent_version: str = Field(..., description="Version of the agent (semantic version starting with 'v')")
    model: Model = Field(..., description="Model name (must be one of the expected models)")
    openness: Openness = Field(..., description="Model openness classification")
    tool_usage: ToolUsage = Field(..., description="Tool usage classification")
    submission_time: datetime = Field(..., description="Submission timestamp")
    directory_name: str = Field(..., description="Directory name for this result")

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
            expected_dir_name = f"{agent_version}_{model.value}"
            if v != expected_dir_name:
                raise ValueError(
                    f"directory_name '{v}' does not match expected format "
                    f"'{{agent_version}}_{{model}}' = '{expected_dir_name}'"
                )
        return v


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
