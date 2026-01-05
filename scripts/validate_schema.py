#!/usr/bin/env python3
"""
Validate JSON files against Pydantic schemas.

This script checks that all metadata.json and scores.json files
in the results directory conform to the expected schema.
"""

import json
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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


class Metadata(BaseModel):
    """Schema for metadata.json files."""
    agent_name: str = Field(..., description="Name of the agent")
    agent_version: str = Field(..., description="Version of the agent")
    model: str = Field(..., description="Model name")
    openness: Openness = Field(..., description="Model openness classification")
    tool_usage: ToolUsage = Field(..., description="Tool usage classification")
    submission_time: datetime = Field(..., description="Submission timestamp")
    directory_name: str = Field(..., description="Directory name for this result")


class Benchmark(str, Enum):
    """Expected benchmark names."""
    SWE_BENCH = "swe-bench"
    SWE_BENCH_MULTIMODAL = "swe-bench-multimodal"
    MULTI_SWE_BENCH = "multi-swe-bench"
    SWT_BENCH = "swt-bench"
    COMMIT0 = "commit0"
    GAIA = "gaia"


class Metric(str, Enum):
    """Expected metric names for the score field."""
    ACCURACY = "accuracy"


class ScoreEntry(BaseModel):
    """Schema for individual score entries in scores.json."""
    benchmark: Benchmark = Field(..., description="Benchmark name")
    score: float = Field(..., ge=0, le=100, description="Score value (0-100)")
    metric: Metric = Field(..., description="Metric type for the score")
    total_cost: Optional[float] = Field(None, ge=0, description="Total cost in USD")
    # total_runtime is retained for backward compatibility; prefer average_runtime going forward.
    total_runtime: Optional[float] = Field(None, ge=0, description="Total runtime in seconds")
    average_runtime: Optional[float] = Field(
        None, ge=0, description="Average runtime in seconds (preferred)"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

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
