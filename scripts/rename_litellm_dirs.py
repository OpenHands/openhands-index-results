#!/usr/bin/env python3
"""
Rename model directories to remove litellm prefix and suffix.

This script:
1. Identifies folders with litellm prefix (e.g., 202601_litellm_proxy-...)
2. Extracts the date prefix and model name from metadata.json
3. Renames the folder to use date_model format
4. Updates the directory_name field in metadata.json
5. If a folder with the target name already exists, merges scores.json
   and keeps the metadata of the latest submission
"""

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def parse_submission_time(time_str: str) -> datetime:
    """Parse submission time from metadata, handling various formats."""
    # Try ISO format with timezone
    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        pass
    # Try without timezone
    try:
        return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        pass
    # Try basic format
    return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")


def load_json(file_path: Path) -> dict | list:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def save_json(file_path: Path, data: dict | list) -> None:
    """Save JSON file with pretty formatting."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def is_litellm_dir(dir_name: str) -> bool:
    """Check if directory name contains litellm prefix."""
    return "litellm_proxy" in dir_name


def extract_date_prefix(dir_name: str) -> str | None:
    """Extract date prefix (e.g., '202601') from directory name."""
    match = re.match(r"^(\d{6})_", dir_name)
    return match.group(1) if match else None


def get_new_dir_name(date_prefix: str, model_name: str) -> str:
    """Generate new directory name from date prefix and model name."""
    return f"{date_prefix}_{model_name}"


def merge_scores(existing_scores: list, new_scores: list) -> list:
    """Merge two score lists, avoiding duplicates based on benchmark."""
    # Create a dict keyed by benchmark for existing scores
    merged = {entry["benchmark"]: entry for entry in existing_scores}
    # Add/update with new scores
    for entry in new_scores:
        merged[entry["benchmark"]] = entry
    return list(merged.values())


def rename_litellm_directories(results_dir: Path, dry_run: bool = False) -> list[dict]:
    """
    Rename litellm directories to use model name from metadata.

    Args:
        results_dir: Path to the results directory
        dry_run: If True, only report what would be done without making changes

    Returns:
        List of operations performed (or would be performed in dry_run mode)
    """
    operations = []

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return operations

    # Find all litellm directories
    litellm_dirs = []
    for model_dir in sorted(results_dir.iterdir()):
        if model_dir.is_dir() and is_litellm_dir(model_dir.name):
            litellm_dirs.append(model_dir)

    for litellm_dir in litellm_dirs:
        dir_name = litellm_dir.name
        date_prefix = extract_date_prefix(dir_name)

        if not date_prefix:
            print(f"Warning: Could not extract date prefix from {dir_name}")
            continue

        # Load metadata to get model name
        metadata_file = litellm_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"Warning: No metadata.json in {dir_name}")
            continue

        metadata = load_json(metadata_file)
        model_name = metadata.get("model")
        if not model_name:
            print(f"Warning: No model field in metadata for {dir_name}")
            continue

        # Generate new directory name
        new_dir_name = get_new_dir_name(date_prefix, model_name)
        new_dir_path = results_dir / new_dir_name

        operation = {
            "source": dir_name,
            "target": new_dir_name,
            "model": model_name,
            "conflict": new_dir_path.exists(),
        }

        if new_dir_path.exists():
            # Handle conflict - merge scores and keep latest metadata
            existing_metadata = load_json(new_dir_path / "metadata.json")
            existing_scores = load_json(new_dir_path / "scores.json")
            new_scores = load_json(litellm_dir / "scores.json")

            # Determine which metadata is newer
            existing_time = parse_submission_time(existing_metadata["submission_time"])
            new_time = parse_submission_time(metadata["submission_time"])

            if new_time > existing_time:
                # Use new metadata (update directory_name)
                final_metadata = metadata.copy()
                final_metadata["directory_name"] = new_dir_name
                operation["kept_metadata"] = "new"
            else:
                # Keep existing metadata
                final_metadata = existing_metadata
                operation["kept_metadata"] = "existing"

            # Merge scores
            merged_scores = merge_scores(existing_scores, new_scores)
            operation["merged_scores"] = len(merged_scores)

            if not dry_run:
                # Save merged data to existing directory
                save_json(new_dir_path / "metadata.json", final_metadata)
                save_json(new_dir_path / "scores.json", merged_scores)
                # Remove the litellm directory
                shutil.rmtree(litellm_dir)
        else:
            # No conflict - simple rename
            if not dry_run:
                # Update metadata first
                metadata["directory_name"] = new_dir_name
                save_json(metadata_file, metadata)
                # Rename directory
                litellm_dir.rename(new_dir_path)

        operations.append(operation)

    return operations


def main():
    """Main entry point."""
    # Determine results directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"

    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv

    # Allow override via command line argument
    for arg in sys.argv[1:]:
        if arg != "--dry-run" and not arg.startswith("-"):
            results_dir = Path(arg)
            break

    print("=" * 60)
    print("Rename LiteLLM Directories")
    print("=" * 60)
    print()
    print(f"Results directory: {results_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    operations = rename_litellm_directories(results_dir, dry_run=dry_run)

    if not operations:
        print("No litellm directories found to rename.")
        return 0

    print("Operations:")
    for op in operations:
        if op["conflict"]:
            print(f"  MERGE: {op['source']} -> {op['target']}")
            print(f"         Kept metadata: {op['kept_metadata']}")
            print(f"         Merged scores: {op['merged_scores']} entries")
        else:
            print(f"  RENAME: {op['source']} -> {op['target']}")
    print()

    print("=" * 60)
    print(f"Total operations: {len(operations)}")
    print(f"  Renames: {sum(1 for op in operations if not op['conflict'])}")
    print(f"  Merges: {sum(1 for op in operations if op['conflict'])}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
