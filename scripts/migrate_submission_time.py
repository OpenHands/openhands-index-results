#!/usr/bin/env python3
"""
Migrate submission_time from metadata.json to individual scores.json entries.

This script:
1. Checks all scores.json files to ensure each entry has submission_time
2. If any entry is missing submission_time, copies it from metadata.json
3. Removes submission_time from all metadata.json files
"""

import json
from pathlib import Path


def migrate_submission_time(results_dir: Path) -> None:
    """Migrate submission_time from metadata to scores for all models.
    
    Args:
        results_dir: Path to the results directory containing model subdirectories
    """
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    models_updated = 0
    scores_updated = 0
    metadata_updated = 0
    
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        metadata_file = model_dir / "metadata.json"
        scores_file = model_dir / "scores.json"
        
        if not metadata_file.exists() or not scores_file.exists():
            print(f"Skipping {model_dir.name}: missing metadata.json or scores.json")
            continue
        
        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Check if metadata has submission_time
        if "submission_time" not in metadata:
            print(f"Skipping {model_dir.name}: no submission_time in metadata.json")
            continue
        
        metadata_submission_time = metadata["submission_time"]
        
        # Load scores
        with open(scores_file) as f:
            scores = json.load(f)
        
        # Check each score entry for submission_time
        scores_modified = False
        for i, score_entry in enumerate(scores):
            if "submission_time" not in score_entry:
                print(f"  Adding submission_time to {model_dir.name}/scores.json entry {i}")
                score_entry["submission_time"] = metadata_submission_time
                scores_modified = True
                scores_updated += 1
        
        # Save scores if modified
        if scores_modified:
            with open(scores_file, "w") as f:
                json.dump(scores, f, indent=2)
                f.write("\n")
            print(f"  Updated {model_dir.name}/scores.json")
        
        # Remove submission_time from metadata
        del metadata["submission_time"]
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
        print(f"  Removed submission_time from {model_dir.name}/metadata.json")
        
        models_updated += 1
        metadata_updated += 1
    
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Models processed: {models_updated}")
    print(f"Score entries updated: {scores_updated}")
    print(f"Metadata files updated: {metadata_updated}")
    print("=" * 60)


def main():
    """Main entry point."""
    # Determine results directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"
    
    print("=" * 60)
    print("Migrating submission_time from metadata to scores")
    print("=" * 60)
    print(f"Results directory: {results_dir}\n")
    
    migrate_submission_time(results_dir)


if __name__ == "__main__":
    main()
