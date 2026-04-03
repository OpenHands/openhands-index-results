#!/usr/bin/env python3
"""
Generate complete-models.csv file with models that have completed all benchmarks.

This script finds all scores.json files that contain exactly 5 benchmark entries
and creates a CSV file with the timestamp and path to each complete model.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def get_git_last_modified_time(file_path: Path) -> datetime:
    """Get the last modified time of a file from git history in UTC."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", str(file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        timestamp_str = result.stdout.strip()
        if not timestamp_str:
            # File might not be in git yet, use current time
            return datetime.now(timezone.utc).replace(tzinfo=None)
        # Parse ISO format timestamp and convert to UTC
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except subprocess.CalledProcessError:
        # If git command fails, use current time
        return datetime.now(timezone.utc).replace(tzinfo=None)


def find_complete_models() -> List[Tuple[datetime, str]]:
    """
    Find all models with complete benchmark results (exactly 5 entries).
    
    Returns:
        List of tuples (timestamp, model_path) for complete models.
    """
    repo_root = Path(__file__).parent.parent
    complete_models = []
    
    # Find all scores.json files
    scores_files = list(repo_root.glob("**/scores.json"))
    
    for scores_file in scores_files:
        try:
            with open(scores_file, 'r') as f:
                data = json.load(f)
            
            # Check if this scores.json has exactly 5 entries
            if isinstance(data, list) and len(data) == 5:
                # Get the last modified timestamp
                timestamp = get_git_last_modified_time(scores_file)
                
                # Get the relative path to the directory containing scores.json
                model_dir = scores_file.parent.relative_to(repo_root)
                
                complete_models.append((timestamp, str(model_dir)))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not process {scores_file}: {e}")
            continue
    
    # Sort by timestamp (most recent first)
    complete_models.sort(reverse=True)
    
    return complete_models


def generate_csv(output_path: Path, complete_models: List[Tuple[datetime, str]]):
    """
    Generate the complete-models.csv file.
    
    Args:
        output_path: Path to the output CSV file
        complete_models: List of (timestamp, model_path) tuples
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write("timestamp,model-path\n")
        
        # Write data rows
        for timestamp, model_path in complete_models:
            # Format timestamp as YYYY-MM-DD-HH-MM-SS in UTC
            timestamp_str = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
            f.write(f"{timestamp_str},{model_path}\n")


def main():
    """Main function to generate complete-models.csv."""
    repo_root = Path(__file__).parent.parent
    output_path = repo_root / "complete-models.csv"
    
    print("Finding models with complete benchmark results...")
    complete_models = find_complete_models()
    
    print(f"Found {len(complete_models)} complete models")
    
    print(f"Generating {output_path}...")
    generate_csv(output_path, complete_models)
    
    print("Done!")


if __name__ == "__main__":
    main()
