#!/usr/bin/env python3
"""
Update tests to remove submission_time from metadata test cases.

This script removes submission_time lines ONLY from metadata test dictionaries
in the TestMetadataSchema class (approximately lines 21-528).
"""

import re
from pathlib import Path


def update_test_file(test_file: Path) -> None:
    """Remove submission_time from metadata tests only.
    
    Args:
        test_file: Path to test_validate_schema.py
    """
    with open(test_file) as f:
        lines = f.readlines()
    
    # Find the boundaries of TestMetadataSchema class
    # This class starts around line 21 and ends before TestScoreEntrySchema (around line 529)
    in_metadata_class = False
    metadata_class_start = None
    metadata_class_end = None
    
    for i, line in enumerate(lines):
        if 'class TestMetadataSchema:' in line:
            in_metadata_class = True
            metadata_class_start = i
        elif in_metadata_class and 'class TestScoreEntrySchema:' in line:
            metadata_class_end = i
            break
    
    if metadata_class_start is None or metadata_class_end is None:
        print("Could not find TestMetadataSchema or TestScoreEntrySchema classes")
        return
    
    print(f"TestMetadataSchema class: lines {metadata_class_start+1} to {metadata_class_end}")
    
    # Remove submission_time lines only within the TestMetadataSchema class
    updated_lines = []
    for i, line in enumerate(lines):
        # Only remove submission_time if we're inside the metadata class
        if metadata_class_start <= i < metadata_class_end and '"submission_time"' in line:
            print(f"  Removing line {i+1}: {line.strip()}")
            continue
        updated_lines.append(line)
    
    # Write back
    with open(test_file, "w") as f:
        f.writelines(updated_lines)
    
    print(f"Updated {test_file}")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    test_file = repo_root / "tests" / "test_validate_schema.py"
    
    print("=" * 60)
    print("Updating tests to remove submission_time from metadata")
    print("=" * 60)
    print(f"Test file: {test_file}\n")
    
    update_test_file(test_file)
    
    print("=" * 60)
    print("Update complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
