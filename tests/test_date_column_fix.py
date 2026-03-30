#!/usr/bin/env python3
"""
Test to verify that the date column fix works correctly.
This test checks that all agents have non-empty submission dates.
"""

import sys
from pathlib import Path


def test_all_scores_have_submission_time():
    """
    Test that all score entries have a submission_time field.
    This is the key requirement after PR #682.
    """
    import json
    
    results_dir = Path(__file__).parent.parent / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return False
    
    errors = []
    checked_count = 0
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        scores_file = model_dir / "scores.json"
        if not scores_file.exists():
            errors.append(f"Missing scores.json in {model_dir.name}")
            continue
        
        with open(scores_file) as f:
            scores = json.load(f)
        
        for i, score_entry in enumerate(scores):
            checked_count += 1
            benchmark = score_entry.get('benchmark', 'unknown')
            
            if 'submission_time' not in score_entry:
                errors.append(
                    f"{model_dir.name}/scores.json entry {i} (benchmark: {benchmark}): "
                    f"missing submission_time field"
                )
            elif not score_entry['submission_time']:
                errors.append(
                    f"{model_dir.name}/scores.json entry {i} (benchmark: {benchmark}): "
                    f"submission_time is empty"
                )
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"✅ All {checked_count} score entries have valid submission_time fields")
    return True


def test_metadata_submission_time_not_required():
    """
    Test that metadata.json files don't require submission_time.
    Per PR #682, submission_time was moved to scores.json.
    """
    import json
    
    results_dir = Path(__file__).parent.parent / "results"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return False
    
    without_submission_time = []
    with_submission_time = []
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        metadata_file = model_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        if 'submission_time' in metadata:
            with_submission_time.append(model_dir.name)
        else:
            without_submission_time.append(model_dir.name)
    
    print(f"\n📊 Metadata submission_time status:")
    print(f"  - With submission_time: {len(with_submission_time)} models")
    print(f"  - Without submission_time: {len(without_submission_time)} models")
    
    if without_submission_time:
        print(f"  - Models without (correct per PR #682): {', '.join(without_submission_time)}")
    
    # This is not an error - both states are valid during migration
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Date Column Fix")
    print("=" * 60)
    
    all_passed = True
    
    print("\nTest 1: Checking all score entries have submission_time...")
    if not test_all_scores_have_submission_time():
        all_passed = False
    
    print("\nTest 2: Checking metadata.json submission_time status...")
    if not test_metadata_submission_time_not_required():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        print("=" * 60)
        sys.exit(1)
