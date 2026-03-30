# Fix for Issue #744: Date Column Empty in Summary Table

## Problem Description
The date column in the summary table on index.openhands.dev was appearing empty for some models, specifically:
- GPT-5.4
- Qwen3-Coder-Next

## Root Cause Analysis
After investigation, the issue was identified in the `simple_data_loader.py` file in the HuggingFace Space (https://huggingface.co/spaces/OpenHands/openhands-index).

### Background
Per PR #682 in this repository, the `submission_time` field was moved from `metadata.json` files to individual score entries in `scores.json` files. However, the data loader in the HuggingFace Space was still trying to read `submission_time` from the metadata, which caused empty dates for models where this field was correctly removed from metadata.

### Affected Code
File: `simple_data_loader.py` (in the HuggingFace Space)
Line: 165

**Before (incorrect):**
```python
'submission_time': metadata.get('submission_time', ''),
```

This reads `submission_time` from metadata, which returns an empty string for models that correctly don't have it in metadata.

**After (correct):**
```python
'submission_time': score_entry.get('submission_time', metadata.get('submission_time', '')),
```

This first tries to read `submission_time` from the individual score entry (the new standard), and only falls back to metadata for backward compatibility with older entries.

## Solution
The fix is to update line 165 of `simple_data_loader.py` in the HuggingFace Space to read `submission_time` from score entries instead of metadata.

### How to Apply
1. Access the HuggingFace Space at https://huggingface.co/spaces/OpenHands/openhands-index
2. Edit the file `simple_data_loader.py`
3. Update line 165 as shown above
4. Commit the change with the message: "Fix empty date column by using per-score submission_time"

Alternatively, apply the patch file `fix-empty-date-column.patch` included in this repository:
```bash
cd /path/to/openhands-index-space
git apply /path/to/fix-empty-date-column.patch
```

## Verification
After applying the fix, all models should show their submission dates in the Date column on:
- https://index.openhands.dev/issue-resolution
- https://index.openhands.dev/greenfield
- https://index.openhands.dev/frontend
- https://index.openhands.dev/testing
- https://index.openhands.dev/information-gathering

## Testing
To test the fix locally:
1. Clone the HuggingFace Space
2. Apply the fix
3. Run the test script included in this directory: `test_date_column_fix.py`

The test verifies that:
- All models have non-empty date values
- The date format is correct (ISO 8601)
- Per-score submission times are prioritized over metadata submission times

## Related Issues
- Closes #744
- Related to PR #682 (which moved submission_time from metadata to scores)
