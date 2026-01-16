"""Tests for the rename_litellm_dirs script."""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from rename_litellm_dirs import (
    is_litellm_dir,
    extract_date_prefix,
    get_new_dir_name,
    merge_scores,
    parse_submission_time,
    rename_litellm_directories,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_litellm_dir_true(self):
        """Test is_litellm_dir returns True for litellm directories."""
        assert is_litellm_dir("202601_litellm_proxy-deepseek-deepseek-reasoner") is True
        assert is_litellm_dir("202601_litellm_proxy-anthropic-claude-opus-4-5-20251101") is True

    def test_is_litellm_dir_false(self):
        """Test is_litellm_dir returns False for non-litellm directories."""
        assert is_litellm_dir("202601_gpt-5.2") is False
        assert is_litellm_dir("202601_claude-4.5-opus") is False

    def test_extract_date_prefix(self):
        """Test extract_date_prefix extracts correct date prefix."""
        assert extract_date_prefix("202601_litellm_proxy-deepseek") == "202601"
        assert extract_date_prefix("202510_gpt-5") == "202510"
        assert extract_date_prefix("invalid") is None

    def test_get_new_dir_name(self):
        """Test get_new_dir_name generates correct directory name."""
        assert get_new_dir_name("202601", "deepseek-v3.2-reasoner") == "202601_deepseek-v3.2-reasoner"
        assert get_new_dir_name("202510", "gpt-5.2") == "202510_gpt-5.2"

    def test_merge_scores_no_overlap(self):
        """Test merge_scores with no overlapping benchmarks."""
        existing = [{"benchmark": "swe-bench", "score": 70.0}]
        new = [{"benchmark": "commit0", "score": 50.0}]
        merged = merge_scores(existing, new)
        assert len(merged) == 2
        benchmarks = {s["benchmark"] for s in merged}
        assert benchmarks == {"swe-bench", "commit0"}

    def test_merge_scores_with_overlap(self):
        """Test merge_scores with overlapping benchmarks (new overwrites)."""
        existing = [{"benchmark": "swe-bench", "score": 70.0}]
        new = [{"benchmark": "swe-bench", "score": 75.0}]
        merged = merge_scores(existing, new)
        assert len(merged) == 1
        assert merged[0]["score"] == 75.0

    def test_parse_submission_time_iso_with_tz(self):
        """Test parsing ISO format with timezone."""
        dt = parse_submission_time("2026-01-14T19:58:10.754133+00:00")
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 14

    def test_parse_submission_time_iso_without_tz(self):
        """Test parsing ISO format without timezone."""
        dt = parse_submission_time("2025-11-24T19:56:00.092865")
        assert dt.year == 2025
        assert dt.month == 11
        assert dt.day == 24


class TestRenameLitellmDirectories:
    """Tests for rename_litellm_directories function."""

    def test_simple_rename(self, tmp_path):
        """Test simple rename of litellm directory."""
        # Create litellm directory
        litellm_dir = tmp_path / "202601_litellm_proxy-deepseek-deepseek-reasoner"
        litellm_dir.mkdir()

        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "1.0.0",
            "model": "deepseek-v3.2-reasoner",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2026-01-14T19:58:10.754133+00:00",
            "directory_name": "202601_litellm_proxy-deepseek-deepseek-reasoner"
        }
        scores = [{"benchmark": "commit0", "score": 33.3, "metric": "accuracy", "tags": []}]

        (litellm_dir / "metadata.json").write_text(json.dumps(metadata))
        (litellm_dir / "scores.json").write_text(json.dumps(scores))

        # Run rename
        operations = rename_litellm_directories(tmp_path, dry_run=False)

        # Verify
        assert len(operations) == 1
        assert operations[0]["source"] == "202601_litellm_proxy-deepseek-deepseek-reasoner"
        assert operations[0]["target"] == "202601_deepseek-v3.2-reasoner"
        assert operations[0]["conflict"] is False

        # Check directory was renamed
        assert not litellm_dir.exists()
        new_dir = tmp_path / "202601_deepseek-v3.2-reasoner"
        assert new_dir.exists()

        # Check metadata was updated
        new_metadata = json.loads((new_dir / "metadata.json").read_text())
        assert new_metadata["directory_name"] == "202601_deepseek-v3.2-reasoner"

    def test_merge_with_conflict(self, tmp_path):
        """Test merge when target directory already exists."""
        # Create existing directory
        existing_dir = tmp_path / "202601_claude-4.5-opus"
        existing_dir.mkdir()

        existing_metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "newer",
            "model": "claude-4.5-opus",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2026-01-15T21:54:45.442941+00:00",  # Newer
            "directory_name": "202601_claude-4.5-opus"
        }
        existing_scores = [{"benchmark": "swe-bench-multimodal", "score": 27.3, "metric": "accuracy", "tags": []}]

        (existing_dir / "metadata.json").write_text(json.dumps(existing_metadata))
        (existing_dir / "scores.json").write_text(json.dumps(existing_scores))

        # Create litellm directory
        litellm_dir = tmp_path / "202601_litellm_proxy-anthropic-claude-opus-4-5-20251101"
        litellm_dir.mkdir()

        litellm_metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "older",
            "model": "claude-4.5-opus",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2026-01-14T21:52:46.978168+00:00",  # Older
            "directory_name": "202601_litellm_proxy-anthropic-claude-opus-4-5-20251101"
        }
        litellm_scores = [
            {"benchmark": "commit0", "score": 50.0, "metric": "accuracy", "tags": []},
            {"benchmark": "gaia", "score": 69.1, "metric": "accuracy", "tags": []}
        ]

        (litellm_dir / "metadata.json").write_text(json.dumps(litellm_metadata))
        (litellm_dir / "scores.json").write_text(json.dumps(litellm_scores))

        # Run rename
        operations = rename_litellm_directories(tmp_path, dry_run=False)

        # Verify
        assert len(operations) == 1
        assert operations[0]["conflict"] is True
        assert operations[0]["kept_metadata"] == "existing"  # Newer metadata kept
        assert operations[0]["merged_scores"] == 3  # 1 existing + 2 new

        # Check litellm directory was removed
        assert not litellm_dir.exists()

        # Check merged scores
        merged_scores = json.loads((existing_dir / "scores.json").read_text())
        benchmarks = {s["benchmark"] for s in merged_scores}
        assert benchmarks == {"swe-bench-multimodal", "commit0", "gaia"}

        # Check metadata kept from newer
        final_metadata = json.loads((existing_dir / "metadata.json").read_text())
        assert final_metadata["agent_version"] == "newer"

    def test_dry_run_no_changes(self, tmp_path):
        """Test dry run does not make any changes."""
        # Create litellm directory
        litellm_dir = tmp_path / "202601_litellm_proxy-deepseek-deepseek-reasoner"
        litellm_dir.mkdir()

        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "1.0.0",
            "model": "deepseek-v3.2-reasoner",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2026-01-14T19:58:10.754133+00:00",
            "directory_name": "202601_litellm_proxy-deepseek-deepseek-reasoner"
        }
        scores = [{"benchmark": "commit0", "score": 33.3, "metric": "accuracy", "tags": []}]

        (litellm_dir / "metadata.json").write_text(json.dumps(metadata))
        (litellm_dir / "scores.json").write_text(json.dumps(scores))

        # Run dry run
        operations = rename_litellm_directories(tmp_path, dry_run=True)

        # Verify operations were reported
        assert len(operations) == 1

        # But directory was NOT renamed
        assert litellm_dir.exists()
        assert not (tmp_path / "202601_deepseek-v3.2-reasoner").exists()

    def test_no_litellm_dirs(self, tmp_path):
        """Test with no litellm directories."""
        # Create non-litellm directory
        normal_dir = tmp_path / "202601_gpt-5.2"
        normal_dir.mkdir()

        metadata = {
            "agent_name": "OpenHands CodeAct",
            "agent_version": "1.0.0",
            "model": "gpt-5.2",
            "openness": "closed_api_available",
            "tool_usage": "standard",
            "submission_time": "2026-01-15T23:17:38.752985+00:00",
            "directory_name": "202601_gpt-5.2"
        }
        scores = [{"benchmark": "swe-bench", "score": 74.6, "metric": "accuracy", "tags": []}]

        (normal_dir / "metadata.json").write_text(json.dumps(metadata))
        (normal_dir / "scores.json").write_text(json.dumps(scores))

        # Run rename
        operations = rename_litellm_directories(tmp_path, dry_run=False)

        # Verify no operations
        assert len(operations) == 0

        # Directory unchanged
        assert normal_dir.exists()


class TestIntegration:
    """Integration tests using the actual results directory."""

    def test_no_litellm_dirs_remain(self):
        """Test that no litellm directories remain after running the script."""
        repo_root = Path(__file__).parent.parent
        results_dir = repo_root / "results"

        if not results_dir.exists():
            return  # Skip if results directory does not exist

        # Check no litellm directories remain
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                assert not is_litellm_dir(model_dir.name), f"Found litellm directory: {model_dir.name}"

    def test_all_directory_names_match(self):
        """Test that all directory_name fields in metadata match actual directory names."""
        repo_root = Path(__file__).parent.parent
        results_dir = repo_root / "results"

        if not results_dir.exists():
            return  # Skip if results directory does not exist

        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    metadata = json.loads(metadata_file.read_text())
                    assert metadata.get("directory_name") == model_dir.name, \
                        f"Mismatch in {model_dir.name}: directory_name={metadata.get('directory_name')}"
