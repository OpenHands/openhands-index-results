"""Tests for the monitor_infer_progress script."""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from monitor_infer_progress import collect_progress, count_lines, write_progress


class TestCountLines:
    """Tests for count_lines function."""

    def test_empty_file(self, tmp_path):
        """Test counting lines in empty file."""
        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("")
        assert count_lines(file_path) == 0

    def test_single_line(self, tmp_path):
        """Test counting lines in file with single line."""
        file_path = tmp_path / "single.jsonl"
        file_path.write_text('{"key": "value"}\n')
        assert count_lines(file_path) == 1

    def test_multiple_lines(self, tmp_path):
        """Test counting lines in file with multiple lines."""
        file_path = tmp_path / "multi.jsonl"
        file_path.write_text('{"line": 1}\n{"line": 2}\n{"line": 3}\n')
        assert count_lines(file_path) == 3

    def test_nonexistent_file(self, tmp_path):
        """Test counting lines in nonexistent file returns 0."""
        file_path = tmp_path / "nonexistent.jsonl"
        assert count_lines(file_path) == 0


class TestCollectProgress:
    """Tests for collect_progress function."""

    def test_all_files_exist(self, tmp_path):
        """Test collecting progress when all files exist."""
        # Create test files with different line counts
        (tmp_path / "output.jsonl").write_text("line1\nline2\nline3\n")
        (tmp_path / "output.critic_attempt_1.jsonl").write_text("line1\nline2\n")
        (tmp_path / "output.critic_attempt_2.jsonl").write_text("line1\n")
        (tmp_path / "output.critic_attempt_3.jsonl").write_text("")

        progress = collect_progress(tmp_path)

        assert "timestamp" in progress
        assert isinstance(progress["timestamp"], datetime)
        assert "counts" in progress
        assert progress["counts"]["output.jsonl"] == 3
        assert progress["counts"]["output.critic_attempt_1.jsonl"] == 2
        assert progress["counts"]["output.critic_attempt_2.jsonl"] == 1
        assert progress["counts"]["output.critic_attempt_3.jsonl"] == 0

    def test_missing_files(self, tmp_path):
        """Test collecting progress when files don't exist."""
        progress = collect_progress(tmp_path)

        assert "timestamp" in progress
        assert "counts" in progress
        assert progress["counts"]["output.jsonl"] == 0
        assert progress["counts"]["output.critic_attempt_1.jsonl"] == 0
        assert progress["counts"]["output.critic_attempt_2.jsonl"] == 0
        assert progress["counts"]["output.critic_attempt_3.jsonl"] == 0

    def test_partial_files(self, tmp_path):
        """Test collecting progress when only some files exist."""
        (tmp_path / "output.jsonl").write_text("line1\nline2\n")
        (tmp_path / "output.critic_attempt_2.jsonl").write_text("line1\n")

        progress = collect_progress(tmp_path)

        assert progress["counts"]["output.jsonl"] == 2
        assert progress["counts"]["output.critic_attempt_1.jsonl"] == 0
        assert progress["counts"]["output.critic_attempt_2.jsonl"] == 1
        assert progress["counts"]["output.critic_attempt_3.jsonl"] == 0


class TestWriteProgress:
    """Tests for write_progress function."""

    def test_write_single_record(self, tmp_path):
        """Test writing a single progress record."""
        progress = {
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            "counts": {
                "output.jsonl": 100,
                "output.critic_attempt_1.jsonl": 75,
                "output.critic_attempt_2.jsonl": 50,
                "output.critic_attempt_3.jsonl": 25,
            },
        }

        write_progress(tmp_path, progress)

        progress_file = tmp_path / "inference_progress.txt"
        assert progress_file.exists()

        content = progress_file.read_text()
        expected = "2024-01-15 10:30:00 UTC, 100, 75, 50, 25\n"
        assert content == expected

    def test_append_multiple_records(self, tmp_path):
        """Test appending multiple progress records."""
        progress1 = {
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            "counts": {
                "output.jsonl": 100,
                "output.critic_attempt_1.jsonl": 75,
                "output.critic_attempt_2.jsonl": 50,
                "output.critic_attempt_3.jsonl": 25,
            },
        }
        progress2 = {
            "timestamp": datetime(2024, 1, 15, 10, 40, 0, tzinfo=timezone.utc),
            "counts": {
                "output.jsonl": 150,
                "output.critic_attempt_1.jsonl": 100,
                "output.critic_attempt_2.jsonl": 75,
                "output.critic_attempt_3.jsonl": 50,
            },
        }

        write_progress(tmp_path, progress1)
        write_progress(tmp_path, progress2)

        progress_file = tmp_path / "inference_progress.txt"
        lines = progress_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "2024-01-15 10:30:00 UTC, 100, 75, 50, 25"
        assert lines[1] == "2024-01-15 10:40:00 UTC, 150, 100, 75, 50"

    def test_zero_counts(self, tmp_path):
        """Test writing progress with zero counts."""
        progress = {
            "timestamp": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "counts": {
                "output.jsonl": 0,
                "output.critic_attempt_1.jsonl": 0,
                "output.critic_attempt_2.jsonl": 0,
                "output.critic_attempt_3.jsonl": 0,
            },
        }

        write_progress(tmp_path, progress)

        progress_file = tmp_path / "inference_progress.txt"
        content = progress_file.read_text()
        expected = "2024-01-15 10:00:00 UTC, 0, 0, 0, 0\n"
        assert content == expected


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_end_to_end(self, tmp_path):
        """Test complete workflow: create files, collect, and write."""
        # Setup: Create output files
        (tmp_path / "output.jsonl").write_text("line1\nline2\nline3\n")
        (tmp_path / "output.critic_attempt_1.jsonl").write_text("line1\nline2\n")
        (tmp_path / "output.critic_attempt_2.jsonl").write_text("line1\n")
        (tmp_path / "output.critic_attempt_3.jsonl").write_text("")

        # Collect and write
        progress = collect_progress(tmp_path)
        write_progress(tmp_path, progress)

        # Verify
        progress_file = tmp_path / "inference_progress.txt"
        assert progress_file.exists()

        content = progress_file.read_text()
        # Should have format: timestamp, 3, 2, 1, 0
        parts = content.strip().split(", ")
        assert len(parts) == 5
        assert "UTC" in parts[0]
        assert parts[1] == "3"
        assert parts[2] == "2"
        assert parts[3] == "1"
        assert parts[4] == "0"
