"""Tests for the generate_complete_models script."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_complete_models import (
    find_complete_models,
    generate_json,
    get_git_last_modified_time,
)


class TestGetGitLastModifiedTime:
    """Tests for get_git_last_modified_time function."""

    def test_valid_git_timestamp(self, tmp_path):
        """Test getting timestamp from git log."""
        test_file = tmp_path / "test.json"
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="2026-04-02T09:01:40-03:00\n",
                returncode=0
            )
            
            result = get_git_last_modified_time(test_file)
            
            assert isinstance(result, datetime)
            # Should be converted to UTC
            assert result.tzinfo is None

    def test_git_command_failure(self, tmp_path):
        """Test fallback when git command fails."""
        test_file = tmp_path / "test.json"
        
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            
            result = get_git_last_modified_time(test_file)
            
            # Should return current time as fallback
            assert isinstance(result, datetime)
            assert result.tzinfo is None

    def test_empty_git_output(self, tmp_path):
        """Test handling of empty git output."""
        test_file = tmp_path / "test.json"
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="",
                returncode=0
            )
            
            result = get_git_last_modified_time(test_file)
            
            # Should return current time as fallback
            assert isinstance(result, datetime)
            assert result.tzinfo is None


class TestFindCompleteModels:
    """Tests for find_complete_models function."""

    def test_find_models_with_five_entries(self, tmp_path, monkeypatch):
        """Test finding models with exactly 5 benchmark entries."""
        # Create test directory structure
        model1 = tmp_path / "model1"
        model1.mkdir()
        scores1 = model1 / "scores.json"
        
        # Model with 5 entries (complete)
        scores1.write_text(json.dumps([
            {"benchmark": "bench1", "score": 1.0},
            {"benchmark": "bench2", "score": 2.0},
            {"benchmark": "bench3", "score": 3.0},
            {"benchmark": "bench4", "score": 4.0},
            {"benchmark": "bench5", "score": 5.0}
        ]))
        
        # Model with 3 entries (incomplete)
        model2 = tmp_path / "model2"
        model2.mkdir()
        scores2 = model2 / "scores.json"
        scores2.write_text(json.dumps([
            {"benchmark": "bench1", "score": 1.0},
            {"benchmark": "bench2", "score": 2.0},
            {"benchmark": "bench3", "score": 3.0}
        ]))
        
        # Change to test directory
        monkeypatch.chdir(tmp_path)
        
        with patch("generate_complete_models.get_git_last_modified_time") as mock_git:
            mock_git.return_value = datetime(2026, 4, 2, 9, 1, 40)
            
            # Temporarily change __file__ to point to tmp_path
            with patch("generate_complete_models.Path") as mock_path:
                mock_path.return_value.parent.parent = tmp_path
                
                results = find_complete_models()
        
        # Should only find model1 (with 5 entries)
        assert len(results) == 1
        assert results[0][1] == "model1"

    def test_invalid_json_handling(self, tmp_path, monkeypatch):
        """Test handling of invalid JSON files."""
        model = tmp_path / "model"
        model.mkdir()
        scores = model / "scores.json"
        scores.write_text("invalid json")
        
        monkeypatch.chdir(tmp_path)
        
        with patch("generate_complete_models.Path") as mock_path:
            mock_path.return_value.parent.parent = tmp_path
            
            results = find_complete_models()
        
        # Should handle error gracefully and return empty list
        assert len(results) == 0

    def test_sorting_by_timestamp(self, tmp_path, monkeypatch):
        """Test that results are sorted by timestamp (most recent first)."""
        # Create two models with 5 entries each
        model1 = tmp_path / "model1"
        model1.mkdir()
        scores1 = model1 / "scores.json"
        scores1.write_text(json.dumps([{"b": i} for i in range(5)]))
        
        model2 = tmp_path / "model2"
        model2.mkdir()
        scores2 = model2 / "scores.json"
        scores2.write_text(json.dumps([{"b": i} for i in range(5)]))
        
        monkeypatch.chdir(tmp_path)
        
        with patch("generate_complete_models.get_git_last_modified_time") as mock_git:
            # model1 is older, model2 is newer
            def side_effect(path):
                if "model1" in str(path):
                    return datetime(2026, 4, 1, 10, 0, 0)
                else:
                    return datetime(2026, 4, 2, 10, 0, 0)
            
            mock_git.side_effect = side_effect
            
            with patch("generate_complete_models.Path") as mock_path:
                mock_path.return_value.parent.parent = tmp_path
                
                results = find_complete_models()
        
        # Should be sorted with most recent first
        assert len(results) == 2
        assert results[0][1] == "model2"  # newer
        assert results[1][1] == "model1"  # older


class TestGenerateJson:
    """Tests for generate_json function."""

    def test_json_generation(self, tmp_path):
        """Test JSON file generation with correct format."""
        output_file = tmp_path / "complete-models.json"
        
        test_data = [
            (datetime(2026, 4, 2, 9, 1, 40), "alternative_agents/model1"),
            (datetime(2026, 4, 1, 10, 30, 15), "results/model2"),
        ]
        
        generate_json(output_file, test_data)
        
        # Check file was created
        assert output_file.exists()
        
        # Check content
        content = output_file.read_text()
        data = json.loads(content)
        
        assert len(data) == 2
        assert data[0]["timestamp"] == "2026-04-02-09-01-40"
        assert data[0]["model-path"] == "alternative_agents/model1"
        assert data[1]["timestamp"] == "2026-04-01-10-30-15"
        assert data[1]["model-path"] == "results/model2"

    def test_empty_data(self, tmp_path):
        """Test JSON generation with no data."""
        output_file = tmp_path / "complete-models.json"
        
        generate_json(output_file, [])
        
        # Should create file with empty array
        assert output_file.exists()
        content = output_file.read_text()
        data = json.loads(content)
        assert data == []

    def test_timestamp_format(self, tmp_path):
        """Test that timestamps are formatted correctly."""
        output_file = tmp_path / "complete-models.json"
        
        test_data = [
            (datetime(2026, 1, 5, 8, 7, 6), "model1"),
        ]
        
        generate_json(output_file, test_data)
        
        content = output_file.read_text()
        data = json.loads(content)
        
        # Check timestamp format: YYYY-MM-DD-HH-MM-SS
        assert data[0]["timestamp"] == "2026-01-05-08-07-06"
        assert data[0]["model-path"] == "model1"
