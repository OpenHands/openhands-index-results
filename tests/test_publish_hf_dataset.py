"""Tests for the publish_hf_dataset script.

These tests focus on the data-collection surface of the publisher and verify
that alternative agents are intentionally excluded from the dataset that
gets pushed to Hugging Face (see issue #1145).
"""

import json
from pathlib import Path
from unittest.mock import patch

import publish_hf_dataset


def _write_model(model_dir: Path) -> None:
    """Write minimal valid metadata.json/scores.json for one model."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(json.dumps({
        "agent_name": "OpenHands",
        "model": model_dir.name,
    }))
    scores = [{
        "benchmark": "swe-bench",
        "score": 0.5,
        "cost_per_instance": 0.1,
        "average_runtime": 60,
    }]
    (model_dir / "scores.json").write_text(json.dumps(scores))


class TestIterModelDirs:
    """Tests for _iter_model_dirs — the source-of-truth on what gets published."""

    def test_yields_only_results_models(self, tmp_path):
        _write_model(tmp_path / "results" / "GPT-X")
        _write_model(tmp_path / "results" / "Claude-Y")
        # Alternative agents must be ignored even when valid.
        _write_model(tmp_path / "alternative_agents" / "acp-codex" / "GPT-X")
        _write_model(tmp_path / "alternative_agents" / "acp-claude" / "Claude-Y")

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            pairs = list(publish_hf_dataset._iter_model_dirs())

        assert {(d.name, label) for d, label in pairs} == {
            ("Claude-Y", "OpenHands"),
            ("GPT-X", "OpenHands"),
        }

    def test_no_results_dir_yields_nothing(self, tmp_path):
        _write_model(tmp_path / "alternative_agents" / "acp-codex" / "GPT-X")

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            assert list(publish_hf_dataset._iter_model_dirs()) == []


class TestBuildDataframe:
    """End-to-end check that the published DataFrame excludes alt agents."""

    def test_dataframe_excludes_alternative_agents(self, tmp_path):
        _write_model(tmp_path / "results" / "GPT-X")
        _write_model(tmp_path / "alternative_agents" / "acp-codex" / "GPT-X")

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            df = publish_hf_dataset.build_dataframe()

        assert list(df["agent_type"].unique()) == ["OpenHands"]
        assert list(df["id"]) == ["OpenHands/GPT-X"]
