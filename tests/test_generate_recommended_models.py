"""Tests for ``scripts/generate_recommended_models.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_recommended_models import (  # noqa: E402
    EXPECTED_BENCHMARKS,
    collect_model_summaries,
    pick_recommendations,
    write_recommendations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_scores(score_by_benchmark: dict[str, float]) -> list[dict]:
    """Build a scores.json list that covers EXPECTED_BENCHMARKS."""
    scores = []
    for bench in EXPECTED_BENCHMARKS:
        scores.append(
            {
                "benchmark": bench,
                "score": score_by_benchmark.get(bench, 50.0),
                "metric": "accuracy",
                "cost_per_instance": 1.0,
                "average_runtime": 100,
            }
        )
    return scores


def _write_model(
    parent: Path,
    name: str,
    model: str,
    scores: list[dict],
    openness: str = "closed_api_available",
    available: bool = True,
) -> None:
    model_dir = parent / name
    model_dir.mkdir(parents=True)
    meta = {"model": model, "openness": openness, "directory_name": name}
    if available is False:
        meta["available"] = False
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    (model_dir / "scores.json").write_text(json.dumps(scores))


# ---------------------------------------------------------------------------
# collect_model_summaries
# ---------------------------------------------------------------------------


def test_collect_returns_complete_models_only(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()

    _write_model(results, "claude-opus-4-7", "claude-opus-4-7", _full_scores({}))

    # Partial submission: only three benchmarks.
    partial_scores = [
        {"benchmark": "swe-bench", "score": 10.0, "metric": "accuracy"},
        {"benchmark": "gaia", "score": 20.0, "metric": "accuracy"},
        {"benchmark": "swt-bench", "score": 30.0, "metric": "accuracy"},
    ]
    _write_model(results, "GPT-5.4", "GPT-5.4", partial_scores)

    summaries = collect_model_summaries(results)
    assert {s.model for s in summaries} == {"claude-opus-4-7"}


def test_collect_skips_unavailable_models(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()

    _write_model(results, "claude-opus-4-7", "claude-opus-4-7", _full_scores({}))
    _write_model(
        results,
        "Gemini-3-Pro",
        "Gemini-3-Pro",
        _full_scores({}),
        available=False,
    )

    summaries = collect_model_summaries(results)
    assert {s.model for s in summaries} == {"claude-opus-4-7"}


def test_collect_computes_average_and_family(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    scores = _full_scores(
        {
            "swe-bench": 70.0,
            "swe-bench-multimodal": 60.0,
            "swt-bench": 80.0,
            "commit0": 50.0,
            "gaia": 90.0,
        }
    )
    _write_model(results, "Gemini-3.1-Pro", "Gemini-3.1-Pro", scores)

    [summary] = collect_model_summaries(results)

    assert summary.model == "Gemini-3.1-Pro"
    assert summary.average_score == 70.0  # (70+60+80+50+90)/5
    assert summary.family == "Gemini"
    # openness should be taken from MODEL_OPENNESS_MAP, not metadata.
    assert summary.openness == "closed_api_available"
    assert summary.model_path == "results/Gemini-3.1-Pro"


def test_collect_skips_unknown_model_names(tmp_path: Path) -> None:
    """Models missing from ``MODEL_OPENNESS_MAP`` keep the metadata openness."""
    results = tmp_path / "results"
    results.mkdir()
    _write_model(
        results,
        "Mystery-Model",
        "Mystery-Model",
        _full_scores({}),
        openness="closed_api_available",
    )

    [summary] = collect_model_summaries(results)
    assert summary.openness == "closed_api_available"
    assert summary.family is None  # No family prefix match.


def test_collect_handles_missing_directory(tmp_path: Path) -> None:
    summaries = collect_model_summaries(tmp_path / "missing")
    assert summaries == []


def test_collect_handles_invalid_json(tmp_path: Path) -> None:
    results = tmp_path / "results"
    bad_dir = results / "broken"
    bad_dir.mkdir(parents=True)
    (bad_dir / "metadata.json").write_text("{")
    (bad_dir / "scores.json").write_text("[]")

    summaries = collect_model_summaries(results)
    assert summaries == []


# ---------------------------------------------------------------------------
# pick_recommendations
# ---------------------------------------------------------------------------


def test_pick_recommendations_selects_best_per_family(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    _write_model(
        results,
        "claude-opus-4-7",
        "claude-opus-4-7",
        _full_scores({b: 80.0 for b in EXPECTED_BENCHMARKS}),
    )
    _write_model(
        results,
        "claude-opus-4-5",
        "claude-opus-4-5",
        _full_scores({b: 70.0 for b in EXPECTED_BENCHMARKS}),
    )
    _write_model(
        results,
        "GPT-5.5",
        "GPT-5.5",
        _full_scores({b: 65.0 for b in EXPECTED_BENCHMARKS}),
    )
    _write_model(
        results,
        "Gemini-3.1-Pro",
        "Gemini-3.1-Pro",
        _full_scores({b: 57.0 for b in EXPECTED_BENCHMARKS}),
    )

    summaries = collect_model_summaries(results)
    rec = pick_recommendations(summaries)

    families = [entry["family"] for entry in rec["cloud_by_family"]]
    assert families == ["Claude", "GPT", "Gemini"]

    claude_entry = rec["cloud_by_family"][0]
    assert claude_entry["model"] == "claude-opus-4-7"  # Best Claude wins.
    assert claude_entry["average_score"] == 80.0
    assert claude_entry["model_string"] == "anthropic/claude-opus-4-7"


def test_pick_recommendations_sorts_open_weights(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    _write_model(
        results,
        "GLM-5.1",
        "GLM-5.1",
        _full_scores({b: 58.0 for b in EXPECTED_BENCHMARKS}),
        openness="open_weights",
    )
    _write_model(
        results,
        "Kimi-K2.6",
        "Kimi-K2.6",
        _full_scores({b: 57.0 for b in EXPECTED_BENCHMARKS}),
        openness="open_weights",
    )
    _write_model(
        results,
        "MiniMax-M2.7",
        "MiniMax-M2.7",
        _full_scores({b: 43.0 for b in EXPECTED_BENCHMARKS}),
        openness="open_weights",
    )

    summaries = collect_model_summaries(results)
    rec = pick_recommendations(summaries, open_weights_limit=2)

    assert [e["model"] for e in rec["open_weights"]] == ["GLM-5.1", "Kimi-K2.6"]


def test_pick_recommendations_returns_empty_when_no_models() -> None:
    rec = pick_recommendations([])
    assert rec["cloud_by_family"] == []
    assert rec["open_weights"] == []
    # generated_at is intentionally not set by pick_recommendations; it is
    # filled in by write_recommendations only when content changes.
    assert "generated_at" not in rec


# ---------------------------------------------------------------------------
# write_recommendations
# ---------------------------------------------------------------------------


def test_write_recommendations_round_trips(tmp_path: Path) -> None:
    """A first write (no existing file) sets generated_at to now."""
    cloud_by_family = [
        {
            "model": "claude-opus-4-7",
            "model_path": "results/claude-opus-4-7",
            "average_score": 68.2,
            "benchmarks_count": 5,
            "family": "Claude",
            "openness": "closed_api_available",
            "model_string": "anthropic/claude-opus-4-7",
        }
    ]
    recommendations = {"cloud_by_family": cloud_by_family, "open_weights": []}

    out = tmp_path / "recommended-models.json"
    write_recommendations(out, recommendations)

    text = out.read_text()
    assert text.endswith("\n")
    written = json.loads(text)
    assert written["cloud_by_family"] == cloud_by_family
    assert written["open_weights"] == []
    assert "generated_at" in written  # set to now on first write


def test_write_recommendations_preserves_timestamp_when_unchanged(
    tmp_path: Path,
) -> None:
    """Re-running with identical content must not bump generated_at.

    This is the guard against churn PRs that touch nothing but a timestamp.
    """
    cloud = [
        {
            "model": "claude-opus-4-7",
            "model_path": "results/claude-opus-4-7",
            "average_score": 68.2,
            "benchmarks_count": 5,
            "family": "Claude",
            "openness": "closed_api_available",
            "model_string": "anthropic/claude-opus-4-7",
        }
    ]
    out = tmp_path / "recommended-models.json"

    # First write: no existing file, so generated_at is set to now.
    write_recommendations(out, {"cloud_by_family": cloud, "open_weights": []})
    first_text = out.read_text()
    original_ts = json.loads(first_text)["generated_at"]

    # Second write: identical content, should keep the old timestamp and be
    # byte-identical to the first write.
    write_recommendations(out, {"cloud_by_family": cloud, "open_weights": []})

    assert json.loads(out.read_text())["generated_at"] == original_ts
    assert out.read_text() == first_text


def test_write_recommendations_bumps_timestamp_on_content_change(
    tmp_path: Path, monkeypatch
) -> None:
    """When the recommendation content changes, generated_at advances."""
    from datetime import datetime as real_datetime, timezone as real_tz

    import generate_recommended_models as grm

    # Fake clock: first call returns t0, second call returns t1.
    times = iter(
        [
            real_datetime(2026, 6, 1, 12, 0, 0, tzinfo=real_tz.utc),
            real_datetime(2026, 6, 2, 12, 0, 0, tzinfo=real_tz.utc),
        ]
    )

    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            return next(times)

        timezone = real_tz

    monkeypatch.setattr(grm, "datetime", _FakeDateTime)

    cloud = [
        {
            "model": "claude-opus-4-7",
            "model_path": "results/claude-opus-4-7",
            "average_score": 68.2,
            "benchmarks_count": 5,
            "family": "Claude",
            "openness": "closed_api_available",
            "model_string": "anthropic/claude-opus-4-7",
        }
    ]
    out = tmp_path / "recommended-models.json"

    write_recommendations(out, {"cloud_by_family": cloud, "open_weights": []})
    original_ts = json.loads(out.read_text())["generated_at"]

    # Content changes: a different model now tops the Claude family.
    new_cloud = [
        {
            "model": "claude-opus-4-8",
            "model_path": "results/claude-opus-4-8",
            "average_score": 72.0,
            "benchmarks_count": 5,
            "family": "Claude",
            "openness": "closed_api_available",
            "model_string": "anthropic/claude-opus-4-8",
        }
    ]
    write_recommendations(out, {"cloud_by_family": new_cloud, "open_weights": []})
    written = json.loads(out.read_text())

    assert written["generated_at"] != original_ts
    assert written["cloud_by_family"][0]["model"] == "claude-opus-4-8"
