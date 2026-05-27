"""Tests for ``scripts/update_docs_recommendations.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from update_docs_recommendations import (  # noqa: E402
    CLOUD_HEADER,
    DOCS_FILE_REL_PATH,
    OPEN_HEADER,
    apply_recommendations,
    render_cloud_table,
    render_open_table,
    update_docs_file,
)


SAMPLE_DOCS = f"""# Top

Intro paragraph that must be preserved.

{CLOUD_HEADER}

| Family | Recommended Model | Model String | OpenHands Index Average | Notes |
|--------|-------------------|--------------|-------------------------|-------|
| Claude | [Old Claude](https://example.com) | `anthropic/old` | 50.0 | old notes |
| GPT | [Old GPT](https://example.com) | `openai/old` | 40.0 | old notes |

A paragraph that explains the table and must stay put.

{OPEN_HEADER}

These open or open-weight models have good OpenHands Index scores:

| Model | Suggested Model String | OpenHands Index Average | Notes |
|-------|------------------------|-------------------------|-------|
| [Old-OW](https://example.com) | `openrouter/old` | 30.0 | obsolete |

<Note>
A note that must be preserved.
</Note>

Another paragraph.

### Local / Self-Hosted Models

Final unrelated content.
"""


SAMPLE_RECOMMENDATIONS = {
    "generated_at": "2026-05-27T00:00:00+00:00",
    "cloud_by_family": [
        {
            "model": "claude-opus-4-7",
            "model_path": "results/claude-opus-4-7",
            "average_score": 68.2,
            "benchmarks_count": 5,
            "family": "Claude",
            "openness": "closed_api_available",
            "model_string": "anthropic/claude-opus-4-7",
        },
        {
            "model": "GPT-5.5",
            "model_path": "results/GPT-5.5",
            "average_score": 65.9,
            "benchmarks_count": 5,
            "family": "GPT",
            "openness": "closed_api_available",
            "model_string": "openai/gpt-5.5",
        },
    ],
    "open_weights": [
        {
            "model": "GLM-5.1",
            "model_path": "results/GLM-5.1",
            "average_score": 58.2,
            "benchmarks_count": 5,
            "family": None,
            "openness": "open_weights",
            "model_string": "openrouter/z-ai/glm-5.1",
        },
    ],
}


def test_render_cloud_table_handles_missing_model_string() -> None:
    table = render_cloud_table(
        [
            {
                "model": "Foo-1",
                "model_path": "results/Foo-1",
                "average_score": 42.5,
                "family": "Claude",
                "model_string": None,
            }
        ]
    )
    assert "| Foo-1 |" in table.replace(
        "[Foo-1](https://github.com/OpenHands/openhands-index-results/tree/main/results/Foo-1)",
        "Foo-1",
    )
    assert "Not yet listed" in table
    assert "42.5" in table


def test_render_open_table_omits_notes_column() -> None:
    table = render_open_table(
        [
            {
                "model": "Foo-OW",
                "model_path": "results/Foo-OW",
                "average_score": 30.0,
                "model_string": "openrouter/foo",
            }
        ]
    )
    assert "Notes" not in table
    assert "| Suggested Model String |" in table
    assert "`openrouter/foo`" in table


def test_apply_recommendations_replaces_only_tables() -> None:
    updated = apply_recommendations(SAMPLE_DOCS, SAMPLE_RECOMMENDATIONS)

    # Surrounding narrative is preserved verbatim.
    assert "Intro paragraph that must be preserved." in updated
    assert "A paragraph that explains the table and must stay put." in updated
    assert "These open or open-weight models have good OpenHands Index scores:" in updated
    assert "A note that must be preserved." in updated
    assert "Another paragraph." in updated
    assert "### Local / Self-Hosted Models" in updated
    assert "Final unrelated content." in updated

    # Old rows are gone, new rows are present.
    assert "Old Claude" not in updated
    assert "Old-OW" not in updated
    assert "claude-opus-4-7" in updated
    assert "GPT-5.5" in updated
    assert "GLM-5.1" in updated

    # Each table is rendered once.
    assert updated.count("| Family | Recommended Model |") == 1
    assert updated.count("| Model | Suggested Model String |") == 1


def test_apply_recommendations_is_idempotent() -> None:
    first = apply_recommendations(SAMPLE_DOCS, SAMPLE_RECOMMENDATIONS)
    second = apply_recommendations(first, SAMPLE_RECOMMENDATIONS)
    assert first == second


def test_apply_recommendations_missing_header() -> None:
    docs = "## Some other content\n\nno relevant sections here\n"
    with pytest.raises(ValueError, match="Best Cloud Models by Family"):
        apply_recommendations(docs, SAMPLE_RECOMMENDATIONS)


def test_update_docs_file_writes_and_reports_change(tmp_path: Path) -> None:
    docs_file = tmp_path / DOCS_FILE_REL_PATH
    docs_file.parent.mkdir(parents=True)
    docs_file.write_text(SAMPLE_DOCS)

    changed = update_docs_file(tmp_path, SAMPLE_RECOMMENDATIONS)
    assert changed is True

    # Second run should be a no-op.
    changed_again = update_docs_file(tmp_path, SAMPLE_RECOMMENDATIONS)
    assert changed_again is False


def test_update_docs_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        update_docs_file(tmp_path, SAMPLE_RECOMMENDATIONS)


def test_recommendations_json_format(tmp_path: Path) -> None:
    """Sanity check: ``apply_recommendations`` accepts the JSON shape we write."""
    payload_path = tmp_path / "recommended-models.json"
    payload_path.write_text(json.dumps(SAMPLE_RECOMMENDATIONS))

    loaded = json.loads(payload_path.read_text())
    apply_recommendations(SAMPLE_DOCS, loaded)
