#!/usr/bin/env python3
"""
Apply ``recommended-models.json`` to the OpenHands docs.

Given a checked-out clone of ``OpenHands/docs``, this script updates
``openhands/usage/llms/llms.mdx`` so the "Best Cloud Models by Family" and
"Strong Open / Open-Weight Models" tables reflect the latest data emitted by
``scripts/generate_recommended_models.py``.

The script targets the markdown table that follows each section header and
replaces only those table lines. Surrounding narrative (intro paragraphs,
``<Note>`` callouts, etc.) is left untouched, which keeps the docs PR
diff focused on the data that actually changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

DOCS_FILE_REL_PATH = "openhands/usage/llms/llms.mdx"

CLOUD_HEADER = "### Best Cloud Models by Family"
OPEN_HEADER = "### Strong Open / Open-Weight Models"

INDEX_RESULTS_BASE = (
    "https://github.com/OpenHands/openhands-index-results/tree/main/"
)


def _format_average(score: Optional[float]) -> str:
    if score is None:
        return "Not yet listed"
    return f"{score:.1f}"


def _model_link(model: str, model_path: Optional[str]) -> str:
    if model_path:
        return f"[{model}]({INDEX_RESULTS_BASE}{model_path})"
    return model


def _model_string_cell(model_string: Optional[str]) -> str:
    if not model_string:
        return "Not yet listed"
    return f"`{model_string}`"


def render_cloud_table(cloud_models: list[dict]) -> str:
    lines = [
        "| Family | Recommended Model | Model String | OpenHands Index Average |",
        "|--------|-------------------|--------------|-------------------------|",
    ]
    for entry in cloud_models:
        lines.append(
            "| {family} | {model} | {model_string} | {avg} |".format(
                family=entry.get("family", ""),
                model=_model_link(entry["model"], entry.get("model_path")),
                model_string=_model_string_cell(entry.get("model_string")),
                avg=_format_average(entry.get("average_score")),
            )
        )
    return "\n".join(lines)


def render_open_table(open_models: list[dict]) -> str:
    lines = [
        "| Model | Suggested Model String | OpenHands Index Average |",
        "|-------|------------------------|-------------------------|",
    ]
    for entry in open_models:
        lines.append(
            "| {model} | {model_string} | {avg} |".format(
                model=_model_link(entry["model"], entry.get("model_path")),
                model_string=_model_string_cell(entry.get("model_string")),
                avg=_format_average(entry.get("average_score")),
            )
        )
    return "\n".join(lines)


def _replace_table_after_header(content: str, header: str, new_table: str) -> str:
    """Replace the first markdown table that follows ``header``.

    The table is identified as the first contiguous block of lines that start
    with ``|`` after the header. Other content (intro paragraphs, callouts,
    text below the table) is preserved.
    """
    lines = content.splitlines(keepends=True)

    header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        # Match the header with optional trailing whitespace.
        if line.rstrip("\n").rstrip() == header:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find section header: {header!r}")

    # Find the first table row after the header.
    table_start: Optional[int] = None
    for j in range(header_idx + 1, len(lines)):
        stripped = lines[j].lstrip()
        if stripped.startswith("|"):
            table_start = j
            break
        # Stop scanning if we hit a new section header before any table.
        if stripped.startswith("## ") or stripped.startswith("### "):
            break
    if table_start is None:
        raise ValueError(
            f"Could not find a markdown table under section: {header!r}"
        )

    # Find the end of the table (first non-pipe line).
    table_end = table_start
    while table_end < len(lines) and lines[table_end].lstrip().startswith("|"):
        table_end += 1

    new_table_lines = [line + "\n" for line in new_table.splitlines()]
    return "".join(lines[:table_start] + new_table_lines + lines[table_end:])


def apply_recommendations(content: str, recommendations: dict) -> str:
    """Return ``content`` with the two recommendation tables refreshed."""
    cloud_table = render_cloud_table(recommendations.get("cloud_by_family", []))
    open_table = render_open_table(recommendations.get("open_weights", []))

    updated = _replace_table_after_header(content, CLOUD_HEADER, cloud_table)
    updated = _replace_table_after_header(updated, OPEN_HEADER, open_table)
    return updated


def update_docs_file(docs_root: Path, recommendations: dict) -> bool:
    """Update the LLMs docs file in-place. Returns True if it changed."""
    docs_file = docs_root / DOCS_FILE_REL_PATH
    if not docs_file.exists():
        raise FileNotFoundError(f"Docs file not found: {docs_file}")

    original = docs_file.read_text()
    updated = apply_recommendations(original, recommendations)
    if updated == original:
        return False
    docs_file.write_text(updated)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-root",
        type=Path,
        required=True,
        help="Path to a checked-out OpenHands/docs repository.",
    )
    parser.add_argument(
        "--recommendations",
        type=Path,
        required=True,
        help="Path to recommended-models.json.",
    )
    args = parser.parse_args()

    with args.recommendations.open() as fh:
        recommendations = json.load(fh)

    changed = update_docs_file(args.docs_root, recommendations)
    if changed:
        print(f"Updated {args.docs_root / DOCS_FILE_REL_PATH}.")
    else:
        print("No docs changes were necessary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
