#!/usr/bin/env python3
"""
Generate ``recommended-models.json`` with the current best models per family.

For every model that has at least one benchmark score we compute the average
of its primary ``score`` values. We then pick:

* the best ``Claude``, ``GPT`` and ``Gemini`` cloud model (each must have
  ``openness == closed_api_available`` per ``MODEL_OPENNESS_MAP``);
* the top ``open_weights`` models, ordered by average score.

The resulting JSON file feeds the docs updater (``scripts/update_docs_recommendations.py``),
which keeps the recommended-models tables in ``OpenHands/docs`` in sync with this
repository.

Only models that have completed all expected benchmarks are eligible. This keeps
the comparison fair across the whole evaluation suite and avoids ranking
partial submissions.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# Make scripts/ importable so we can reuse the model definitions.
sys.path.insert(0, str(Path(__file__).parent))

from validate_schema import (  # noqa: E402  (path tweak above)
    MODEL_OPENNESS_MAP,
    Model,
    Openness,
)

# Benchmarks that count toward a "complete" submission for ranking purposes.
# Mirrors EXPECTED_BENCHMARKS in measure_progress.py.
EXPECTED_BENCHMARKS = (
    "swe-bench",
    "swe-bench-multimodal",
    "swt-bench",
    "commit0",
    "gaia",
)

# Top N open-weights models to surface in the docs.
DEFAULT_OPEN_WEIGHTS_LIMIT = 5

# Suggested LiteLLM-style model strings. The docs page uses these as the
# canonical identifier shown next to each recommendation. We keep them here
# rather than computing them so the docs stay stable when a provider
# renames an endpoint or a new alias appears.
DEFAULT_MODEL_STRINGS: dict[str, str] = {
    "claude-opus-4-7": "anthropic/claude-opus-4-7",
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
    "claude-opus-4-5": "anthropic/claude-opus-4-5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "GPT-5.5": "openai/gpt-5.5",
    "GPT-5.4": "openai/gpt-5.4",
    "GPT-5.2": "openai/gpt-5.2",
    "GPT-5.2-Codex": "openai/gpt-5.2-codex",
    "Gemini-3-Pro": "gemini/gemini-3-pro-preview",
    "Gemini-3.1-Pro": "gemini/gemini-3.1-pro-preview",
    "Gemini-3-Flash": "gemini/gemini-3-flash-preview",
    "GLM-5.1": "openrouter/z-ai/glm-5.1",
    "GLM-5": "openrouter/z-ai/glm-5",
    "GLM-4.7": "openrouter/z-ai/glm-4.7",
    "Kimi-K2.6": "openrouter/moonshotai/kimi-k2.6",
    "Kimi-K2.5": "openrouter/moonshotai/kimi-k2.5",
    "Kimi-K2-Thinking": "openrouter/moonshotai/kimi-k2-thinking",
    "DeepSeek-V4-Pro": "openrouter/deepseek/deepseek-v4-pro",
    "DeepSeek-V3.2-Reasoner": "openrouter/deepseek/deepseek-v3.2-reasoner",
    "MiniMax-M2.7": "openrouter/minimax/minimax-m2.7",
    "MiniMax-M2.5": "openrouter/minimax/minimax-m2.5",
    "MiniMax-M2.1": "openrouter/minimax/minimax-m2.1",
    "MiniMax-M3": "openrouter/minimax/minimax-m3",
    "Minimax-2.7": "openrouter/minimax/minimax-2.7",
    "Qwen3-Coder-Next": "openrouter/qwen/qwen3-coder-next",
    "Qwen3-Coder-480B": "openrouter/qwen/qwen3-coder-480b",
    "Qwen3.5-Flash": "openrouter/qwen/qwen3.5-flash",
    "Qwen3.6-Plus": "openrouter/qwen/qwen3.6-plus",
    "Nemotron-3-Super": "openrouter/nvidia/nemotron-3-super",
    "Nemotron-3-Nano": "openrouter/nvidia/nemotron-3-nano",
    "Trinity-Large-Thinking": "openrouter/trinity/trinity-large-thinking",
}

# Family detection rules for cloud models. Order matters: the first matching
# prefix wins. Keep these aligned with the families surfaced in the docs.
CLOUD_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Claude", ("claude-",)),
    ("GPT", ("GPT-", "gpt-")),
    ("Gemini", ("Gemini-", "gemini-")),
)


@dataclass
class ModelSummary:
    """Aggregate view of a single model used by the docs updater."""

    model: str
    model_path: str
    average_score: float
    benchmarks_count: int
    family: Optional[str]
    openness: Optional[str]
    model_string: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_family(model_name: str) -> Optional[str]:
    """Return the cloud family for ``model_name`` or ``None``."""
    for family, prefixes in CLOUD_FAMILY_RULES:
        if any(model_name.startswith(prefix) for prefix in prefixes):
            return family
    return None


def _safe_load_json(path: Path) -> Optional[object]:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not load {path}: {exc}", file=sys.stderr)
        return None


def collect_model_summaries(
    results_dir: Path,
    expected_benchmarks: Iterable[str] = EXPECTED_BENCHMARKS,
) -> list[ModelSummary]:
    """Walk ``results_dir`` and build a :class:`ModelSummary` per complete model.

    "Complete" means the model has score entries for every benchmark in
    ``expected_benchmarks``. Incomplete submissions are skipped so cross-family
    rankings stay comparable.
    """
    expected = set(expected_benchmarks)
    summaries: list[ModelSummary] = []

    if not results_dir.is_dir():
        return summaries

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        scores_path = model_dir / "scores.json"
        metadata_path = model_dir / "metadata.json"
        if not scores_path.exists() or not metadata_path.exists():
            continue

        scores = _safe_load_json(scores_path)
        metadata = _safe_load_json(metadata_path)
        if not isinstance(scores, list) or not isinstance(metadata, dict):
            continue

        if metadata.get("available") is False:
            continue

        benchmarks_seen = {
            entry.get("benchmark")
            for entry in scores
            if isinstance(entry, dict) and entry.get("benchmark")
        }
        if not expected.issubset(benchmarks_seen):
            continue

        numeric_scores = [
            float(entry["score"])
            for entry in scores
            if isinstance(entry, dict)
            and isinstance(entry.get("score"), (int, float))
        ]
        if not numeric_scores:
            continue

        model_name = metadata.get("model") or model_dir.name
        openness = metadata.get("openness")
        try:
            openness_enum = MODEL_OPENNESS_MAP.get(Model(model_name))
        except ValueError:
            openness_enum = None
        if openness_enum is not None:
            openness = openness_enum.value

        summaries.append(
            ModelSummary(
                model=model_name,
                model_path=str(model_dir.relative_to(results_dir.parent)),
                average_score=round(sum(numeric_scores) / len(numeric_scores), 1),
                benchmarks_count=len(numeric_scores),
                family=_detect_family(model_name),
                openness=openness,
                model_string=DEFAULT_MODEL_STRINGS.get(model_name),
            )
        )

    return summaries


def pick_recommendations(
    summaries: list[ModelSummary],
    open_weights_limit: int = DEFAULT_OPEN_WEIGHTS_LIMIT,
) -> dict:
    """Pick the best model per cloud family and the top open-weights models."""
    closed = [
        s
        for s in summaries
        if s.openness == Openness.CLOSED_API_AVAILABLE.value and s.family
    ]
    open_weights = [
        s for s in summaries if s.openness == Openness.OPEN_WEIGHTS.value
    ]

    cloud_by_family: list[dict] = []
    for family, _prefixes in CLOUD_FAMILY_RULES:
        family_models = [s for s in closed if s.family == family]
        if not family_models:
            continue
        family_models.sort(key=lambda s: s.average_score, reverse=True)
        cloud_by_family.append(family_models[0].to_dict())

    open_weights.sort(key=lambda s: s.average_score, reverse=True)
    open_weights_data = [s.to_dict() for s in open_weights[:open_weights_limit]]

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "cloud_by_family": cloud_by_family,
        "open_weights": open_weights_data,
    }


def write_recommendations(output_path: Path, recommendations: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(recommendations, fh, indent=2)
        fh.write("\n")


def main() -> None:
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    output_path = repo_root / "recommended-models.json"

    print(f"Scanning {results_dir} for complete models...")
    summaries = collect_model_summaries(results_dir)
    print(f"Found {len(summaries)} complete models.")

    recommendations = pick_recommendations(summaries)
    print(
        "Selected "
        f"{len(recommendations['cloud_by_family'])} cloud families and "
        f"{len(recommendations['open_weights'])} open-weights models."
    )

    write_recommendations(output_path, recommendations)
    print(f"Wrote {output_path}.")


if __name__ == "__main__":
    main()
