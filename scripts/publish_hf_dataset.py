#!/usr/bin/env python3
"""
Build the OpenHands Index dataset on the HF Hub from ``results/`` and
publish two configs in lockstep:

* ``default``   — one row per model (leaderboard parquet at ``test.parquet``)
* ``instances`` — one row per (model × benchmark × instance), sourced from
  ``results/<model>/instance_results/<benchmark>.json`` sidecars
  (parquet at ``instances.parquet``)

Consumers do:

    from datasets import load_dataset
    ds = load_dataset("OpenHands/openhands-index", split="test")
    instances = load_dataset("OpenHands/openhands-index", "instances", split="test")

Run from CI on every push to main. Requires HF_TOKEN env var (fine-grained
token, write-scoped to OpenHands/openhands-index dataset only).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import tabulate  # noqa: F401  # required by DataFrame.to_markdown(); fail loudly at startup if missing
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

logger = logging.getLogger("publish_hf_dataset")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = os.environ.get("DATASET_REPO", "OpenHands/openhands-index")
PARQUET_PATH = "test.parquet"
INSTANCES_PARQUET_PATH = "instances.parquet"
README_PATH = "README.md"
HASH_PATH = ".source_hash"  # stored in the dataset to detect no-op runs

BENCHMARK_TO_CATEGORY = {
    "swe-bench": "Issue Resolution",
    "swe-bench-multimodal": "Frontend",
    "commit0": "Greenfield",
    "swt-bench": "Testing",
    "gaia": "Information Gathering",
}
BENCHMARKS = list(BENCHMARK_TO_CATEGORY.keys())
INSTANCE_RESULTS_DIRNAME = "instance_results"


def _iter_model_dirs() -> Iterable[tuple[Path, str]]:
    """Yield (model_dir, agent_label) pairs from results/.

    Layout handled:
      results/<model>/{metadata.json,scores.json}          -> agent_label="OpenHands"

    Alternative agents under `alternative_agents/` are intentionally not
    published to the HF dataset yet (see #1145). Anything nested deeper than
    the documented layout is ignored; if the layout changes, extend this
    function rather than the call sites.
    """
    results_dir = REPO_ROOT / "results"
    if results_dir.is_dir():
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                yield d, "OpenHands"


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _flatten(model_dir: Path, agent_label: str) -> dict[str, Any] | None:
    """Build one row from a model directory. Returns None if the dir is invalid."""
    try:
        meta = _load_json(model_dir / "metadata.json")
        scores = _load_json(model_dir / "scores.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Skipping %s: %s", model_dir, e)
        return None

    by_bench: dict[str, Any] = {}
    for entry in scores:
        b = entry.get("benchmark")
        if not b:
            continue
        if b in by_bench:
            logger.warning("Duplicate benchmark %r in %s/scores.json; keeping last.", b, model_dir.name)
        by_bench[b] = entry

    bench_scores: dict[str, dict[str, Any]] = {}
    for b in BENCHMARKS:
        if b in by_bench:
            e = by_bench[b]
            bench_scores[b] = {
                "score": e.get("score"),
                "cost": e.get("cost_per_instance"),
                "runtime": e.get("average_runtime"),
                "logs_url": e.get("full_archive"),
                "visualization_url": e.get("eval_visualization_page"),
            }

    completed = [bench_scores[b] for b in BENCHMARKS if b in bench_scores and bench_scores[b]["score"] is not None]
    if not completed:
        logger.info(
            "Skipping %s (%s): no completed benchmarks with non-null scores.",
            model_dir.name,
            agent_label,
        )
        return None

    def _mean(key: str) -> float | None:
        vals = [c[key] for c in completed if c.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # `id` mirrors the on-disk path, which the repo already treats as unique
    # (e.g. "OpenHands/GPT-5.2"). Stable across runs and never empty,
    # regardless of which optional metadata fields are filled in.
    row: dict[str, Any] = {
        "id": f"{agent_label}/{model_dir.name}",
        "agent_name": meta.get("agent_name", agent_label),
        "agent_type": agent_label,
        "language_model": meta.get("model", model_dir.name),
        "sdk_version": meta.get("agent_version"),
        "openness": meta.get("openness"),
        "country": meta.get("country"),
        "supports_vision": meta.get("supports_vision"),
        "release_date": meta.get("release_date"),
        "average_score": _mean("score"),
        "average_cost": _mean("cost"),
        "average_runtime": _mean("runtime"),
        "categories_completed": len(completed),
    }

    for b in BENCHMARKS:
        cat_slug = BENCHMARK_TO_CATEGORY[b].lower().replace(" ", "_")
        bs = bench_scores.get(b, {})
        row[f"{cat_slug}_score"] = bs.get("score")
        row[f"{cat_slug}_cost"] = bs.get("cost")
        row[f"{cat_slug}_runtime"] = bs.get("runtime")
        row[f"{cat_slug}_logs_url"] = bs.get("logs_url")
        row[f"{cat_slug}_visualization_url"] = bs.get("visualization_url")

    return row


def build_dataframe() -> pd.DataFrame:
    rows = [r for d, label in _iter_model_dirs() if (r := _flatten(d, label)) is not None]
    if not rows:
        raise RuntimeError("No model rows found; refusing to publish empty dataset.")
    df = pd.DataFrame(rows)
    df = df.sort_values("average_score", ascending=False, na_position="last").reset_index(drop=True)
    return df


# Columns for the ``instances`` config. Declared explicitly (rather than
# inferred from the first row) so an all-empty publish still produces a
# parquet with the documented schema, and so downstream consumers can rely
# on column order. Keep this in sync with the dataset card.
INSTANCES_COLUMNS = [
    "id",
    "agent_name",
    "agent_type",
    "language_model",
    "benchmark",
    "category",
    "instance_id",
    "resolved",
    "cost",
]


def _iter_instance_results_files() -> Iterable[tuple[Path, str, str, Path]]:
    """Yield (model_dir, agent_label, benchmark, sidecar_path) for every
    ``results/<model>/instance_results/<benchmark>.json`` sidecar.

    Mirrors the policy in ``_iter_model_dirs``: only ``results/`` is
    published (see issue #1145). Sidecars whose stem isn't a known
    benchmark are skipped — schema validation lives in
    ``scripts/validate_schema.py`` and runs in its own CI job, so we don't
    duplicate it here.
    """
    for model_dir, agent_label in _iter_model_dirs():
        instance_dir = model_dir / INSTANCE_RESULTS_DIRNAME
        if not instance_dir.is_dir():
            continue
        for sidecar in sorted(instance_dir.glob("*.json")):
            benchmark = sidecar.stem
            if benchmark not in BENCHMARK_TO_CATEGORY:
                logger.info(
                    "Skipping unknown benchmark sidecar %s", sidecar.relative_to(REPO_ROOT)
                )
                continue
            yield model_dir, agent_label, benchmark, sidecar


def _instance_rows(
    model_dir: Path, agent_label: str, benchmark: str, sidecar: Path
) -> list[dict[str, Any]]:
    """Flatten one ``instance_results/<benchmark>.json`` sidecar into rows.

    The sidecar shape is ``{instance_id: {"resolved": bool|null, "cost": number|null}}``
    (validated by ``validate_schema.py``). Malformed or non-dict entries are
    skipped with a warning rather than failing the publish — the per-instance
    config is best-effort, and a single bad sidecar shouldn't take down the
    leaderboard push.
    """
    try:
        data = _load_json(sidecar)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Skipping %s: %s", sidecar.relative_to(REPO_ROOT), e)
        return []

    if not isinstance(data, dict):
        logger.warning("Skipping %s: top-level value is not a JSON object", sidecar.relative_to(REPO_ROOT))
        return []

    try:
        meta = _load_json(model_dir / "metadata.json")
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}

    row_id = f"{agent_label}/{model_dir.name}"
    agent_name = meta.get("agent_name", agent_label)
    language_model = meta.get("model", model_dir.name)
    category = BENCHMARK_TO_CATEGORY[benchmark]

    rows: list[dict[str, Any]] = []
    for instance_id, outcome in data.items():
        if not isinstance(instance_id, str) or not isinstance(outcome, dict):
            continue
        resolved = outcome.get("resolved")
        cost = outcome.get("cost")
        # Preserve schema types: resolved is bool|None, cost is float|None.
        if resolved is not None and not isinstance(resolved, bool):
            resolved = None
        if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float))):
            cost = None
        rows.append({
            "id": row_id,
            "agent_name": agent_name,
            "agent_type": agent_label,
            "language_model": language_model,
            "benchmark": benchmark,
            "category": category,
            "instance_id": instance_id,
            "resolved": resolved,
            "cost": float(cost) if cost is not None else None,
        })
    return rows


def build_instances_dataframe() -> pd.DataFrame:
    """Build the long-form per-instance table for the ``instances`` config.

    Returns an empty DataFrame (with the documented schema) when no sidecars
    are present, rather than raising — unlike the leaderboard, an empty
    instances table is a legitimate state (e.g. before #1219 landed).
    """
    rows: list[dict[str, Any]] = []
    for model_dir, label, benchmark, sidecar in _iter_instance_results_files():
        rows.extend(_instance_rows(model_dir, label, benchmark, sidecar))

    if not rows:
        return pd.DataFrame({c: [] for c in INSTANCES_COLUMNS}).astype({
            "id": "string",
            "agent_name": "string",
            "agent_type": "string",
            "language_model": "string",
            "benchmark": "string",
            "category": "string",
            "instance_id": "string",
            "resolved": "boolean",
            "cost": "float64",
        })

    df = pd.DataFrame(rows, columns=INSTANCES_COLUMNS)
    # Nullable dtypes keep ``resolved=None`` / ``cost=None`` round-trippable
    # through parquet without coercing to NaN/False.
    df = df.astype({
        "id": "string",
        "agent_name": "string",
        "agent_type": "string",
        "language_model": "string",
        "benchmark": "string",
        "category": "string",
        "instance_id": "string",
        "resolved": "boolean",
        "cost": "float64",
    })
    df = df.sort_values(["id", "benchmark", "instance_id"], kind="stable").reset_index(drop=True)
    return df


def content_hash(*dfs: pd.DataFrame) -> str:
    """Hash all published DataFrames together so sidecar-only updates (which
    leave the leaderboard table unchanged) still trigger a publish.

    Accepts a variable number of DataFrames in publish order; each is
    serialised with stable column order and joined by a separator to
    prevent collisions where the same bytes could be split differently
    between tables.
    """
    h = hashlib.sha256()
    for i, df in enumerate(dfs):
        if i:
            h.update(b"\n--\n")
        payload = df.reindex(sorted(df.columns), axis=1).to_csv(
            index=False, float_format="%.10g"
        ).encode("utf-8")
        h.update(payload)
    return h.hexdigest()


def get_remote_hash(api: HfApi) -> str | None:
    """Best-effort fetch of the previously-published content hash.

    Returns None on missing file, missing repo, or transient network errors —
    a missing remote hash just forces a publish, which is the safe default.
    """
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO, repo_type="dataset", filename=HASH_PATH,
            token=api.token,
        )
        return Path(path).read_text().strip()
    except (EntryNotFoundError, HfHubHTTPError):
        return None
    except Exception as e:  # network blip, DNS, etc.
        logger.warning("Could not fetch remote hash (%s); will publish anyway.", e)
        return None


def resolve_source_version() -> tuple[str, str, str, str, datetime]:
    """Return (full_sha, short_sha, version_string, info_version, commit_datetime_utc).

    ``version_string`` = ``"YYYY.MM.DD-<short_sha>"`` (e.g. ``2026.05.29-4c92417``)
    is the human-facing version used in tags, commit messages and the rendered
    dataset card. It embeds the source commit hash so each published snapshot
    is unambiguously traceable.

    ``info_version`` = ``"YYYY.M.D"`` (digits only, ``x.y.z`` form) is the
    *machine-facing* version used for the ``dataset_info.version`` YAML field
    in the dataset card. The HF ``datasets`` library parses that field as a
    ``datasets.Version`` and rejects anything other than three dot-separated
    digit groups — including the ``-<short_sha>`` suffix — which used to break
    ``get_dataset_config_names()`` for the leaderboard dataset (see #1189).

    The date in both versions comes from the source commit's committer date so
    the version reflects when the data was finalized, not when CI happened to
    run.
    """
    sha = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    short = sha[:7]
    iso = _git("show", "-s", "--format=%cI", sha)
    dt = datetime.fromisoformat(iso).astimezone(timezone.utc)
    version = f"{dt.year}.{dt.month:02d}.{dt.day:02d}-{short}"
    info_version = f"{dt.year}.{dt.month}.{dt.day}"
    return sha, short, version, info_version, dt


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def dataset_card(
    df: pd.DataFrame,
    generated_at: str,
    source_sha: str,
    version: str,
    info_version: str,
    instances_df: pd.DataFrame | None = None,
) -> str:
    top = df[["language_model", "sdk_version", "agent_name", "average_score",
              "categories_completed", "release_date"]].head(15)
    top_md = top.to_markdown(index=False, floatfmt=".2f") if not top.empty else "_(empty)_"
    instances_rows = 0 if instances_df is None else len(instances_df)
    # NOTE: ``dataset_info.version`` MUST be a digits-only ``x.y.z`` string —
    # the HF datasets library parses it as a ``datasets.Version`` and rejects
    # anything else (e.g. a ``-<short_sha>`` suffix). Using ``info_version``
    # here keeps ``get_dataset_config_names()`` working; the full ``version``
    # with the commit hash is kept further down for human-facing display.
    return f"""---
license: apache-2.0
pretty_name: OpenHands Index Leaderboard
tags:
- leaderboard
- code
- agents
- benchmark
dataset_info:
- config_name: default
  version: {info_version}
  description: >-
    Snapshot of the OpenHands Index leaderboard built from
    openhands-index-results commit {source_sha} on {generated_at}.
- config_name: instances
  version: {info_version}
  description: >-
    Per-instance benchmark outcomes (resolved/cost) for every model in the
    `default` config, sourced from `results/<model>/instance_results/*.json`
    sidecars at commit {source_sha}.
configs:
- config_name: default
  data_files:
  - split: test
    path: test.parquet
- config_name: instances
  data_files:
  - split: test
    path: instances.parquet
---

# OpenHands Index — Leaderboard Snapshot

Auto-published from <https://github.com/OpenHands/openhands-index-results>
on every push to `main`. Matches the table shown at the
[OpenHands Index Space](https://huggingface.co/spaces/OpenHands/openhands-index).

```python
from datasets import load_dataset

# leaderboard (one row per model)
ds = load_dataset("{DATASET_REPO}", split="test")
ds.info.version          # → "{version}"

# per-instance outcomes (one row per model × instance)
instances = load_dataset("{DATASET_REPO}", "instances", split="test")

# pin to this exact snapshot
ds = load_dataset("{DATASET_REPO}", split="test", revision="v{version}")
```

## Categories

| Category | Backing benchmark |
|---|---|
| Issue Resolution | SWE-Bench |
| Frontend | SWE-Bench Multimodal |
| Greenfield | Commit0 |
| Testing | SWT-Bench |
| Information Gathering | GAIA |

`average_score` is the mean of the per-benchmark scores actually completed.
`categories_completed` tells you how many benchmarks the model has run.

## Configs

### `default` — leaderboard (one row per model)

The aggregate table backing the leaderboard Space. Use this for ranking,
averages, and per-category scores.

### `instances` — per-instance outcomes (one row per model × benchmark instance)

Long-form table of every benchmark instance's outcome for every model in
`default`. Join to `default` on `id`.

| Column | Type | Notes |
|---|---|---|
| `id` | string | Matches `default.id` (e.g. `OpenHands/GPT-5.5`) |
| `agent_name` | string | Display name from `metadata.json` |
| `agent_type` | string | Currently always `OpenHands` (see [#1145](https://github.com/OpenHands/openhands-index-results/issues/1145)) |
| `language_model` | string | LLM identifier |
| `benchmark` | string | One of `swe-bench`, `swe-bench-multimodal`, `commit0`, `swt-bench`, `gaia` |
| `category` | string | Human-facing category label |
| `instance_id` | string | Benchmark-specific instance ID |
| `resolved` | bool? | `true` / `false` / `null` when the archive didn't record an outcome |
| `cost` | float? | USD; `null` when unavailable |

## This snapshot

- Version: **`{version}`**
- Rows: **{len(df)}** (`default`) · **{instances_rows}** (`instances`)
- Generated: `{generated_at}`
- Source commit: [`{source_sha}`](https://github.com/OpenHands/openhands-index-results/commit/{source_sha})

## Top 15 by average score

{top_md}
"""


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN env var not set.")
        return 2

    api = HfApi(token=token)

    try:
        df = build_dataframe()
    except RuntimeError as e:
        logger.error("%s", e)
        return 1

    instances_df = build_instances_dataframe()

    new_hash = content_hash(df, instances_df)
    logger.info(
        "Built %d leaderboard rows + %d instance rows; content hash %s",
        len(df), len(instances_df), new_hash[:12],
    )

    remote_hash = get_remote_hash(api)
    force = os.environ.get("FORCE", "").lower() in {"1", "true", "yes"}
    if not force and remote_hash == new_hash:
        logger.info("Dataset already up-to-date (remote hash matches). Skipping push.")
        return 0

    try:
        api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)
    except HfHubHTTPError as e:
        logger.error("create_repo failed: %s", e)
        return 1

    try:
        source_sha, _, version, info_version, _ = resolve_source_version()
    except subprocess.CalledProcessError as e:
        logger.error("Failed to resolve source version: %s", e)
        return 1
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    instances_buf = io.BytesIO()
    instances_df.to_parquet(instances_buf, index=False)
    instances_parquet_bytes = instances_buf.getvalue()

    readme_bytes = dataset_card(
        df, generated_at, source_sha, version, info_version,
        instances_df=instances_df,
    ).encode("utf-8")

    ops = [
        CommitOperationAdd(path_in_repo=PARQUET_PATH, path_or_fileobj=parquet_bytes),
        CommitOperationAdd(path_in_repo=INSTANCES_PARQUET_PATH, path_or_fileobj=instances_parquet_bytes),
        CommitOperationAdd(path_in_repo=README_PATH, path_or_fileobj=readme_bytes),
        CommitOperationAdd(path_in_repo=HASH_PATH, path_or_fileobj=new_hash.encode()),
    ]
    try:
        commit_info = api.create_commit(
            repo_id=DATASET_REPO, repo_type="dataset",
            operations=ops,
            commit_message=(
                f"Update leaderboard v{version} "
                f"({len(df)} models, {len(instances_df)} instance rows)"
            ),
        )
    except HfHubHTTPError as e:
        logger.error("create_commit failed: %s", e)
        return 1
    logger.info("Published %d rows to %s as v%s", len(df), DATASET_REPO, version)

    # Immutable tag so users can pin: load_dataset(..., revision="v2026.05.29-4c92417")
    tag = f"v{version}"
    try:
        api.create_tag(
            repo_id=DATASET_REPO, repo_type="dataset",
            tag=tag, tag_message=f"Leaderboard snapshot from {source_sha}",
            revision=commit_info.oid,
            exist_ok=True,
        )
        logger.info("Tagged %s", tag)
    except HfHubHTTPError as e:
        logger.warning("Could not create tag %s: %s", tag, e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
