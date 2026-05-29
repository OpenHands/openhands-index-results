#!/usr/bin/env python3
"""
Build a flat leaderboard table from results/ and alternative_agents/, then
publish it to the HF Dataset repo `OpenHands/openhands-index` so consumers
can do:

    from datasets import load_dataset
    ds = load_dataset("OpenHands/openhands-index", split="test")

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


def _iter_model_dirs() -> Iterable[tuple[Path, str]]:
    """Yield (model_dir, agent_label) pairs from results/ and alternative_agents/.

    Layouts handled:
      results/<model>/{metadata.json,scores.json}          -> agent_label="OpenHands"
      alternative_agents/<agent_type>/<model>/{...}        -> agent_label="<agent_type>"

    Anything nested deeper than that is ignored; if the layout changes,
    extend this function rather than the call sites.
    """
    results_dir = REPO_ROOT / "results"
    if results_dir.is_dir():
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                yield d, "OpenHands"

    alt_root = REPO_ROOT / "alternative_agents"
    if alt_root.is_dir():
        for agent_type_dir in sorted(alt_root.iterdir()):
            if not agent_type_dir.is_dir():
                continue
            for d in sorted(agent_type_dir.iterdir()):
                if d.is_dir() and (d / "metadata.json").exists():
                    yield d, agent_type_dir.name


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
        return None

    def _mean(key: str) -> float | None:
        vals = [c[key] for c in completed if c.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # `id` mirrors the on-disk path, which the repo already treats as unique
    # (e.g. "OpenHands/GPT-5.2", "acp-codex/GPT-5.5"). Stable across runs and
    # never empty, regardless of which optional metadata fields are filled in.
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


def content_hash(df: pd.DataFrame) -> str:
    payload = df.reindex(sorted(df.columns), axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def resolve_source_version() -> tuple[str, str, str, datetime]:
    """Return (full_sha, short_sha, version_string, commit_datetime_utc).

    version_string = "YYYY.MM.DD-<short_sha>" (e.g. 2026.05.29-4c92417).
    Date comes from the source commit's committer date so the version
    reflects when the data was finalized, not when CI happened to run.
    """
    sha = os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD")
    short = sha[:7]
    iso = _git("show", "-s", "--format=%cI", sha)
    dt = datetime.fromisoformat(iso).astimezone(timezone.utc)
    version = f"{dt.year}.{dt.month}.{dt.day}-{short}"
    return sha, short, version, dt


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def dataset_card(
    df: pd.DataFrame, generated_at: str, source_sha: str, version: str
) -> str:
    top = df[["language_model", "sdk_version", "agent_name", "average_score",
              "categories_completed", "release_date"]].head(15)
    top_md = top.to_markdown(index=False, floatfmt=".2f") if not top.empty else "_(empty)_"
    return f"""---
license: apache-2.0
pretty_name: OpenHands Index Leaderboard
tags:
- leaderboard
- code
- agents
- benchmark
dataset_info:
  config_name: default
  version: {version}
  description: >-
    Snapshot of the OpenHands Index leaderboard built from
    openhands-index-results commit {source_sha} on {generated_at}.
configs:
- config_name: default
  data_files:
  - split: test
    path: test.parquet
---

# OpenHands Index — Leaderboard Snapshot

Auto-published from <https://github.com/OpenHands/openhands-index-results>
on every push to `main`. Matches the table shown at the
[OpenHands Index Space](https://huggingface.co/spaces/OpenHands/openhands-index).

```python
from datasets import load_dataset

# latest
ds = load_dataset("{DATASET_REPO}", split="test")
ds.info.version          # → "{version}"

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

## This snapshot

- Version: **`{version}`**
- Rows: **{len(df)}**
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

    new_hash = content_hash(df)
    logger.info("Built %d rows; content hash %s", len(df), new_hash[:12])

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
        source_sha, _, version, _ = resolve_source_version()
    except subprocess.CalledProcessError as e:
        logger.error("Failed to resolve source version: %s", e)
        return 1
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()
    readme_bytes = dataset_card(df, generated_at, source_sha, version).encode("utf-8")

    ops = [
        CommitOperationAdd(path_in_repo=PARQUET_PATH, path_or_fileobj=parquet_bytes),
        CommitOperationAdd(path_in_repo=README_PATH, path_or_fileobj=readme_bytes),
        CommitOperationAdd(path_in_repo=HASH_PATH, path_or_fileobj=new_hash.encode()),
    ]
    try:
        commit_info = api.create_commit(
            repo_id=DATASET_REPO, repo_type="dataset",
            operations=ops,
            commit_message=f"Update leaderboard v{version} ({len(df)} models)",
        )
    except HfHubHTTPError as e:
        logger.error("create_commit failed: %s", e)
        return 1
    logger.info("Published %d rows to %s as v%s", len(df), DATASET_REPO, version)

    # Immutable tag so users can pin: load_dataset(..., revision="v2026.5.29-4c92417")
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
