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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
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
CATEGORIES = list(BENCHMARK_TO_CATEGORY.values())
BENCHMARKS = list(BENCHMARK_TO_CATEGORY.keys())


def _iter_model_dirs() -> Iterable[tuple[Path, str]]:
    """Yield (model_dir, agent_label) pairs from results/ and alternative_agents/."""
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

    by_bench = {entry["benchmark"]: entry for entry in scores if "benchmark" in entry}

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

    row: dict[str, Any] = {
        "id": f"{agent_label}_{meta.get('agent_version', '')}_{meta.get('model', model_dir.name)}",
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
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO, repo_type="dataset", filename=HASH_PATH,
            token=api.token,
        )
        return Path(path).read_text().strip()
    except (EntryNotFoundError, HfHubHTTPError):
        return None


def dataset_card(df: pd.DataFrame, generated_at: str, source_sha: str | None) -> str:
    top = df[["language_model", "sdk_version", "agent_name", "average_score",
              "categories_completed", "release_date"]].head(15)
    top_md = top.to_markdown(index=False, floatfmt=".2f") if not top.empty else "_(empty)_"
    src = f"`{source_sha[:8]}`" if source_sha else "_unknown_"
    return f"""---
license: apache-2.0
pretty_name: OpenHands Index Leaderboard
tags:
- leaderboard
- code
- agents
- benchmark
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
ds = load_dataset("{DATASET_REPO}", split="test")
df = ds.to_pandas()
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

## Snapshot

- Rows: **{len(df)}**
- Generated: `{generated_at}`
- Source commit: {src}

## Top 15 by average score

{top_md}
"""


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN env var not set.")
        return 2

    api = HfApi(token=token)
    df = build_dataframe()
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

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    source_sha = os.environ.get("GITHUB_SHA")
    readme_bytes = dataset_card(df, generated_at, source_sha).encode("utf-8")

    from huggingface_hub import CommitOperationAdd
    ops = [
        CommitOperationAdd(path_in_repo=PARQUET_PATH, path_or_fileobj=parquet_bytes),
        CommitOperationAdd(path_in_repo=README_PATH, path_or_fileobj=readme_bytes),
        CommitOperationAdd(path_in_repo=HASH_PATH, path_or_fileobj=new_hash.encode()),
    ]
    short_sha = (source_sha or "")[:7]
    msg = f"Update leaderboard: {len(df)} models" + (f" (source {short_sha})" if short_sha else "")
    api.create_commit(
        repo_id=DATASET_REPO, repo_type="dataset",
        operations=ops, commit_message=msg,
    )
    logger.info("Published %d rows to %s", len(df), DATASET_REPO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
