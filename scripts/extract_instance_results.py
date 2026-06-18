#!/usr/bin/env python3
"""Extract per-instance benchmark outcomes from archived evaluation results."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

logger = logging.getLogger("extract_instance_results")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "instance-results"
RESULT_ROOTS = ("results", "alternative_agents")

STATUS_FIELDS = (
    ("invalidated", "invalidated_ids"),
    ("error", "error_ids"),
    ("incomplete", "incomplete_ids"),
    ("empty_patch", "empty_patch_ids"),
    ("resolved", "resolved_ids"),
    ("unresolved", "unresolved_ids"),
    ("completed", "completed_ids"),
    ("submitted", "submitted_ids"),
)

TERMINAL_STATUSES = {
    "resolved": True,
    "unresolved": False,
    "invalidated": False,
    "error": False,
    "incomplete": False,
    "empty_patch": False,
}


@dataclass(frozen=True)
class ScoreRef:
    score_path: Path
    model_dir: Path
    benchmark: str
    archive_url: str


def _safe_name(url: str) -> str:
    return "".join(c if c.isalnum() or c in ".-" else "_" for c in url)


def _download(url: str, cache_dir: Path, force: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _safe_name(url)
    if path.exists() and not force:
        return path

    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    logger.info("Downloading %s", url)
    subprocess.run(
        ["curl", "-L", "--fail", "--silent", "--show-error", url, "-o", str(tmp)],
        check=True,
    )
    tmp.replace(path)
    return path


def _iter_score_refs(repo_root: Path) -> Iterable[ScoreRef]:
    for root_name in RESULT_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for scores_path in sorted(root.glob("**/scores.json")):
            scores = json.loads(scores_path.read_text())
            if not isinstance(scores, list):
                continue
            for entry in scores:
                url = entry.get("full_archive")
                benchmark = entry.get("benchmark")
                if not url or not benchmark:
                    continue
                yield ScoreRef(
                    score_path=scores_path,
                    model_dir=scores_path.parent,
                    benchmark=benchmark,
                    archive_url=url,
                )


def _is_appledouble(name: str) -> bool:
    return PurePosixPath(name).name.startswith("._")


def _merge_record(
    records: dict[str, dict[str, Any]],
    benchmark: str,
    instance_id: str,
    status: str,
    source_archive: str,
    source_path: str,
    cost: float | None = None,
) -> None:
    existing = records.get(instance_id)
    if existing is not None:
        if cost is not None and existing.get("cost") is None:
            existing["cost"] = cost
        if existing["status"] != "unknown" or status == "unknown":
            return

    resolved = TERMINAL_STATUSES.get(status)
    records[instance_id] = {
        "benchmark": benchmark,
        "instance_id": instance_id,
        "status": status,
        "resolved": resolved,
        "cost": cost if cost is not None else existing.get("cost") if existing else None,
        "source_archive": source_archive,
        "source_path": source_path,
    }


def _records_from_top_report(
    report: dict[str, Any],
    benchmark: str,
    source_archive: str,
    source_path: str,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for status, field in STATUS_FIELDS:
        ids = report.get(field)
        if not isinstance(ids, list):
            continue
        for instance_id in ids:
            if isinstance(instance_id, str):
                _merge_record(records, benchmark, instance_id, status, source_archive, source_path)

    return [records[k] for k in sorted(records)]


def _records_from_instance_report_payload(
    report: dict[str, Any],
    benchmark: str,
    source_archive: str,
    source_path: str,
    records: dict[str, dict[str, Any]],
) -> None:
    for instance_id, payload in report.items():
        if not isinstance(instance_id, str) or not isinstance(payload, dict):
            continue
        resolved = payload.get("resolved")
        if resolved is True:
            status = "resolved"
        elif payload.get("patch_is_None") is True:
            status = "empty_patch"
        elif payload.get("patch_exists") is False:
            status = "empty_patch"
        elif resolved is False:
            status = "unresolved"
        else:
            status = "unknown"
        _merge_record(records, benchmark, instance_id, status, source_archive, source_path)


def _status_from_output_row(row: dict[str, Any], default_status: str = "unknown") -> str:
    if row.get("error"):
        return "error"

    test_result = row.get("test_result")
    if not isinstance(test_result, dict):
        return default_status

    score = test_result.get("score")
    if isinstance(score, bool):
        return "resolved" if score else "unresolved"

    resolved = test_result.get("resolved")
    if isinstance(resolved, bool):
        return "resolved" if resolved else "unresolved"

    eval_result = test_result.get("eval_result")
    if isinstance(eval_result, dict):
        passed = eval_result.get("passed")
        if isinstance(passed, bool):
            return "resolved" if passed else "unresolved"
        if isinstance(passed, (int, float)):
            return "resolved" if passed >= 1 else "unresolved"

    git_patch = test_result.get("git_patch")
    if git_patch == "":
        return "empty_patch"

    return default_status


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _cost_from_output_row(row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        cost = _number_or_none(metrics.get("accumulated_cost"))
        if cost is not None:
            return cost

    test_result = row.get("test_result")
    if isinstance(test_result, dict):
        cost = _number_or_none(test_result.get("proxy_cost"))
        if cost is not None:
            return cost

    if isinstance(metrics, dict):
        costs = metrics.get("costs")
        if isinstance(costs, list):
            total = 0.0
            found = False
            for item in costs:
                if not isinstance(item, dict):
                    continue
                cost = _number_or_none(item.get("cost"))
                if cost is None:
                    continue
                total += cost
                found = True
            if found:
                return total

    for key in ("total_cost", "cost"):
        cost = _number_or_none(row.get(key))
        if cost is not None:
            return cost

    return None


def _records_from_output_jsonl_payload(
    raw: bytes,
    benchmark: str,
    source_archive: str,
    source_path: str,
    records: dict[str, dict[str, Any]],
    default_status: str = "unknown",
) -> None:
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Skipping malformed JSONL row in %s", source_path)
            continue
        if not isinstance(row, dict):
            continue
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str):
            continue
        status = _status_from_output_row(row, default_status=default_status)
        cost = _cost_from_output_row(row)
        _merge_record(records, benchmark, instance_id, status, source_archive, source_path, cost)


def extract_records(archive_path: Path, benchmark: str, source_archive: str) -> list[dict[str, Any]]:
    fallback_records: dict[str, dict[str, Any]] = {}
    top_report_found = False
    with tarfile.open(archive_path, "r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if _is_appledouble(name):
                continue
            path = PurePosixPath(name)
            if path.name not in {"output.report.json", "report.json", "output.jsonl", "output_errors.jsonl"}:
                continue

            fileobj = tf.extractfile(member)
            if fileobj is None:
                continue
            raw = fileobj.read()

            if path.name in {"output.jsonl", "output_errors.jsonl"}:
                _records_from_output_jsonl_payload(
                    raw,
                    benchmark=benchmark,
                    source_archive=source_archive,
                    source_path=name,
                    records=fallback_records,
                    default_status="error" if path.name == "output_errors.jsonl" else "unknown",
                )
                continue

            report = json.loads(raw.decode("utf-8"))
            if not isinstance(report, dict):
                continue

            if path.name == "output.report.json":
                records = _records_from_top_report(report, benchmark, source_archive, name)
                if records:
                    for record in records:
                        _merge_record(
                            fallback_records,
                            benchmark=benchmark,
                            instance_id=record["instance_id"],
                            status=record["status"],
                            source_archive=source_archive,
                            source_path=name,
                        )
                    top_report_found = True
                continue

            if not top_report_found and "run_evaluation" in path.parts:
                _records_from_instance_report_payload(
                    report,
                    benchmark=benchmark,
                    source_archive=source_archive,
                    source_path=name,
                    records=fallback_records,
                )

    return [fallback_records[k] for k in sorted(fallback_records)]


def _output_path(model_dir: Path, benchmark: str) -> Path:
    return model_dir / "instance_results" / f"{benchmark}.json"


def _write_resolved_map(path: Path, records: list[dict[str, Any]], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    results_by_instance = {
        record["instance_id"]: {
            "resolved": record["resolved"],
            "cost": record.get("cost"),
        }
        for record in sorted(records, key=lambda item: item["instance_id"])
    }
    path.write_text(json.dumps(results_by_instance, sort_keys=True, separators=(",", ":")) + "\n")


def extract_all(
    repo_root: Path,
    cache_dir: Path,
    force_download: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
) -> int:
    refs = list(_iter_score_refs(repo_root))
    if limit is not None:
        refs = refs[:limit]

    failures = 0
    for index, ref in enumerate(refs, start=1):
        rel = ref.score_path.relative_to(repo_root)
        logger.info("[%s/%s] %s %s", index, len(refs), rel, ref.benchmark)
        try:
            archive_path = _download(ref.archive_url, cache_dir, force=force_download)
            records = extract_records(archive_path, ref.benchmark, ref.archive_url)
        except Exception as exc:
            failures += 1
            logger.error("Failed to extract %s %s: %s", rel, ref.benchmark, exc)
            continue

        if not records:
            failures += 1
            logger.error("No instance records extracted for %s %s", rel, ref.benchmark)
            continue

        _write_resolved_map(_output_path(ref.model_dir, ref.benchmark), records, dry_run=dry_run)

    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--clean-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.clean_cache and args.cache_dir.exists():
        shutil.rmtree(args.cache_dir)
    return extract_all(
        repo_root=args.repo_root,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
