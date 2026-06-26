#!/usr/bin/env python3
"""Extract per-instance benchmark outcomes from archived evaluation results."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from io import BytesIO
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
    pricing: "Pricing"
    cost_per_instance: float | None


@dataclass(frozen=True)
class Pricing:
    input_cache_miss_cost: float
    input_cache_hit_cost: float
    output_cost: float

    @property
    def input_miss_per_token(self) -> float:
        return self.input_cache_miss_cost / 1_000_000

    @property
    def input_hit_per_token(self) -> float:
        return self.input_cache_hit_cost / 1_000_000

    @property
    def output_per_token(self) -> float:
        return self.output_cost / 1_000_000


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
            pricing = _pricing_from_model_dir(scores_path.parent)
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
                    pricing=pricing,
                    cost_per_instance=_number_or_none(entry.get("cost_per_instance")),
                )


def _pricing_from_model_dir(model_dir: Path) -> Pricing:
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    input_price = metadata.get("input_price")
    output_price = metadata.get("output_price")
    cache_read_price = metadata.get("cache_read_price")

    if input_price is None:
        raise ValueError(f"{metadata_path} missing input_price")
    if output_price is None:
        raise ValueError(f"{metadata_path} missing output_price")
    if cache_read_price is None:
        cache_read_price = input_price

    return Pricing(
        input_cache_miss_cost=float(input_price),
        input_cache_hit_cost=float(cache_read_price),
        output_cost=float(output_price),
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


def _extract_usage_from_metrics(metrics: dict[str, Any] | None) -> tuple[int, int, int]:
    usage = ((metrics or {}).get("accumulated_token_usage") or {})
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(usage.get("cache_read_tokens") or 0),
    )


def _has_usage(prompt_tokens: int, completion_tokens: int, cache_read_tokens: int) -> bool:
    return prompt_tokens != 0 or completion_tokens != 0 or cache_read_tokens != 0


def _calculate_instance_cost(
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int,
    pricing: Pricing,
) -> float:
    non_cached_prompt_tokens = max(prompt_tokens - cache_read_tokens, 0)
    return (
        non_cached_prompt_tokens * pricing.input_miss_per_token
        + cache_read_tokens * pricing.input_hit_per_token
        + completion_tokens * pricing.output_per_token
    )


def _extract_usage_from_base_state(base_state: dict[str, Any]) -> tuple[int, int, int]:
    usage_to_metrics = ((base_state.get("stats") or {}).get("usage_to_metrics") or {})

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cache_read_tokens = 0

    for metrics in usage_to_metrics.values():
        prompt_tokens, completion_tokens, cache_read_tokens = _extract_usage_from_metrics(metrics)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_cache_read_tokens += cache_read_tokens

    return total_prompt_tokens, total_completion_tokens, total_cache_read_tokens


def _collect_conversation_usage_by_instance(
    tf: tarfile.TarFile,
) -> dict[str, tuple[int, int, int]]:
    usage_by_instance: dict[str, tuple[int, int, int]] = {}
    conv_archives = [
        member
        for member in tf.getmembers()
        if member.name.endswith(".tar.gz") and "/conversations/" in member.name
    ]

    for conv_archive in conv_archives:
        try:
            extracted = tf.extractfile(conv_archive)
            if extracted is None:
                continue
            conv_tar_data = extracted.read()
            instance_id = conv_archive.name.split("/conversations/")[-1].removesuffix(".tar.gz")

            with tarfile.open(fileobj=BytesIO(conv_tar_data), mode="r:gz") as conv_tf:
                base_state_file = next(
                    (
                        member
                        for member in conv_tf.getmembers()
                        if member.name.endswith("base_state.json")
                    ),
                    None,
                )
                if base_state_file is None:
                    continue

                fileobj = conv_tf.extractfile(base_state_file)
                if fileobj is None:
                    continue
                base_state = json.loads(fileobj.read())
                usage_by_instance[instance_id] = _extract_usage_from_base_state(base_state)
        except Exception as exc:
            logger.debug("Skipping unreadable conversation archive %s: %s", conv_archive.name, exc)

    return usage_by_instance


def _add_usage_cost(
    costs_by_instance: dict[str, float],
    instance_id: str,
    usage: tuple[int, int, int],
    pricing: Pricing,
) -> None:
    prompt_tokens, completion_tokens, cache_read_tokens = usage
    if not _has_usage(prompt_tokens, completion_tokens, cache_read_tokens):
        return
    costs_by_instance[instance_id] = costs_by_instance.get(instance_id, 0.0) + _calculate_instance_cost(
        prompt_tokens,
        completion_tokens,
        cache_read_tokens,
        pricing,
    )


def _costs_from_jsonl_members(
    tf: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    pricing: Pricing,
) -> dict[str, float]:
    costs_by_instance: dict[str, float] = {}
    instance_ids: set[str] = set()
    instance_ids_with_jsonl_usage: set[str] = set()

    for member in members:
        fileobj = tf.extractfile(member)
        if fileobj is None:
            continue

        for raw_line in fileobj:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSONL row in %s", member.name)
                continue
            if not isinstance(row, dict):
                continue

            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str):
                continue
            instance_ids.add(instance_id)

            usage = _extract_usage_from_metrics(row.get("metrics"))
            if not _has_usage(*usage):
                continue
            instance_ids_with_jsonl_usage.add(instance_id)
            _add_usage_cost(costs_by_instance, instance_id, usage, pricing)

    missing_instance_ids = instance_ids - instance_ids_with_jsonl_usage
    if missing_instance_ids:
        conversation_usage_by_instance = _collect_conversation_usage_by_instance(tf)
        for instance_id in sorted(missing_instance_ids):
            usage = conversation_usage_by_instance.get(instance_id)
            if usage is not None:
                _add_usage_cost(costs_by_instance, instance_id, usage, pricing)

    return costs_by_instance


def _costs_from_conversation_archives(
    tf: tarfile.TarFile,
    pricing: Pricing,
) -> dict[str, float]:
    costs_by_instance: dict[str, float] = {}
    for instance_id, usage in _collect_conversation_usage_by_instance(tf).items():
        _add_usage_cost(costs_by_instance, instance_id, usage, pricing)
    return costs_by_instance


def _calculated_costs_by_instance(
    tf: tarfile.TarFile,
    pricing: Pricing | None,
) -> dict[str, float]:
    if pricing is None:
        return {}

    critic_members = sorted(
        [
            member
            for member in tf.getmembers()
            if member.isfile() and re.search(r"(^|/)output\.critic_attempt_\d+\.jsonl$", member.name)
        ],
        key=lambda member: member.name,
    )
    output_members = sorted(
        [
            member
            for member in tf.getmembers()
            if member.isfile() and re.search(r"(^|/)output\.jsonl$", member.name)
        ],
        key=lambda member: member.name,
    )

    if critic_members:
        return _costs_from_jsonl_members(tf, critic_members, pricing)
    if output_members:
        return _costs_from_jsonl_members(tf, output_members, pricing)
    return _costs_from_conversation_archives(tf, pricing)


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
    calculated_costs: dict[str, float] | None = None,
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
        cost = (calculated_costs or {}).get(instance_id)
        if cost is None:
            cost = _cost_from_output_row(row)
        _merge_record(records, benchmark, instance_id, status, source_archive, source_path, cost)


def extract_records(
    archive_path: Path,
    benchmark: str,
    source_archive: str,
    pricing: Pricing | None = None,
) -> list[dict[str, Any]]:
    fallback_records: dict[str, dict[str, Any]] = {}
    top_report_found = False
    with tarfile.open(archive_path, "r:gz") as tf:
        calculated_costs = _calculated_costs_by_instance(tf, pricing)
        for member in tf.getmembers():
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
                    calculated_costs=calculated_costs,
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
                            cost=calculated_costs.get(record["instance_id"]),
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


def _normalize_costs_to_target_mean(
    records: list[dict[str, Any]],
    target_mean: float | None,
) -> None:
    if target_mean is None or target_mean <= 0:
        return

    graded_records = [
        record
        for record in records
        if record.get("resolved") is not None and record.get("cost") is not None
    ]
    if not graded_records:
        missing_cost_records = [
            record
            for record in records
            if record.get("resolved") is not None and record.get("cost") is None
        ]
        if not missing_cost_records:
            logger.warning("Cannot normalize costs: no graded records have costs")
            return
        logger.info(
            "Assigning published cost_per_instance %.4f to %s graded records without costs",
            target_mean,
            len(missing_cost_records),
        )
        for record in missing_cost_records:
            record["cost"] = target_mean
        return

    current_mean = sum(record["cost"] for record in graded_records) / len(graded_records)
    if current_mean <= 0:
        logger.info(
            "Replacing zero graded mean with published cost_per_instance %.4f",
            target_mean,
        )
        for record in graded_records:
            record["cost"] = target_mean
        return

    scale = target_mean / current_mean
    if abs(scale - 1.0) > 0.01:
        logger.info(
            "Normalizing per-instance costs by %.4f to match published cost_per_instance %.4f",
            scale,
            target_mean,
        )

    for record in records:
        if record.get("cost") is not None:
            record["cost"] *= scale


def extract_all(
    repo_root: Path,
    cache_dir: Path,
    force_download: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    normalize_costs: bool = True,
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
            records = extract_records(
                archive_path,
                ref.benchmark,
                ref.archive_url,
                pricing=ref.pricing,
            )
        except Exception as exc:
            failures += 1
            logger.error("Failed to extract %s %s: %s", rel, ref.benchmark, exc)
            continue

        if not records:
            failures += 1
            logger.error("No instance records extracted for %s %s", rel, ref.benchmark)
            continue

        if normalize_costs:
            _normalize_costs_to_target_mean(records, ref.cost_per_instance)
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
    parser.add_argument(
        "--no-normalize-costs",
        action="store_true",
        help="Do not scale calculated per-instance costs to scores.json cost_per_instance",
    )
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
        normalize_costs=not args.no_normalize_costs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
