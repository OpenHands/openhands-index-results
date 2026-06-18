#!/usr/bin/env python3
"""Flag score entries whose cost_per_instance came from the unreliable estimate
instead of real proxy spend (advisory, warn-only).

Each run's ``cost_report.jsonl`` (fetched as a sibling of ``results.tar.gz``)
holds two costs: ``summary.total_cost`` (the SDK estimate) and
``proxy_cost_summary.total_proxy_cost`` (real LiteLLM-proxy spend). They diverge
sharply when an agent misreports cached input -- e.g. gemini-cli on the ACP path
reports ``cache_read_tokens=0``, inflating the estimate ~3x. If a leaderboard
``cost_per_instance`` was taken from that estimate it is wrong, so this warns
when an entry looks estimate-sourced. It never fails the build.

Usage: ``python scripts/compare_costs.py [scores.json ...]`` (no args = scan
``results/`` and ``alternative_agents/``). See the PR for the
sibling-availability limitation.
"""

import contextlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Final, Optional

# Estimate/proxy ratio above which a non-proxy-sourced cost is suspicious. The
# cache_read=0 bug inflates ~3x; clean runs sit near 1.0-1.25x.
WARN_RATIO: Final[float] = 1.5
# Slack for matching the rounded cost_per_instance to total / N.
MATCH_TOLERANCE: Final[float] = 0.10
HTTP_TIMEOUT: Final[int] = 20

# Instance counts the index divides by (mirrors push-to-index). swe-bench-
# multimodal has shipped against both the 68 (curated) and 102 (full) splits.
INSTANCE_COUNTS: Final[dict[str, tuple[int, ...]]] = {
    "swe-bench": (500,),
    "swt-bench": (433,),
    "gaia": (165,),
    "commit0": (16,),
    "swe-bench-multimodal": (68, 102),
}
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def fetch_report(full_archive: str) -> Optional[dict]:
    """Fetch the cost_report.jsonl sibling of a benchmark-format archive URL.

    Returns None for legacy URLs (no sibling) or any network/parse error.
    """
    if not full_archive.endswith("/results.tar.gz"):
        return None
    url = full_archive[: -len("results.tar.gz")] + "cost_report.jsonl"
    with contextlib.suppress(OSError, ValueError):
        req = urllib.request.Request(url, headers={"User-Agent": "compare-costs"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read())
    return None


def cost_source(benchmark: str, est: float, proxy: float, cpi: float) -> str:
    """Classify cost_per_instance as 'proxy', 'estimate', or 'unknown'.

    Matches cpi against ``total / N`` for each candidate instance count, picking
    the closest within tolerance.
    """
    best, best_dist = "unknown", MATCH_TOLERANCE
    for n in INSTANCE_COUNTS.get(benchmark, ()):
        for source, total in (("proxy", proxy), ("estimate", est)):
            per_instance = total / n
            dist = abs(cpi - per_instance) / per_instance
            if dist < best_dist:
                best, best_dist = source, dist
    return best


def review_entry(label: str, entry: dict) -> tuple[str, str]:
    """Return (status, message) for one entry; status is 'ok', 'skip' or 'warn'."""
    cpi = entry.get("cost_per_instance")
    if not cpi:
        return "skip", f"{label}: no cost_per_instance"

    report = fetch_report(entry.get("full_archive", ""))
    if report is None:
        return "skip", f"{label}: cost_report.jsonl unavailable"

    est = (report.get("summary") or {}).get("total_cost") or 0.0
    proxy_summary = report.get("proxy_cost_summary") or {}
    proxy = proxy_summary.get("total_proxy_cost") or 0.0
    if est <= 0 or proxy <= 0:
        return "skip", f"{label}: estimate or proxy total missing"
    if proxy_summary.get("zero_proxy_cost_instances"):
        return "skip", f"{label}: proxy spend incomplete"

    ratio = est / proxy
    source = cost_source(entry.get("benchmark", ""), est, proxy, cpi)
    facts = f"{label}: estimate=${est:,.0f} proxy=${proxy:,.0f} ratio={ratio:.1f}x cost/inst=${cpi}"

    # A high ratio only matters when the cost is not confirmed proxy-sourced.
    if ratio < WARN_RATIO or source == "proxy":
        return "ok", f"{facts} ({source})"
    hint = "re-push with --use-proxy-costs" if source == "estimate" else "confirm --use-proxy-costs was used"
    return "warn", f"{facts} -- looks {source}-sourced, {hint}"


def score_files(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths if p.strip().endswith("scores.json")]
    return sorted(
        f for top in ("results", "alternative_agents")
        for f in (REPO_ROOT / top).rglob("scores.json")
    )


def main(argv: list[str]) -> int:
    files = score_files(argv)
    print(f"compare_costs: checking {len(files)} scores.json file(s)")
    warnings = 0
    for path in files:
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"  SKIP {path}: unreadable ({exc})")
            continue
        rel = path
        with contextlib.suppress(ValueError):
            rel = path.resolve().relative_to(REPO_ROOT)
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            label = f"{rel} [{entry.get('benchmark', '?')}]"
            try:
                status, message = review_entry(label, entry)
            except Exception as exc:  # advisory check must never fail the build
                status, message = "skip", f"{label}: check error ({exc})"
            print(f"  {status.upper():4} {message}")
            warnings += status == "warn"

    print(f"\ncompare_costs: {warnings} cost-divergence warning(s) (advisory; never blocks).")
    return 0  # warn-only


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
