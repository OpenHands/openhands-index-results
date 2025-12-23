#!/usr/bin/env python3
"""
Push evaluation results to the OpenHands index repository.

This script reads evaluation results and creates a PR to add them to the
openhands-index-results repository for display on the leaderboard.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Benchmark name mapping from eval benchmark names to index benchmark names
BENCHMARK_NAME_MAP = {
    "swe-bench": "swe-bench",
    "swe_bench": "swe-bench",
    "swebench": "swe-bench",
    "swt-bench": "swt-bench",
    "swt_bench": "swt-bench",
    "swtbench": "swt-bench",
    "gaia": "gaia",
    "commit0": "commit0",
    "commit-0": "commit0",
    "multi-swe-bench": "multi-swe-bench",
    "multi_swe_bench": "multi-swe-bench",
    "multiswebench": "multi-swe-bench",
    "swe-bench-multimodal": "swe-bench-multimodal",
    "swe_bench_multimodal": "swe-bench-multimodal",
    "swebench-multimodal": "swe-bench-multimodal",
}

# Valid benchmark names for the index
VALID_BENCHMARKS = {
    "swe-bench",
    "swt-bench",
    "gaia",
    "commit0",
    "multi-swe-bench",
    "swe-bench-multimodal",
}

INDEX_REPO_URL = "https://github.com/OpenHands/openhands-index-results.git"
INDEX_REPO_NAME = "openhands-index-results"


def load_json(file_path: Path) -> dict | list | None:
    """Load JSON file, return None if file doesn't exist or is invalid."""
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_jsonl(file_path: Path) -> list[dict]:
    """Load JSONL file, return empty list if file doesn't exist or is invalid."""
    if not file_path.exists():
        return []
    results = []
    try:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except (json.JSONDecodeError, IOError):
        return []
    return results


def calculate_accuracy(report_data: dict) -> float:
    """Calculate accuracy from output.report.json data."""
    resolved = report_data.get("resolved_instances", 0)
    submitted = report_data.get("submitted_instances", 0)
    if submitted == 0:
        return 0.0
    return round((resolved / submitted) * 100, 2)


def get_cost_and_duration(cost_report: list[dict]) -> tuple[float, float]:
    """Extract total_cost and total_duration from cost_report.jsonl."""
    total_cost = 0.0
    total_duration = 0.0
    for entry in cost_report:
        total_cost += entry.get("total_cost", 0) or 0
        total_duration += entry.get("total_duration", 0) or 0
    return round(total_cost, 2), round(total_duration, 2)


def normalize_benchmark_name(benchmark: str) -> str:
    """Normalize benchmark name to the standard index format."""
    normalized = benchmark.lower().strip()
    return BENCHMARK_NAME_MAP.get(normalized, normalized)


def generate_directory_name(model_name: str, date: datetime | None = None) -> str:
    """Generate directory name in YYYYMM_model format."""
    if date is None:
        date = datetime.now()
    year_month = date.strftime("%Y%m")
    # Clean model name for directory
    clean_model = model_name.replace("/", "-").replace(" ", "-")
    return f"{year_month}_{clean_model}"


def create_metadata(
    model_name: str,
    agent_name: str = "OpenHands CodeAct",
    agent_version: str = "unknown",
    openness: str = "closed_api_available",
    tool_usage: str = "standard",
    directory_name: str = "",
) -> dict:
    """Create metadata.json content."""
    return {
        "agent_name": agent_name,
        "agent_version": agent_version,
        "model": model_name,
        "openness": openness,
        "tool_usage": tool_usage,
        "submission_time": datetime.now().isoformat(),
        "directory_name": directory_name,
    }


def create_score_entry(
    benchmark: str,
    score: float,
    total_cost: float = 0,
    total_runtime: float = 0,
) -> dict:
    """Create a score entry for scores.json."""
    normalized_benchmark = normalize_benchmark_name(benchmark)
    return {
        "benchmark": normalized_benchmark,
        "score": score,
        "metric": "accuracy",
        "total_cost": total_cost,
        "total_runtime": total_runtime,
        "tags": [normalized_benchmark],
    }


def update_scores(existing_scores: list[dict], new_entry: dict) -> list[dict]:
    """Update scores list, replacing existing entry for same benchmark or adding new."""
    benchmark = new_entry["benchmark"]
    updated = False
    result = []
    for entry in existing_scores:
        if entry.get("benchmark") == benchmark:
            result.append(new_entry)
            updated = True
        else:
            result.append(entry)
    if not updated:
        result.append(new_entry)
    return result


def run_command(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=full_env,
    )
    return result.returncode, result.stdout, result.stderr


def clone_index_repo(work_dir: Path, token: str) -> Path:
    """Clone the index repository."""
    repo_path = work_dir / INDEX_REPO_NAME
    if repo_path.exists():
        # Pull latest changes
        run_command(["git", "fetch", "origin"], cwd=repo_path)
        run_command(["git", "checkout", "main"], cwd=repo_path)
        run_command(["git", "pull", "origin", "main"], cwd=repo_path)
        return repo_path

    # Clone with token
    auth_url = f"https://x-access-token:{token}@github.com/OpenHands/openhands-index-results.git"
    exit_code, stdout, stderr = run_command(
        ["git", "clone", auth_url, INDEX_REPO_NAME],
        cwd=work_dir,
    )
    if exit_code != 0:
        raise RuntimeError(f"Failed to clone repository: {stderr}")
    return repo_path


def create_branch(repo_path: Path, branch_name: str) -> None:
    """Create and checkout a new branch."""
    run_command(["git", "checkout", "-b", branch_name], cwd=repo_path)


def commit_and_push(repo_path: Path, branch_name: str, message: str) -> None:
    """Commit changes and push to remote."""
    run_command(["git", "add", "."], cwd=repo_path)
    run_command(["git", "commit", "-m", message], cwd=repo_path)
    exit_code, stdout, stderr = run_command(
        ["git", "push", "-u", "origin", branch_name],
        cwd=repo_path,
    )
    if exit_code != 0:
        raise RuntimeError(f"Failed to push: {stderr}")


def create_pull_request(
    token: str,
    branch_name: str,
    title: str,
    body: str,
    reviewer: str = "juanmichelini",
) -> dict:
    """Create a pull request using GitHub API."""
    import urllib.request
    import urllib.error

    api_url = "https://api.github.com/repos/OpenHands/openhands-index-results/pulls"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    data = {
        "title": title,
        "body": body,
        "head": branch_name,
        "base": "main",
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            pr_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"Failed to create PR: {e.code} - {error_body}")

    # Request review
    pr_number = pr_data["number"]
    review_url = f"https://api.github.com/repos/OpenHands/openhands-index-results/pulls/{pr_number}/requested_reviewers"
    review_data = {"reviewers": [reviewer]}
    review_req = urllib.request.Request(
        review_url,
        data=json.dumps(review_data).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(review_req) as response:
            pass
    except urllib.error.HTTPError:
        # Non-fatal: reviewer might not have access
        print(f"Warning: Could not request review from {reviewer}")

    return pr_data


def push_results_to_index(
    benchmark: str,
    model_name: str,
    report_path: Path,
    cost_report_path: Path,
    token: str,
    work_dir: Path,
    agent_name: str = "OpenHands CodeAct",
    agent_version: str = "unknown",
    openness: str = "closed_api_available",
    tool_usage: str = "standard",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Push evaluation results to the index repository.

    Returns a dict with the results of the operation.
    """
    # Load report data
    report_data = load_json(report_path)
    if report_data is None:
        report_data = {}

    # Calculate accuracy
    accuracy = calculate_accuracy(report_data)

    # Load cost report
    cost_report = load_jsonl(cost_report_path)
    total_cost, total_duration = get_cost_and_duration(cost_report)

    # Generate directory name
    dir_name = generate_directory_name(model_name)

    # Normalize benchmark name
    normalized_benchmark = normalize_benchmark_name(benchmark)
    if normalized_benchmark not in VALID_BENCHMARKS:
        raise ValueError(f"Invalid benchmark: {benchmark}. Valid benchmarks: {VALID_BENCHMARKS}")

    result = {
        "benchmark": normalized_benchmark,
        "model": model_name,
        "accuracy": accuracy,
        "total_cost": total_cost,
        "total_duration": total_duration,
        "directory_name": dir_name,
    }

    if dry_run:
        print(f"Dry run - would create/update results for {model_name} on {normalized_benchmark}")
        print(f"  Accuracy: {accuracy}%")
        print(f"  Total cost: ${total_cost}")
        print(f"  Total duration: {total_duration}s")
        return result

    # Clone repository
    print(f"Cloning index repository to {work_dir}...")
    repo_path = clone_index_repo(work_dir, token)

    # Create results directory
    results_dir = repo_path / "results" / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load or create metadata
    metadata_path = results_dir / "metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path) or {}
    else:
        metadata = create_metadata(
            model_name=model_name,
            agent_name=agent_name,
            agent_version=agent_version,
            openness=openness,
            tool_usage=tool_usage,
            directory_name=dir_name,
        )

    # Update submission time
    metadata["submission_time"] = datetime.now().isoformat()
    metadata["directory_name"] = dir_name

    # Write metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    # Load or create scores
    scores_path = results_dir / "scores.json"
    if scores_path.exists():
        scores = load_json(scores_path) or []
    else:
        scores = []

    # Create new score entry
    new_score = create_score_entry(
        benchmark=normalized_benchmark,
        score=accuracy,
        total_cost=total_cost,
        total_runtime=total_duration,
    )

    # Update scores
    scores = update_scores(scores, new_score)

    # Write scores
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")

    # Create branch
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    branch_name = f"results/{dir_name}-{normalized_benchmark}-{timestamp}"
    print(f"Creating branch: {branch_name}")
    create_branch(repo_path, branch_name)

    # Commit and push
    commit_message = f"Add {normalized_benchmark} results for {model_name}"
    print(f"Committing: {commit_message}")
    commit_and_push(repo_path, branch_name, commit_message)

    # Create PR
    pr_title = f"Add {normalized_benchmark} results for {model_name}"
    pr_body = f"""## Benchmark Results

This PR adds evaluation results for **{model_name}** on **{normalized_benchmark}**.

### Results Summary

| Metric | Value |
|--------|-------|
| Benchmark | {normalized_benchmark} |
| Model | {model_name} |
| Accuracy | {accuracy}% |
| Total Cost | ${total_cost} |
| Total Duration | {total_duration}s |

### Files Changed

- `results/{dir_name}/metadata.json`
- `results/{dir_name}/scores.json`

---
*This PR was automatically generated by the evaluation pipeline.*
"""

    print("Creating pull request...")
    pr_data = create_pull_request(
        token=token,
        branch_name=branch_name,
        title=pr_title,
        body=pr_body,
        reviewer="juanmichelini",
    )

    result["pr_url"] = pr_data.get("html_url")
    result["pr_number"] = pr_data.get("number")

    print(f"Pull request created: {result['pr_url']}")
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Push evaluation results to the OpenHands index repository"
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name (e.g., swe-bench, gaia, commit0)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("output.report.json"),
        help="Path to output.report.json",
    )
    parser.add_argument(
        "--cost-report-path",
        type=Path,
        default=Path("cost_report.jsonl"),
        help="Path to cost_report.jsonl",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("PR_TO_INDEX"),
        help="GitHub token for creating PR (default: PR_TO_INDEX env var)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/index-repo"),
        help="Working directory for cloning the repository",
    )
    parser.add_argument(
        "--agent-name",
        default="OpenHands CodeAct",
        help="Agent name for metadata",
    )
    parser.add_argument(
        "--agent-version",
        default="unknown",
        help="Agent version for metadata",
    )
    parser.add_argument(
        "--openness",
        default="closed_api_available",
        choices=["open_weights", "closed_api_available", "closed"],
        help="Model openness classification",
    )
    parser.add_argument(
        "--tool-usage",
        default="standard",
        choices=["standard", "custom", "none"],
        help="Tool usage classification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )

    args = parser.parse_args()

    if not args.token and not args.dry_run:
        print("Error: GitHub token required. Set PR_TO_INDEX environment variable or use --token")
        sys.exit(1)

    try:
        result = push_results_to_index(
            benchmark=args.benchmark,
            model_name=args.model,
            report_path=args.report_path,
            cost_report_path=args.cost_report_path,
            token=args.token or "",
            work_dir=args.work_dir,
            agent_name=args.agent_name,
            agent_version=args.agent_version,
            openness=args.openness,
            tool_usage=args.tool_usage,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
