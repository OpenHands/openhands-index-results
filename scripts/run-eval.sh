#!/bin/bash
# Run evaluation and push results to the index repository.
#
# This script is designed to be called after an evaluation run completes.
# It sends a Slack notification and then pushes results to the index repo.
#
# Required environment variables:
#   PR_TO_INDEX - GitHub token for creating PRs to the index repository
#
# Optional environment variables:
#   SLACK_WEBHOOK_URL - Slack webhook URL for notifications
#   BENCHMARK - Benchmark name (default: swe-bench)
#   MODEL_NAME - Model name (required)
#   REPORT_PATH - Path to output.report.json (default: output.report.json)
#   COST_REPORT_PATH - Path to cost_report.jsonl (default: cost_report.jsonl)
#   AGENT_NAME - Agent name (default: OpenHands CodeAct)
#   AGENT_VERSION - Agent version (default: unknown)
#   OPENNESS - Model openness (default: closed_api_available)
#   TOOL_USAGE - Tool usage (default: standard)
#   DRY_RUN - Set to "true" to skip actual PR creation

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
BENCHMARK="${BENCHMARK:-swe-bench}"
REPORT_PATH="${REPORT_PATH:-output.report.json}"
COST_REPORT_PATH="${COST_REPORT_PATH:-cost_report.jsonl}"
AGENT_NAME="${AGENT_NAME:-OpenHands CodeAct}"
AGENT_VERSION="${AGENT_VERSION:-unknown}"
OPENNESS="${OPENNESS:-closed_api_available}"
TOOL_USAGE="${TOOL_USAGE:-standard}"
WORK_DIR="${WORK_DIR:-/tmp/index-repo}"
DRY_RUN="${DRY_RUN:-false}"

# Function to send Slack notification
send_slack_notification() {
    local message="$1"
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\": \"$message\"}" \
            "$SLACK_WEBHOOK_URL" || true
    fi
}

# Function to push results to index
push_to_index() {
    echo "=== Pushing results to index repository ==="

    if [ -z "$MODEL_NAME" ]; then
        echo "Error: MODEL_NAME environment variable is required"
        return 1
    fi

    if [ -z "$PR_TO_INDEX" ] && [ "$DRY_RUN" != "true" ]; then
        echo "Error: PR_TO_INDEX environment variable is required"
        return 1
    fi

    local dry_run_flag=""
    if [ "$DRY_RUN" = "true" ]; then
        dry_run_flag="--dry-run"
    fi

    python3 "$SCRIPT_DIR/push_to_index.py" \
        --benchmark "$BENCHMARK" \
        --model "$MODEL_NAME" \
        --report-path "$REPORT_PATH" \
        --cost-report-path "$COST_REPORT_PATH" \
        --work-dir "$WORK_DIR" \
        --agent-name "$AGENT_NAME" \
        --agent-version "$AGENT_VERSION" \
        --openness "$OPENNESS" \
        --tool-usage "$TOOL_USAGE" \
        $dry_run_flag

    return $?
}

# Main execution
main() {
    echo "=== Evaluation Post-Processing ==="
    echo "Benchmark: $BENCHMARK"
    echo "Model: $MODEL_NAME"
    echo "Report path: $REPORT_PATH"
    echo "Cost report path: $COST_REPORT_PATH"

    # Send Slack notification about evaluation completion
    send_slack_notification "Evaluation completed for $MODEL_NAME on $BENCHMARK. Pushing results to index..."

    # Push results to index repository
    if push_to_index; then
        send_slack_notification "Successfully created PR for $MODEL_NAME on $BENCHMARK results."
        echo "=== Results pushed successfully ==="
    else
        send_slack_notification "Failed to push results for $MODEL_NAME on $BENCHMARK."
        echo "=== Failed to push results ==="
        exit 1
    fi
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
