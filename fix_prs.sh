#!/bin/bash

CLAUDE_OPUS_46_METADATA='{
  "agent_name": "OpenHands",
  "agent_version": "v1.15.0",
  "model": "claude-opus-4-6",
  "openness": "closed_api_available",
  "country": "us",
  "tool_usage": "standard",
  "directory_name": "claude-opus-4-6",
  "release_date": "2026-02-05",
  "supports_vision": true,
  "input_price": 5.0,
  "output_price": 25.0,
  "cache_read_price": 0.5,
  "cache_write_price": 6.25
}'

CLAUDE_SONNET_45_METADATA='{
  "agent_name": "OpenHands",
  "agent_version": "v1.15.0",
  "model": "claude-sonnet-4-5",
  "openness": "closed_api_available",
  "country": "us",
  "tool_usage": "standard",
  "directory_name": "claude-sonnet-4-5",
  "release_date": "2025-09-29",
  "supports_vision": true,
  "input_price": 3.0,
  "output_price": 15.0,
  "cache_read_price": 0.3,
  "cache_write_price": 3.75
}'

fix_pr() {
    local BRANCH_NAME=$1
    local METADATA_PATH=$2
    local METADATA_CONTENT=$3
    local PR_NUM=$4
    
    echo "=== Fixing PR #$PR_NUM: $BRANCH_NAME ==="
    
    git checkout main
    git branch -D fix-pr-$PR_NUM 2>/dev/null || true
    git checkout origin/$BRANCH_NAME -b fix-pr-$PR_NUM
    
    echo "$METADATA_CONTENT" > "$METADATA_PATH"
    
    git add -A
    git commit -m "Fix metadata.json: Add required fields

Co-authored-by: openhands <openhands@all-hands.dev>"
    
    git push origin fix-pr-$PR_NUM:$BRANCH_NAME
    
    echo "=== Done fixing PR #$PR_NUM ==="
    echo ""
}

# PR #763 - claude-opus-4-6 gaia
fix_pr "eval/openhands_subagents/claude-opus-4-6/gaia-20260330-235733" \
       "alternative_agents/openhands_subagents/claude-opus-4-6/metadata.json" \
       "$CLAUDE_OPUS_46_METADATA" \
       763

# PR #756 - claude-opus-4-6 commit0
fix_pr "eval/openhands_subagents/claude-opus-4-6/commit0-20260330-152800" \
       "alternative_agents/openhands_subagents/claude-opus-4-6/metadata.json" \
       "$CLAUDE_OPUS_46_METADATA" \
       756

# PR #754 - claude-sonnet-4-5 commit0
fix_pr "eval/openhands_subagents/claude-sonnet-4-5/commit0-20260330-152631" \
       "alternative_agents/openhands_subagents/claude-sonnet-4-5/metadata.json" \
       "$CLAUDE_SONNET_45_METADATA" \
       754

# PR #753 - claude-sonnet-4-5 gaia
fix_pr "eval/openhands_subagents/claude-sonnet-4-5/gaia-20260330-152535" \
       "alternative_agents/openhands_subagents/claude-sonnet-4-5/metadata.json" \
       "$CLAUDE_SONNET_45_METADATA" \
       753

# PR #752 - claude-opus-4-6 swe-bench-multimodal
fix_pr "eval/openhands_subagents/claude-opus-4-6/swe-bench-multimodal-20260330-152228" \
       "alternative_agents/openhands_subagents/claude-opus-4-6/metadata.json" \
       "$CLAUDE_OPUS_46_METADATA" \
       752

# PR #751 - claude-sonnet-4-5 swe-bench-multimodal
fix_pr "eval/openhands_subagents/claude-sonnet-4-5/swe-bench-multimodal-20260330-151838" \
       "alternative_agents/openhands_subagents/claude-sonnet-4-5/metadata.json" \
       "$CLAUDE_SONNET_45_METADATA" \
       751

# PR #750 - claude-opus-4-6 swe-bench
fix_pr "eval/openhands_subagents/claude-opus-4-6/swe-bench-20260330-151736" \
       "alternative_agents/openhands_subagents/claude-opus-4-6/metadata.json" \
       "$CLAUDE_OPUS_46_METADATA" \
       750

# PR #749 - claude-sonnet-4-5 swe-bench
fix_pr "eval/openhands_subagents/claude-sonnet-4-5/swe-bench-20260330-151613" \
       "alternative_agents/openhands_subagents/claude-sonnet-4-5/metadata.json" \
       "$CLAUDE_SONNET_45_METADATA" \
       749

echo "All PRs fixed!"
