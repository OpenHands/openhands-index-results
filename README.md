# OpenHands Index Results

This repository contains benchmark results for various OpenHands agents and LLM configurations.

## Data Structure

### Agent-Centric Format (Recommended)

Results are organized in the `results/` directory with the following structure:

```
results/
├── {version}_{model_name}/
│   ├── metadata.json
│   └── scores.json
```

#### Directory Naming Convention

Each agent directory follows the format: `{version}_{model_name}/`
- `{version}`: Agent version (semantic version starting with 'v', e.g., `v1.8.3`)
- `{model_name}`: LLM model name (e.g., `claude-4.5-sonnet`, `gpt-5.2`)

#### metadata.json

Contains agent metadata and configuration:

```json
{
  "agent_name": "OpenHands CodeAct v2.0",
  "agent_version": "v1.8.3",
  "model": "claude-4.5-sonnet",
  "openness": "closed_api_available",
  "tool_usage": "standard",
  "submission_time": "2025-11-24T19:56:00.092895",
  "directory_name": "v1.8.3_claude-4.5-sonnet"
}
```

**Fields:**
- `agent_name`: Display name of the agent
- `agent_version`: Semantic version number (e.g., "1.0.0", "1.0.2")
- `model`: LLM model used
- `openness`: Model availability type
  - `closed_api_available`: Commercial API-based models
  - `open_api_available`: Open-source models with API access
  - `open_weights_available`: Open-weights models that can be self-hosted
- `tool_usage`: Agent tooling type
  - `standard`: Standard tool usage
  - `custom_interface`: Custom tool interface
- `submission_time`: ISO 8601 timestamp

#### scores.json

Contains benchmark scores and performance metrics:

```json
[
  {
    "benchmark": "swe-bench",
    "score": 45.1,
    "metric": "resolve_rate",
    "total_cost": 32.55,
    "average_runtime": 3600,
    "tags": ["bug_fixing"]
  },
  ...
]
```

**Fields:**
- `benchmark`: Benchmark identifier (e.g., "swe-bench", "commit0")
- `score`: Primary metric score (percentage or numeric value)
- `metric`: Type of metric (e.g., "resolve_rate", "success_rate")
- `total_cost`: Total API cost in USD
- `average_runtime`: Average runtime per instance in seconds (optional)
- `tags`: Category tags for grouping (e.g., ["bug_fixing"], ["app_creation"])

### Legacy Format (Backward Compatible)

The `1.0.0-dev1/` directory contains the original benchmark-centric JSONL files:
- `swe-bench.jsonl`
- `swe-bench-multimodal.jsonl`
- `commit0.jsonl`
- `swt-bench.jsonl`
- `gaia.jsonl`

This format is maintained for backward compatibility.

## Supported Benchmarks

### Bug Fixing
- **SWE-Bench**: Resolving GitHub issues from real Python repositories
- **SWE-Bench-Multimodal**: Similar to SWE-Bench with multimodal inputs

### App Creation
- **Commit0**: Building applications from scratch based on specifications

### Test Generation
- **SWT-Bench**: Generating comprehensive test suites

### Information Gathering
- **GAIA**: General AI assistant tasks requiring web search and reasoning

## Benchmark Categories

Results are grouped into 4 main categories on the leaderboard:

1. **Bug Fixing**: SWE-Bench, SWE-Bench-Multimodal
2. **App Creation**: Commit0
3. **Test Generation**: SWT-Bench
4. **Information Gathering**: GAIA

## Adding New Results

To add new benchmark results:

1. Create a directory following the naming convention: `results/{version}_{model_name}/`
2. Add `metadata.json` with agent configuration
3. Add `scores.json` with benchmark results
4. Commit and push to the repository

Example:
```bash
# Create directory
mkdir -p results/v1.8.3_claude-4.5-sonnet/

# Add metadata
cat > results/v1.8.3_claude-4.5-sonnet/metadata.json << 'EOF'
{
  "agent_name": "OpenHands CodeAct v2.0",
  "agent_version": "v1.8.3",
  "model": "claude-4.5-sonnet",
  "openness": "closed_api_available",
  "tool_usage": "standard",
  "submission_time": "2025-11-24T19:56:00.092895",
  "directory_name": "v1.8.3_claude-4.5-sonnet"
}
EOF

# Add scores
cat > results/v1.8.3_claude-4.5-sonnet/scores.json << 'EOF'
[
  {
    "benchmark": "swe-bench",
    "score": 45.1,
    "metric": "accuracy",
    "cost_per_instance": 0.412,
    "average_runtime": 3600,
    "tags": ["bug_fixing"]
  },
  ...
]
EOF

# Commit and push
git add results/v1.8.3_claude-4.5-sonnet/
git commit -m "Add results for OpenHands CodeAct v1.8.3 with Claude 4.5 Sonnet"
git push origin main
```

## Leaderboard

View the live leaderboard at: https://huggingface.co/spaces/OpenHands/openhands-index

## License

MIT License - See repository for details.
