# OpenHands Index Data Structure

This document describes the expected data structure for the `openhands-index-results` GitHub repository.

## Repository Structure

The data should be organized in the following structure:

```
openhands-index-results/
├── 1.0.0-dev1/              # Version directory (matches CONFIG_NAME in config.py)
│   ├── test.jsonl            # Test split results
│   ├── validation.jsonl      # Validation split results
│   ├── swe-bench.jsonl       # Individual benchmark results
│   ├── swe-bench-multimodal.jsonl
│   ├── swt-bench.jsonl
│   ├── commit0.jsonl
│   ├── gaia.jsonl
│   └── agenteval.json        # Configuration file
```

## File Formats

### Agent Directory Structure

Each agent has its own directory containing two files:

**metadata.json** - Agent and model information:
```json
{
  "agent_name": "OpenHands CodeAct",
  "agent_version": "v1.8.3",
  "model": "claude-4.5-opus",
  "openness": "closed_api_available",
  "country": "us",
  "tool_usage": "standard",
  "submission_time": "2026-01-27T01:24:15.735789+00:00",
  "directory_name": "claude-4.5-opus",
  "release_date": "2025-11-24",
  "parameter_count_b": null,
  "active_parameter_count_b": null
}
```

**scores.json** - Array of benchmark results:
```json
[
  {
    "benchmark": "swe-bench",
    "score": 76.6,
    "metric": "accuracy",
    "cost_per_instance": 1.82,
    "average_runtime": 325.0,
    "full_archive": "https://results.eval.all-hands.dev/eval-21370451733-...",
    "tags": ["swe-bench"],
    "agent_version": "v1.8.3",
    "submission_time": "2026-01-27T01:24:15.735789+00:00"
  }
]
```

### Configuration File (agenteval.json)

The configuration file defines the benchmark structure:

```json
{
  "suite_config": {
    "name": "openhands-index",
    "version": "1.0.0-dev1",
    "splits": [
      {
        "name": "test",
        "tasks": [
          {
            "name": "swe-bench",
            "tags": ["swe-bench"]
          },
          {
            "name": "swe-bench-multimodal",
            "tags": ["swe-bench-multimodal"]
          },
          {
            "name": "swt-bench",
            "tags": ["swt-bench"]
          },
          {
            "name": "commit0",
            "tags": ["commit0"]
          },
          {
            "name": "gaia",
            "tags": ["gaia"]
          }
        ]
      },
      {
        "name": "validation",
        "tasks": [
          {
            "name": "swe-bench",
            "tags": ["swe-bench"]
          },
          {
            "name": "swe-bench-multimodal",
            "tags": ["swe-bench-multimodal"]
          },
          {
            "name": "swt-bench",
            "tags": ["swt-bench"]
          },
          {
            "name": "commit0",
            "tags": ["commit0"]
          },
          {
            "name": "gaia",
            "tags": ["gaia"]
          }
        ]
      }
    ]
  }
}
```

## Data Loading Process

1. **GitHub Repository Check**: The app first attempts to clone the `openhands-index-results` repository
2. **Version Directory**: Looks for a directory matching `CONFIG_NAME` (currently "1.0.0-dev1")
3. **Fallback to Mock Data**: If GitHub data is unavailable, falls back to local mock data in `mock_results/`
4. **Data Extraction**: Copies data to `/tmp/oh_index/data/{version}/extracted/{version}/`

## Updating Data

To update the leaderboard data:

1. Push new JSONL files to the `openhands-index-results` repository
2. Ensure the version directory matches `CONFIG_NAME` in `config.py`
3. The app will automatically fetch the latest data on restart

## Mock Data

Mock data is stored in `mock_results/1.0.0-dev1/` and is used:
- During development and testing
- When the GitHub repository is unavailable
- As a template for the expected data format
