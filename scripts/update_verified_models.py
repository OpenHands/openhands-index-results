#!/usr/bin/env python3
"""
Update verified_models.py in the software-agent-sdk repository.

This script reads complete-models.json and the current verified_models.py from
the software-agent-sdk repo, determines which completed models are missing from
the verified lists, and generates an updated verified_models.py.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Provider detection: ordered list of (prefix_pattern, provider_key)
PROVIDER_RULES = [
    (r"^gpt-", "openai"),
    (r"^o[0-9]", "openai"),
    (r"^codex-", "openai"),
    (r"^claude-", "anthropic"),
    (r"^gemini-", "gemini"),
    (r"^deepseek-", "deepseek"),
    (r"^kimi-", "moonshot"),
    (r"^minimax-", "minimax"),
    (r"^glm-", "glm"),
    (r"^nemotron-", "nvidia"),
    (r"^qwen", "qwen"),
]

VERIFIED_MODELS_URL = (
    "https://raw.githubusercontent.com/OpenHands/software-agent-sdk/"
    "main/openhands-sdk/openhands/sdk/llm/utils/verified_models.py"
)


def normalize_model_name(directory_name: str) -> str:
    """Normalize a model directory name to the verified model ID format."""
    return directory_name.lower()


def detect_provider(model_id: str) -> Optional[str]:
    """Detect the provider for a normalized model ID."""
    for pattern, provider in PROVIDER_RULES:
        if re.match(pattern, model_id):
            return provider
    return None


def extract_completed_model_ids(
    complete_models_path: Path,
    repo_root: Path,
) -> Set[str]:
    """
    Extract unique normalized model IDs from complete-models.json.

    Only includes models from 'results/' directories (standard OpenHands agent).
    """
    with open(complete_models_path, "r") as f:
        models = json.load(f)

    model_ids: Set[str] = set()
    for entry in models:
        model_path = entry["model-path"]
        # Only consider results/ directory (standard OpenHands agent)
        if not model_path.startswith("results/"):
            continue
        # Extract the model directory name (last component of path)
        dir_name = Path(model_path).name
        model_ids.add(normalize_model_name(dir_name))

    return model_ids


def parse_verified_list(content: str, list_name: str) -> List[str]:
    """Parse a single VERIFIED_*_MODELS list from the file content."""
    pattern = rf"{list_name}\s*=\s*\[(.*?)\]"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    items_str = match.group(1)
    return re.findall(r'"([^"]+)"', items_str)


def parse_all_verified_lists(content: str) -> Dict[str, List[str]]:
    """Parse all VERIFIED_*_MODELS lists from the file content."""
    lists = {}
    # Find all list names
    for match in re.finditer(r"(VERIFIED_\w+_MODELS)\s*=\s*\[", content):
        list_name = match.group(1)
        items = parse_verified_list(content, list_name)
        lists[list_name] = items
    return lists


def provider_list_name(provider: str) -> str:
    """Get the VERIFIED_*_MODELS variable name for a provider."""
    return f"VERIFIED_{provider.upper()}_MODELS"


def find_missing_models(
    completed_ids: Set[str],
    verified_lists: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Find models that are completed but missing from verified lists.

    Returns:
        Tuple of:
        - List of model IDs missing from VERIFIED_OPENHANDS_MODELS
        - Dict mapping provider list names to lists of missing model IDs
    """
    openhands_list = set(verified_lists.get("VERIFIED_OPENHANDS_MODELS", []))
    missing_openhands = []
    missing_providers: Dict[str, List[str]] = {}

    for model_id in sorted(completed_ids):
        # Check VERIFIED_OPENHANDS_MODELS
        if model_id not in openhands_list:
            missing_openhands.append(model_id)

        # Check provider-specific list
        provider = detect_provider(model_id)
        if provider:
            plist_name = provider_list_name(provider)
            provider_models = set(verified_lists.get(plist_name, []))
            if model_id not in provider_models:
                if plist_name not in missing_providers:
                    missing_providers[plist_name] = []
                missing_providers[plist_name].append(model_id)

    return missing_openhands, missing_providers


def add_models_to_list(
    content: str, list_name: str, new_models: List[str]
) -> str:
    """Add new model entries to a VERIFIED_*_MODELS list in the file content."""
    pattern = rf"({list_name}\s*=\s*\[)(.*?)(\])"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return content

    items_str = match.group(2)

    # Build new entries string
    new_entries = "".join(f'    "{m}",\n' for m in new_models)

    # Insert before the closing bracket
    # Find the last item and add after it
    if items_str.rstrip().endswith(","):
        updated_items = items_str + new_entries
    else:
        # If the list has content but no trailing comma, add one
        stripped = items_str.rstrip()
        if stripped:
            updated_items = items_str.rstrip() + ",\n" + new_entries
        else:
            updated_items = "\n" + new_entries

    return content[: match.start(2)] + updated_items + content[match.end(2) :]


def generate_updated_content(
    original_content: str,
    missing_openhands: List[str],
    missing_providers: Dict[str, List[str]],
) -> str:
    """Generate updated verified_models.py content with missing models added."""
    content = original_content

    # Add to provider-specific lists first
    for list_name, models in sorted(missing_providers.items()):
        content = add_models_to_list(content, list_name, models)

    # Add to VERIFIED_OPENHANDS_MODELS
    if missing_openhands:
        content = add_models_to_list(
            content, "VERIFIED_OPENHANDS_MODELS", missing_openhands
        )

    return content


def fetch_verified_models_content(url: str = VERIFIED_MODELS_URL) -> str:
    """Fetch the current verified_models.py content from GitHub."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    complete_models_path = repo_root / "complete-models.json"

    if not complete_models_path.exists():
        print(f"Error: {complete_models_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Extracting completed model IDs...")
    completed_ids = extract_completed_model_ids(complete_models_path, repo_root)
    print(f"Found {len(completed_ids)} unique completed models from results/")

    print("Fetching current verified_models.py...")
    content = fetch_verified_models_content()

    print("Parsing verified lists...")
    verified_lists = parse_all_verified_lists(content)

    print("Finding missing models...")
    missing_openhands, missing_providers = find_missing_models(
        completed_ids, verified_lists
    )

    total_missing = len(missing_openhands) + sum(
        len(v) for v in missing_providers.values()
    )

    if total_missing == 0:
        print("All completed models are already in verified_models.py!")
        return

    if missing_openhands:
        print(f"\nMissing from VERIFIED_OPENHANDS_MODELS: {missing_openhands}")
    for list_name, models in sorted(missing_providers.items()):
        print(f"Missing from {list_name}: {models}")

    updated = generate_updated_content(content, missing_openhands, missing_providers)

    output_path = repo_root / "verified_models_updated.py"
    with open(output_path, "w") as f:
        f.write(updated)

    print(f"\nUpdated file written to: {output_path}")
    print(f"Total additions: {total_missing}")


if __name__ == "__main__":
    main()
