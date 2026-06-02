#!/usr/bin/env python3
"""
Update verified_models.py in the software-agent-sdk repository.

Reads ``complete-models.json`` from this repo and the locally checked-out
``verified_models.py`` from ``software-agent-sdk`` to compute which completed
benchmark models are missing from the verified-model lists, then writes an
updated file in place.

Key safety rules:

* The canonical LiteLLM model ID is *never* inferred from a directory name
  alone. We read ``litellm_model_id`` from each model's ``metadata.json``.
* As a backstop, a lower-cased directory name is accepted only when that
  exact string is already present in one of the SDK's ``VERIFIED_*_MODELS``
  lists -- proof that the lower-cased form is a real LiteLLM ID. Any other
  unmapped directory is skipped with a loud warning.
* The script exits non-zero if any completed model is skipped due to a
  missing ID mapping, so the workflow fails loudly rather than silently
  publishing a stale list.

Only ``results/<model>/`` directories are considered. The
``alternative_agents/`` tree captures *agent* variants (acp-claude,
acp-codex, openhands_subagents, ...). ``verified_models.py`` is an
agent-agnostic list of LiteLLM IDs, so alternative-agent runs do not need
their own entries -- the underlying model is already covered by its
``results/`` counterpart when verified.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

# Provider detection: ordered list of (prefix_regex, provider_key).
# Order matters: more specific patterns first.
PROVIDER_RULES: list[tuple[str, str]] = [
    (r"^gpt-", "openai"),
    (r"^o[0-9]", "openai"),
    (r"^codex-", "openai"),
    (r"^claude-", "anthropic"),
    (r"^devstral-", "mistral"),
    (r"^gemini-", "gemini"),
    (r"^deepseek-", "deepseek"),
    (r"^kimi-", "moonshot"),
    (r"^minimax-", "minimax"),
    (r"^glm-", "glm"),
    (r"^nemotron-", "nvidia"),
    (r"^qwen", "qwen"),
]

DEFAULT_SDK_FILE = Path(
    "software-agent-sdk/openhands-sdk/openhands/sdk/llm/utils/verified_models.py"
)


def detect_provider(model_id: str) -> str | None:
    """Return the provider key for a canonical model ID, or None if unknown."""
    for pattern, provider in PROVIDER_RULES:
        if re.match(pattern, model_id):
            return provider
    return None


def provider_list_name(provider: str) -> str:
    """Return the ``VERIFIED_*_MODELS`` variable name for a provider key."""
    return f"VERIFIED_{provider.upper()}_MODELS"


def parse_verified_lists(content: str) -> dict[str, list[str]]:
    """
    Parse every ``VERIFIED_*_MODELS = [...]`` assignment using ``ast``.

    Only assignments whose RHS is a list literal of strings are returned; the
    aggregate ``VERIFIED_MODELS = {...}`` dict (which references the lists by
    name) is correctly skipped.
    """
    tree = ast.parse(content)
    lists: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.List) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not (target.id.startswith("VERIFIED_") and target.id.endswith("_MODELS")):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            lists[target.id] = value
    return lists


def resolve_model_id(
    model_path: str,
    repo_root: Path,
    known_ids: set[str],
) -> tuple[str | None, str | None]:
    """
    Resolve the canonical LiteLLM model ID for a ``results/<dir>`` entry.

    Returns ``(model_id, None)`` on success, or ``(None, reason)`` if the
    model has to be skipped. ``reason`` is a human-readable explanation
    suitable for printing.

    Resolution order:

    1. ``litellm_model_id`` field in the directory's ``metadata.json``.
    2. Lower-cased directory name, but only if it is *already* a known ID
       in some ``VERIFIED_*_MODELS`` list (i.e. the lowercase form is
       provably correct). This handles the common case automatically.
    3. Otherwise: refuse to guess; require an explicit ``litellm_model_id``.
    """
    dir_name = Path(model_path).name
    metadata_path = repo_root / model_path / "metadata.json"

    explicit_id: str | None = None
    if metadata_path.is_file():
        try:
            with metadata_path.open() as f:
                explicit_id = json.load(f).get("litellm_model_id")
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"failed to read {metadata_path}: {exc}"

    if explicit_id:
        return explicit_id, None

    fallback = dir_name.lower()
    if fallback in known_ids:
        return fallback, None

    return None, (
        f"no canonical ID for '{dir_name}' (lowercased '{fallback}' is not in "
        f"any VERIFIED_*_MODELS list). Add a 'litellm_model_id' field to "
        f"{metadata_path.relative_to(repo_root)}."
    )


def extract_completed_model_paths(complete_models_path: Path) -> list[str]:
    """
    Return ``model-path`` values for completed models we care about.

    Only entries under ``results/`` are returned: ``verified_models.py`` is
    agent-agnostic, and ``alternative_agents/`` paths describe agent runs
    rather than new models.
    """
    with complete_models_path.open() as f:
        entries = json.load(f)
    return [e["model-path"] for e in entries if e["model-path"].startswith("results/")]


def find_missing_models(
    model_ids: set[str],
    verified_lists: dict[str, list[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Compute which canonical model IDs are missing from the SDK lists.

    Returns ``(missing_openhands, missing_providers)``:
      * ``missing_openhands`` -- IDs missing from VERIFIED_OPENHANDS_MODELS
      * ``missing_providers`` -- mapping of ``VERIFIED_<X>_MODELS`` -> sorted IDs
    """
    openhands_set = set(verified_lists.get("VERIFIED_OPENHANDS_MODELS", []))
    missing_openhands: list[str] = []
    missing_providers: dict[str, list[str]] = {}

    for model_id in sorted(model_ids):
        if model_id not in openhands_set:
            missing_openhands.append(model_id)

        provider = detect_provider(model_id)
        if provider is None:
            continue
        list_name = provider_list_name(provider)
        if model_id not in set(verified_lists.get(list_name, [])):
            missing_providers.setdefault(list_name, []).append(model_id)

    return missing_openhands, missing_providers


# Capture the assignment, body, and the line containing the closing bracket so
# we can splice new entries in without disturbing surrounding formatting.
_LIST_BLOCK_RE = re.compile(
    r"(?P<head>^(?P<name>VERIFIED_\w+_MODELS)\s*=\s*\[\n)"
    r"(?P<body>.*?)"
    r"(?P<tail>^\])",
    re.DOTALL | re.MULTILINE,
)


def _detect_indent(body: str) -> str:
    """Return the leading whitespace used by entries in a list body."""
    match = re.search(r"^( +)\"", body, re.MULTILINE)
    return match.group(1) if match else "    "


def insert_into_list(content: str, list_name: str, new_models: list[str]) -> str:
    """
    Append entries to a ``VERIFIED_*_MODELS`` list, preserving formatting.

    If the list does not exist, the content is returned unchanged.
    """
    if not new_models:
        return content

    def _replace(match: re.Match[str]) -> str:
        if match.group("name") != list_name:
            return match.group(0)
        body = match.group("body")
        indent = _detect_indent(body)
        # Make sure the existing last entry ends with a comma+newline so our
        # additions don't fuse onto its line.
        if body and not body.rstrip("\n ").endswith(","):
            body = body.rstrip() + ",\n"
        new_lines = "".join(f"{indent}\"{m}\",\n" for m in new_models)
        return match.group("head") + body + new_lines + match.group("tail")

    return _LIST_BLOCK_RE.sub(_replace, content, count=0)


def generate_updated_content(
    original_content: str,
    missing_openhands: list[str],
    missing_providers: dict[str, list[str]],
) -> str:
    """Return updated file content with missing entries appended to each list."""
    content = original_content
    for list_name, models in sorted(missing_providers.items()):
        content = insert_into_list(content, list_name, models)
    if missing_openhands:
        content = insert_into_list(
            content, "VERIFIED_OPENHANDS_MODELS", missing_openhands
        )
    return content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root of the openhands-index-results checkout (default: parent of scripts/).",
    )
    parser.add_argument(
        "--complete-models",
        type=Path,
        default=None,
        help="Path to complete-models.json (default: <repo-root>/complete-models.json).",
    )
    parser.add_argument(
        "--sdk-verified-models",
        type=Path,
        default=None,
        help=(
            "Path to the local checkout of verified_models.py "
            f"(default: <repo-root>/{DEFAULT_SDK_FILE})."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the SDK file in place. Without this flag the script only reports.",
    )
    args = parser.parse_args(argv)

    repo_root: Path = args.repo_root
    complete_models_path: Path = args.complete_models or repo_root / "complete-models.json"
    sdk_file: Path = args.sdk_verified_models or repo_root / DEFAULT_SDK_FILE

    if not complete_models_path.is_file():
        print(f"error: {complete_models_path} not found", file=sys.stderr)
        return 2
    if not sdk_file.is_file():
        print(f"error: {sdk_file} not found", file=sys.stderr)
        return 2

    original_content = sdk_file.read_text()
    verified_lists = parse_verified_lists(original_content)
    known_ids: set[str] = {mid for ids in verified_lists.values() for mid in ids}

    print(f"Parsed {len(verified_lists)} VERIFIED_*_MODELS lists from {sdk_file}")

    completed_paths = extract_completed_model_paths(complete_models_path)
    print(f"Found {len(completed_paths)} completed results/ entries in {complete_models_path.name}")

    resolved_ids: set[str] = set()
    skipped: list[str] = []
    for model_path in completed_paths:
        model_id, reason = resolve_model_id(model_path, repo_root, known_ids)
        if model_id is None:
            assert reason is not None
            skipped.append(f"{model_path}: {reason}")
        else:
            resolved_ids.add(model_id)

    if skipped:
        print("\nSkipped models (require an explicit canonical ID):", file=sys.stderr)
        for line in skipped:
            print(f"  - {line}", file=sys.stderr)

    missing_openhands, missing_providers = find_missing_models(
        resolved_ids, verified_lists
    )
    total_missing = len(missing_openhands) + sum(
        len(v) for v in missing_providers.values()
    )

    if missing_openhands:
        print(f"\nMissing from VERIFIED_OPENHANDS_MODELS: {missing_openhands}")
    for list_name, models in sorted(missing_providers.items()):
        print(f"Missing from {list_name}: {models}")

    if total_missing == 0:
        print("\nAll resolved completed models are already in verified_models.py.")
    else:
        updated = generate_updated_content(
            original_content, missing_openhands, missing_providers
        )
        # Sanity check: the rewritten file must still parse as Python.
        ast.parse(updated)
        if args.write:
            sdk_file.write_text(updated)
            print(f"\nWrote {total_missing} new entries to {sdk_file}")
        else:
            print(
                f"\nWould add {total_missing} new entries (dry run; pass --write to apply)."
            )

    # Fail loudly if any model was skipped so the workflow surfaces the gap
    # instead of opening a PR that silently omits models.
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
