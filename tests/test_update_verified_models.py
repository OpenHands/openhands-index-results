"""Tests for scripts/update_verified_models.py."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

# Make the script importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import update_verified_models as uvm  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
LIVE_VERIFIED_MODELS = FIXTURES / "verified_models.py"


# ---------------------------------------------------------------------------
# detect_provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("gpt-5.5", "openai"),
        ("gpt-5.1-codex-max", "openai"),
        ("o4-mini", "openai"),
        ("o3", "openai"),
        ("codex-mini-latest", "openai"),
        ("claude-opus-4-5", "anthropic"),
        # Regression: ^devstral- must map to mistral (was missing pre-fix).
        ("devstral-medium-2512", "mistral"),
        ("devstral-small-2505", "mistral"),
        ("gemini-3-flash", "gemini"),
        ("deepseek-v3.2-reasoner", "deepseek"),
        ("kimi-k2-thinking", "moonshot"),
        ("minimax-m2.1", "minimax"),
        ("glm-4.7", "glm"),
        ("nemotron-3-nano", "nvidia"),
        ("qwen3-coder-480b", "qwen"),
        ("qwen3.6-plus", "qwen"),
        ("trinity-large-thinking", None),
        ("unknown-model", None),
    ],
)
def test_detect_provider(model_id: str, expected: str | None) -> None:
    assert uvm.detect_provider(model_id) == expected


def test_provider_table_covers_every_sdk_list() -> None:
    """
    Every ``VERIFIED_<X>_MODELS`` list in the SDK file (except OPENHANDS, which
    is cross-provider) must be reachable from at least one entry in
    PROVIDER_RULES. This is the test that would have caught the missing
    `^devstral-` rule.
    """
    lists = uvm.parse_verified_lists(LIVE_VERIFIED_MODELS.read_text())
    provider_list_names = {
        name for name in lists if name != "VERIFIED_OPENHANDS_MODELS"
    }
    reachable = {
        uvm.provider_list_name(provider) for _, provider in uvm.PROVIDER_RULES
    }
    missing = provider_list_names - reachable
    assert not missing, (
        f"PROVIDER_RULES does not cover these SDK lists: {sorted(missing)}. "
        "Add a regex rule mapping a model-ID prefix to the provider key."
    )


# ---------------------------------------------------------------------------
# parse_verified_lists
# ---------------------------------------------------------------------------


def test_parse_verified_lists_handles_live_file() -> None:
    """
    Round-trip test: every list parsed from the live ``verified_models.py``
    must be a non-empty list of strings, and the cross-provider OPENHANDS
    list must exist.
    """
    lists = uvm.parse_verified_lists(LIVE_VERIFIED_MODELS.read_text())
    assert "VERIFIED_OPENHANDS_MODELS" in lists
    for name, values in lists.items():
        assert values, f"{name} parsed as empty"
        assert all(isinstance(v, str) for v in values)
    # Smoke-check a few known entries.
    assert "gpt-5.5" in lists["VERIFIED_OPENAI_MODELS"]
    assert "devstral-medium-2512" in lists["VERIFIED_MISTRAL_MODELS"]


def test_parse_verified_lists_skips_dict_assignment() -> None:
    """The trailing ``VERIFIED_MODELS = {...}`` dict must not be parsed as a list."""
    lists = uvm.parse_verified_lists(LIVE_VERIFIED_MODELS.read_text())
    assert "VERIFIED_MODELS" not in lists


def test_parse_verified_lists_synthetic() -> None:
    content = (
        'VERIFIED_OPENAI_MODELS = ["gpt-5.2", "gpt-5.4"]\n'
        'VERIFIED_OPENHANDS_MODELS = []\n'
        'OTHER = ["ignore-me"]\n'
    )
    lists = uvm.parse_verified_lists(content)
    assert lists == {
        "VERIFIED_OPENAI_MODELS": ["gpt-5.2", "gpt-5.4"],
        "VERIFIED_OPENHANDS_MODELS": [],
    }


# ---------------------------------------------------------------------------
# resolve_model_id
# ---------------------------------------------------------------------------


def _write_metadata(model_dir: Path, **fields: object) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(json.dumps(fields))


def test_resolve_model_id_prefers_explicit_field(tmp_path: Path) -> None:
    _write_metadata(
        tmp_path / "results" / "Qwen3.5-Flash", litellm_model_id="qwen3-5-flash"
    )
    model_id, reason = uvm.resolve_model_id(
        "results/Qwen3.5-Flash", tmp_path, known_ids=set()
    )
    assert model_id == "qwen3-5-flash"
    assert reason is None


def test_resolve_model_id_lowercase_when_already_known(tmp_path: Path) -> None:
    """Lowercased name is accepted iff already verified somewhere -- proves it's valid."""
    _write_metadata(tmp_path / "results" / "GPT-5.2", model="GPT-5.2")
    model_id, reason = uvm.resolve_model_id(
        "results/GPT-5.2", tmp_path, known_ids={"gpt-5.2"}
    )
    assert model_id == "gpt-5.2"
    assert reason is None


def test_resolve_model_id_skips_unknown_lowercase(tmp_path: Path) -> None:
    """
    The qwen3.5-flash regression: lowercased dir doesn't match any verified ID
    and there's no litellm_model_id, so we MUST refuse rather than guess.
    """
    _write_metadata(tmp_path / "results" / "Qwen3.5-Flash", model="Qwen3.5-Flash")
    model_id, reason = uvm.resolve_model_id(
        "results/Qwen3.5-Flash", tmp_path, known_ids={"qwen3-coder-480b"}
    )
    assert model_id is None
    assert reason is not None
    assert "Qwen3.5-Flash" in reason
    assert "litellm_model_id" in reason


def test_resolve_model_id_skips_when_metadata_absent(tmp_path: Path) -> None:
    (tmp_path / "results" / "NoMeta").mkdir(parents=True)
    model_id, reason = uvm.resolve_model_id(
        "results/NoMeta", tmp_path, known_ids=set()
    )
    # No metadata, no fallback match -> skipped.
    assert model_id is None
    assert reason is not None


# ---------------------------------------------------------------------------
# extract_completed_model_paths
# ---------------------------------------------------------------------------


def test_extract_completed_model_paths_filters_results_only(tmp_path: Path) -> None:
    data = [
        {"timestamp": "t", "model-path": "results/GPT-5.2"},
        {"timestamp": "t", "model-path": "results/Qwen3.6-Plus"},
        {"timestamp": "t", "model-path": "alternative_agents/acp-claude/claude-opus-4-7"},
    ]
    path = tmp_path / "complete-models.json"
    path.write_text(json.dumps(data))
    assert uvm.extract_completed_model_paths(path) == [
        "results/GPT-5.2",
        "results/Qwen3.6-Plus",
    ]


# ---------------------------------------------------------------------------
# find_missing_models
# ---------------------------------------------------------------------------


def test_find_missing_models_handles_provider_and_openhands() -> None:
    verified = {
        "VERIFIED_OPENHANDS_MODELS": ["gpt-5.2"],
        "VERIFIED_OPENAI_MODELS": ["gpt-5.2"],
        "VERIFIED_MISTRAL_MODELS": [],
    }
    missing_oh, missing_prov = uvm.find_missing_models(
        {"gpt-5.2", "gpt-5.5", "devstral-medium-2512"}, verified
    )
    assert missing_oh == ["devstral-medium-2512", "gpt-5.5"]
    assert missing_prov == {
        "VERIFIED_MISTRAL_MODELS": ["devstral-medium-2512"],
        "VERIFIED_OPENAI_MODELS": ["gpt-5.5"],
    }


def test_find_missing_models_unknown_provider_only_hits_openhands() -> None:
    verified = {"VERIFIED_OPENHANDS_MODELS": []}
    missing_oh, missing_prov = uvm.find_missing_models(
        {"trinity-large-thinking"}, verified
    )
    assert missing_oh == ["trinity-large-thinking"]
    assert missing_prov == {}


# ---------------------------------------------------------------------------
# insert_into_list / generate_updated_content
# ---------------------------------------------------------------------------


def test_insert_into_list_preserves_indent() -> None:
    content = LIVE_VERIFIED_MODELS.read_text()
    updated = uvm.insert_into_list(
        content, "VERIFIED_OPENAI_MODELS", ["gpt-test-new"]
    )
    # The new entry must use the same indent as existing entries (4 spaces).
    assert '    "gpt-test-new",\n' in updated
    # Original entries are untouched.
    assert '    "gpt-5.5",\n' in updated


def test_insert_into_list_handles_missing_list_unchanged() -> None:
    content = LIVE_VERIFIED_MODELS.read_text()
    updated = uvm.insert_into_list(content, "VERIFIED_DOESNOTEXIST_MODELS", ["x"])
    assert updated == content


def test_insert_into_list_no_models_is_noop() -> None:
    content = LIVE_VERIFIED_MODELS.read_text()
    assert uvm.insert_into_list(content, "VERIFIED_OPENAI_MODELS", []) == content


def test_generate_updated_content_roundtrip_parses() -> None:
    """The updated file must still be valid Python (the core safety property)."""
    content = LIVE_VERIFIED_MODELS.read_text()
    updated = uvm.generate_updated_content(
        content,
        missing_openhands=["gpt-test-new"],
        missing_providers={"VERIFIED_OPENAI_MODELS": ["gpt-test-new"]},
    )
    # Must parse as Python.
    ast.parse(updated)
    # Both lists got the new entry.
    lists = uvm.parse_verified_lists(updated)
    assert "gpt-test-new" in lists["VERIFIED_OPENAI_MODELS"]
    assert "gpt-test-new" in lists["VERIFIED_OPENHANDS_MODELS"]


def test_generate_updated_content_preserves_existing_entries() -> None:
    """
    Critical: adding new entries must not remove or reorder existing ones.
    Round-trip the live fixture and assert every existing entry is still there.
    """
    content = LIVE_VERIFIED_MODELS.read_text()
    original_lists = uvm.parse_verified_lists(content)

    updated = uvm.generate_updated_content(
        content,
        missing_openhands=["trinity-new-model"],
        missing_providers={"VERIFIED_MISTRAL_MODELS": ["devstral-future-2999"]},
    )
    updated_lists = uvm.parse_verified_lists(updated)

    # Every original entry still present (no accidental deletion).
    for name, values in original_lists.items():
        assert set(values).issubset(set(updated_lists[name])), (
            f"existing entries dropped from {name}"
        )

    assert "trinity-new-model" in updated_lists["VERIFIED_OPENHANDS_MODELS"]
    assert "devstral-future-2999" in updated_lists["VERIFIED_MISTRAL_MODELS"]


def test_generate_updated_content_empty_diff_unchanged() -> None:
    content = LIVE_VERIFIED_MODELS.read_text()
    assert uvm.generate_updated_content(content, [], {}) == content


# ---------------------------------------------------------------------------
# end-to-end via main()
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, model_dirs: dict[str, dict[str, object]]) -> Path:
    """
    Build a fake repo layout with ``results/<name>/metadata.json`` per entry
    and a ``complete-models.json`` pointing to each. Returns the repo root.
    """
    sdk_target = tmp_path / "sdk_verified_models.py"
    sdk_target.write_text(LIVE_VERIFIED_MODELS.read_text())

    entries: list[dict[str, str]] = []
    for dir_name, meta in model_dirs.items():
        _write_metadata(tmp_path / "results" / dir_name, **meta)
        entries.append(
            {"timestamp": "2026-04-01T00:00:00.000+00:00", "model-path": f"results/{dir_name}"}
        )
    (tmp_path / "complete-models.json").write_text(json.dumps(entries))
    return sdk_target


def test_main_writes_resolvable_additions_and_returns_zero(tmp_path: Path) -> None:
    sdk_target = _make_repo(
        tmp_path,
        {"GPT-5.2": {"model": "GPT-5.2"}},  # lowercase 'gpt-5.2' already known
    )
    rc = uvm.main(
        [
            "--repo-root",
            str(tmp_path),
            "--sdk-verified-models",
            str(sdk_target),
            "--write",
        ]
    )
    assert rc == 0
    # gpt-5.2 already in OPENHANDS list, so nothing actually added.
    assert sdk_target.read_text() == LIVE_VERIFIED_MODELS.read_text()


def test_main_returns_nonzero_when_models_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    Regression for the qwen3.5-flash bug: a model without litellm_model_id and
    whose lowercased dir isn't already verified must NOT be added, and the
    workflow must see a non-zero exit code.
    """
    sdk_target = _make_repo(
        tmp_path,
        {"Qwen3.5-Flash": {"model": "Qwen3.5-Flash"}},
    )
    rc = uvm.main(
        [
            "--repo-root",
            str(tmp_path),
            "--sdk-verified-models",
            str(sdk_target),
            "--write",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "Qwen3.5-Flash" in err
    # Nothing was added because no resolvable models.
    assert sdk_target.read_text() == LIVE_VERIFIED_MODELS.read_text()


def test_main_writes_explicit_id_and_picks_mistral_for_devstral(tmp_path: Path) -> None:
    """End-to-end proof that ^devstral- now lands in VERIFIED_MISTRAL_MODELS."""
    sdk_target = _make_repo(
        tmp_path,
        {
            "Devstral-Future": {
                "model": "Devstral-Future",
                "litellm_model_id": "devstral-future-2999",
            }
        },
    )
    rc = uvm.main(
        [
            "--repo-root",
            str(tmp_path),
            "--sdk-verified-models",
            str(sdk_target),
            "--write",
        ]
    )
    assert rc == 0
    lists = uvm.parse_verified_lists(sdk_target.read_text())
    assert "devstral-future-2999" in lists["VERIFIED_MISTRAL_MODELS"]
    assert "devstral-future-2999" in lists["VERIFIED_OPENHANDS_MODELS"]


def test_main_dry_run_does_not_write(tmp_path: Path) -> None:
    sdk_target = _make_repo(
        tmp_path,
        {
            "Devstral-Future": {
                "model": "Devstral-Future",
                "litellm_model_id": "devstral-future-2999",
            }
        },
    )
    original = sdk_target.read_text()
    rc = uvm.main(
        [
            "--repo-root",
            str(tmp_path),
            "--sdk-verified-models",
            str(sdk_target),
            # no --write
        ]
    )
    assert rc == 0
    assert sdk_target.read_text() == original
