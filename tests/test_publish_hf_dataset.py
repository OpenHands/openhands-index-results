"""Tests for the publish_hf_dataset script.

These tests focus on the data-collection surface of the publisher and verify
that alternative agents are intentionally excluded from the dataset that
gets pushed to Hugging Face (see issue #1145), and that the dataset card
emits a HF-valid ``dataset_info.version`` (see issue #1189).
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import publish_hf_dataset


def _write_model(model_dir: Path) -> None:
    """Write minimal valid metadata.json/scores.json for one model."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(json.dumps({
        "agent_name": "OpenHands",
        "model": model_dir.name,
    }))
    scores = [{
        "benchmark": "swe-bench",
        "score": 0.5,
        "cost_per_instance": 0.1,
        "average_runtime": 60,
    }]
    (model_dir / "scores.json").write_text(json.dumps(scores))


def _write_sidecar(model_dir: Path, benchmark: str, entries: dict) -> None:
    """Write a results/<model>/instance_results/<benchmark>.json sidecar."""
    instance_dir = model_dir / "instance_results"
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / f"{benchmark}.json").write_text(json.dumps(entries))


class TestIterModelDirs:
    """Tests for _iter_model_dirs — the source-of-truth on what gets published."""

    def test_yields_only_results_models(self, tmp_path):
        _write_model(tmp_path / "results" / "GPT-X")
        _write_model(tmp_path / "results" / "Claude-Y")
        # Alternative agents must be ignored even when valid.
        _write_model(tmp_path / "alternative_agents" / "acp-codex" / "GPT-X")
        _write_model(tmp_path / "alternative_agents" / "acp-claude" / "Claude-Y")

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            pairs = list(publish_hf_dataset._iter_model_dirs())

        assert {(d.name, label) for d, label in pairs} == {
            ("Claude-Y", "OpenHands"),
            ("GPT-X", "OpenHands"),
        }

    def test_no_results_dir_yields_nothing(self, tmp_path):
        _write_model(tmp_path / "alternative_agents" / "acp-codex" / "GPT-X")

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            assert list(publish_hf_dataset._iter_model_dirs()) == []


class TestBuildDataframe:
    """End-to-end check that the published DataFrame excludes alt agents."""

    def test_dataframe_excludes_alternative_agents(self, tmp_path):
        _write_model(tmp_path / "results" / "GPT-X")
        _write_model(tmp_path / "alternative_agents" / "acp-codex" / "GPT-X")

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            df = publish_hf_dataset.build_dataframe()

        assert list(df["agent_type"].unique()) == ["OpenHands"]
        assert list(df["id"]) == ["OpenHands/GPT-X"]


# HF's ``datasets.Version`` accepts exactly three dot-separated digit groups.
# Mirror the same regex used by the upstream library so the test fails for the
# exact same inputs that would crash ``get_dataset_config_names()``.
_HF_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


class TestResolveSourceVersion:
    """``resolve_source_version`` must return both a tag-style ``version``
    (with the git short SHA) and a HF-valid ``info_version`` (digits only)."""

    def _run(self, tmp_path, commit_iso):
        """Run resolve_source_version against a freshly initialised repo whose
        single commit has the given committer date."""
        run = lambda *a: subprocess.run(  # noqa: E731
            a, cwd=tmp_path, check=True, capture_output=True, text=True
        )
        run("git", "init", "-q", "-b", "main")
        run("git", "config", "user.email", "t@t")
        run("git", "config", "user.name", "t")
        (tmp_path / "x").write_text("x")
        run("git", "add", "x")
        env_run = subprocess.run(
            ["git", "commit", "-q", "-m", "x"],
            cwd=tmp_path,
            env={
                "GIT_COMMITTER_DATE": commit_iso,
                "GIT_AUTHOR_DATE": commit_iso,
                "PATH": os.environ["PATH"],
                "HOME": str(tmp_path),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        assert env_run.returncode == 0
        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_SHA", None)
            return publish_hf_dataset.resolve_source_version()

    def test_info_version_is_digits_only(self, tmp_path):
        sha, short, version, info_version, dt = self._run(
            tmp_path, "2026-06-08T12:00:00+0000"
        )

        # The full ``version`` is still the human-facing string with the SHA.
        assert version == f"2026.06.08-{short}"
        assert sha.startswith(short)
        assert dt == datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)

        # ``info_version`` is what gets written to ``dataset_info.version`` —
        # it must be a plain ``x.y.z`` string of digits or HF's datasets
        # library will refuse to load the dataset card (issue #1189).
        assert info_version == "2026.6.8"
        assert _HF_VERSION_RE.match(info_version), info_version

    def test_info_version_drops_leading_zeros(self, tmp_path):
        # Single-digit months/days must not be zero-padded — ``06`` would be
        # a digit string too but using ``2026.6.8`` matches the upstream
        # ``x.y.z`` convention used in the issue's suggested fix.
        _, _, _, info_version, _ = self._run(
            tmp_path, "2026-01-02T00:00:00+0000"
        )
        assert info_version == "2026.1.2"
        assert _HF_VERSION_RE.match(info_version)


class TestDatasetCardVersion:
    """The rendered dataset card must embed the HF-valid ``info_version`` in
    the YAML frontmatter and keep the SHA-bearing ``version`` for display."""

    def _card(self, version="2026.06.08-525c6b4", info_version="2026.6.8"):
        return publish_hf_dataset.dataset_card(
            df=pd.DataFrame(
                {
                    "language_model": ["m"],
                    "sdk_version": ["1"],
                    "agent_name": ["OpenHands"],
                    "average_score": [0.5],
                    "categories_completed": [1],
                    "release_date": ["2026-06-08"],
                }
            ),
            generated_at="2026-06-08 12:00:00 UTC",
            source_sha="525c6b4abcdef",
            version=version,
            info_version=info_version,
        )

    def test_yaml_dataset_info_version_is_hf_valid(self):
        card = self._card()

        # Extract the ``version:`` line under ``dataset_info:`` and assert it
        # parses as the upstream regex (``x.y.z`` digits) — this is the exact
        # surface that failed in #1189 with ``2026.06.08-525c6b4``.
        m = re.search(
            r"^dataset_info:\s*\n(?:.*\n)*?\s*version:\s*(\S+)\s*$",
            card,
            flags=re.MULTILINE,
        )
        assert m is not None, "dataset_info.version not found in YAML frontmatter"
        emitted = m.group(1)
        assert _HF_VERSION_RE.match(emitted), (
            f"dataset_info.version {emitted!r} is not a HF-valid x.y.z string"
        )
        assert emitted == "2026.6.8"
        # The git-hash form must never end up in the ``version:`` field even
        # though the SHA legitimately appears elsewhere (e.g. in the
        # ``description:`` field that references the source commit).
        assert "-" not in emitted

    def test_human_version_still_present_in_body(self):
        # The full ``YYYY.MM.DD-<sha>`` tag-style version is still useful for
        # humans and for the ``load_dataset(..., revision=...)`` example, so
        # it must remain in the rendered body even though the YAML uses the
        # digits-only form.
        card = self._card()
        body = card.split("---", 2)[2]
        assert "2026.06.08-525c6b4" in body

    def test_rejects_card_with_invalid_version_format(self):
        # Sanity check: if a future change accidentally feeds the SHA-bearing
        # version back into ``info_version``, this regex assertion fires —
        # which is exactly what would break HF dataset loading.
        bad_card = self._card(info_version="2026.06.08-525c6b4")
        frontmatter = bad_card.split("---", 2)[1]
        m = re.search(r"version:\s*(\S+)", frontmatter)
        assert m is not None
        assert not _HF_VERSION_RE.match(m.group(1))


class TestInstancesDataframe:
    """Tests for ``build_instances_dataframe`` — the per-instance config that
    gets published alongside the leaderboard. The policy from #1145 (only
    ``results/`` is published) must apply here too, and the schema must
    survive an end-to-end parquet round-trip with ``None`` values for both
    ``resolved`` and ``cost``.
    """

    def test_excludes_alternative_agents(self, tmp_path):
        # Both trees have valid sidecars; only the ``results/`` one should
        # surface in the published instances table.
        _write_model(tmp_path / "results" / "GPT-X")
        _write_sidecar(
            tmp_path / "results" / "GPT-X",
            "swe-bench",
            {"django__django-1": {"resolved": True, "cost": 0.1}},
        )
        _write_model(tmp_path / "alternative_agents" / "acp-codex" / "GPT-X")
        _write_sidecar(
            tmp_path / "alternative_agents" / "acp-codex" / "GPT-X",
            "swe-bench",
            {"django__django-2": {"resolved": True, "cost": 0.2}},
        )

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            df = publish_hf_dataset.build_instances_dataframe()

        assert list(df["id"].unique()) == ["OpenHands/GPT-X"]
        assert list(df["instance_id"]) == ["django__django-1"]

    def test_flattens_multiple_benchmarks_per_model(self, tmp_path):
        # Each sidecar contributes one row per instance; the discriminator
        # column distinguishes them.
        m = tmp_path / "results" / "GPT-X"
        _write_model(m)
        _write_sidecar(m, "swe-bench", {
            "a": {"resolved": True, "cost": 0.1},
            "b": {"resolved": False, "cost": 0.2},
        })
        _write_sidecar(m, "gaia", {
            "g1": {"resolved": None, "cost": None},
        })
        # A non-benchmark stem in the sidecar dir must be skipped, not
        # crash — the script intentionally doesn't re-validate schemas.
        _write_sidecar(m, "not-a-real-benchmark", {"x": {"resolved": True, "cost": 0.0}})

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            df = publish_hf_dataset.build_instances_dataframe()

        assert len(df) == 3
        assert set(df["benchmark"]) == {"swe-bench", "gaia"}
        # ``category`` is derived from the same mapping used by the
        # leaderboard so the two configs stay in sync.
        assert set(df["category"]) == {"Issue Resolution", "Information Gathering"}
        # All required columns are present and ordered deterministically.
        assert list(df.columns) == publish_hf_dataset.INSTANCES_COLUMNS

    def test_preserves_null_resolved_and_cost(self, tmp_path):
        # The sidecar schema explicitly allows ``null`` for both fields when
        # the archive didn't record an outcome — the published parquet must
        # preserve that nullability rather than coercing to ``False`` / ``NaN``.
        m = tmp_path / "results" / "GPT-X"
        _write_model(m)
        _write_sidecar(m, "swe-bench", {
            "known": {"resolved": True, "cost": 0.5},
            "unknown_outcome": {"resolved": None, "cost": 0.3},
            "unknown_cost": {"resolved": False, "cost": None},
        })

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            df = publish_hf_dataset.build_instances_dataframe()

        df = df.set_index("instance_id")
        # ``boolean`` (nullable) dtype keeps ``None`` distinct from ``False`` —
        # the nullable scalars compare ``==`` to native bools but are
        # ``numpy.bool_`` instances, not ``True``/``False`` singletons.
        assert df.loc["known", "resolved"] == True  # noqa: E712
        assert df.loc["unknown_outcome", "resolved"] is pd.NA
        assert df.loc["unknown_cost", "resolved"] == False  # noqa: E712
        assert pd.isna(df.loc["unknown_cost", "cost"])
        assert df.loc["known", "cost"] == 0.5

    def test_empty_when_no_sidecars_present(self, tmp_path):
        # The leaderboard config refuses to publish empty (RuntimeError);
        # the instances config returns an empty frame with the documented
        # schema so the workflow can still publish a leaderboard-only
        # snapshot when sidecars haven't been generated yet.
        _write_model(tmp_path / "results" / "GPT-X")

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            df = publish_hf_dataset.build_instances_dataframe()

        assert df.empty
        assert list(df.columns) == publish_hf_dataset.INSTANCES_COLUMNS

    def test_parquet_roundtrip_preserves_nullability(self, tmp_path):
        # Catches regressions where ``to_parquet`` would silently downcast
        # nullable ``boolean`` to ``object`` and lose the distinction
        # between ``False`` and ``None``.
        m = tmp_path / "results" / "GPT-X"
        _write_model(m)
        _write_sidecar(m, "swe-bench", {
            "i1": {"resolved": None, "cost": None},
            "i2": {"resolved": True, "cost": 1.5},
        })

        with patch.object(publish_hf_dataset, "REPO_ROOT", tmp_path):
            df = publish_hf_dataset.build_instances_dataframe()

        out = tmp_path / "instances.parquet"
        df.to_parquet(out, index=False)
        rt = pd.read_parquet(out)
        rt = rt.set_index("instance_id")
        assert rt.loc["i1", "resolved"] is pd.NA
        assert pd.isna(rt.loc["i1", "cost"])
        assert rt.loc["i2", "resolved"] == True  # noqa: E712
        assert rt.loc["i2", "cost"] == 1.5


class TestContentHash:
    """The publish skip-check hashes *both* DataFrames so sidecar-only
    changes (leaderboard unchanged) still trigger a publish."""

    def test_hash_changes_when_only_instances_change(self):
        leaderboard = pd.DataFrame({"id": ["a"], "score": [0.5]})
        instances_a = pd.DataFrame({
            "id": ["a"], "benchmark": ["swe-bench"], "instance_id": ["x"],
            "resolved": [True], "cost": [0.1],
        })
        instances_b = pd.DataFrame({
            "id": ["a"], "benchmark": ["swe-bench"], "instance_id": ["x"],
            "resolved": [False], "cost": [0.1],
        })
        h_a = publish_hf_dataset.content_hash(leaderboard, instances_a)
        h_b = publish_hf_dataset.content_hash(leaderboard, instances_b)
        assert h_a != h_b, (
            "content_hash must distinguish sidecar-only changes; otherwise "
            "PR #1219-style updates would silently skip publish."
        )

    def test_hash_stable_when_inputs_unchanged(self):
        leaderboard = pd.DataFrame({"id": ["a"], "score": [0.5]})
        instances = pd.DataFrame({
            "id": ["a"], "benchmark": ["swe-bench"], "instance_id": ["x"],
            "resolved": [True], "cost": [0.1],
        })
        assert (
            publish_hf_dataset.content_hash(leaderboard, instances)
            == publish_hf_dataset.content_hash(leaderboard, instances)
        )


class TestDatasetCardConfigs:
    """The card must declare both configs in the YAML frontmatter so HF
    exposes them via ``load_dataset(..., name=...)``."""

    def _card(self, instances_df=None):
        return publish_hf_dataset.dataset_card(
            df=pd.DataFrame({
                "language_model": ["m"], "sdk_version": ["1"],
                "agent_name": ["OpenHands"], "average_score": [0.5],
                "categories_completed": [1], "release_date": ["2026-06-08"],
            }),
            generated_at="2026-06-18 12:00:00 UTC",
            source_sha="abc1234deadbeef",
            version="2026.06.18-abc1234",
            info_version="2026.6.18",
            instances_df=instances_df,
        )

    def test_yaml_declares_both_configs(self):
        # Parsed as YAML so a future formatting tweak (e.g. inline lists)
        # can't silently break the contract that consumers rely on.
        yaml = pytest.importorskip("yaml")
        card = self._card(instances_df=pd.DataFrame({"x": [1, 2, 3]}))
        frontmatter = card.split("---", 2)[1]
        meta = yaml.safe_load(frontmatter)

        configs = {c["config_name"]: c for c in meta["configs"]}
        assert set(configs) == {"default", "instances"}
        assert configs["default"]["data_files"] == [
            {"split": "test", "path": "test.parquet"}
        ]
        assert configs["instances"]["data_files"] == [
            {"split": "test", "path": "instances.parquet"}
        ]

        # ``dataset_info`` must carry one HF-valid version per config —
        # both share the same source commit so the versions match.
        infos = {d["config_name"]: d for d in meta["dataset_info"]}
        assert set(infos) == {"default", "instances"}
        assert all(_HF_VERSION_RE.match(d["version"]) for d in infos.values())

    def test_body_reports_instances_row_count(self):
        card = self._card(instances_df=pd.DataFrame({"x": list(range(42))}))
        body = card.split("---", 2)[2]
        # The card surfaces both row counts so a quick eyeball of the
        # dataset page confirms a publish actually included the sidecars.
        assert "42" in body
        assert "instances" in body.lower()

    def test_works_without_instances_df(self):
        # Backwards compatibility: the function still renders if a caller
        # (e.g. an ad-hoc local invocation) omits ``instances_df``.
        card = self._card(instances_df=None)
        body = card.split("---", 2)[2]
        assert "0" in body  # instances_rows defaults to 0
