"""Tests for extracting per-instance benchmark outcomes."""

import io
import json
import tarfile

from extract_instance_results import Pricing, _normalize_costs_to_target_mean, extract_records


def _write_tar(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, payload in members.items():
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


def _conversation_archive(base_state):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tf:
        payload = json.dumps(base_state).encode()
        info = tarfile.TarInfo("state/base_state.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return raw.getvalue()


def test_extract_records_from_top_level_report(tmp_path):
    archive = tmp_path / "results.tar.gz"
    _write_tar(
        archive,
        {
            "run/output.report.json": {
                "resolved_ids": ["a", "b"],
                "unresolved_ids": ["c"],
                "invalidated_ids": ["d"],
                "completed_ids": ["a", "b", "c", "d", "e"],
            }
        },
    )

    records = extract_records(
        archive,
        benchmark="commit0",
        source_archive="https://results.eval.all-hands.dev/commit0/model/run/results.tar.gz",
    )

    by_id = {record["instance_id"]: record for record in records}
    assert by_id["a"]["status"] == "resolved"
    assert by_id["a"]["resolved"] is True
    assert by_id["c"]["status"] == "unresolved"
    assert by_id["c"]["resolved"] is False
    assert by_id["d"]["status"] == "invalidated"
    assert by_id["d"]["resolved"] is False
    assert by_id["e"]["status"] == "completed"
    assert by_id["e"]["resolved"] is None


def test_extract_records_from_instance_reports_when_top_report_missing(tmp_path):
    archive = tmp_path / "results.tar.gz"
    _write_tar(
        archive,
        {
            "run/logs/run_evaluation/1/OpenHands/foo/report.json": {
                "foo": {"resolved": True, "patch_exists": True}
            },
            "run/logs/run_evaluation/1/OpenHands/bar/report.json": {
                "bar": {"resolved": False, "patch_exists": True}
            },
            "run/logs/run_evaluation/1/OpenHands/baz/report.json": {
                "baz": {"resolved": False, "patch_is_None": True}
            },
        },
    )

    records = extract_records(
        archive,
        benchmark="swe-bench-multimodal",
        source_archive="https://results.eval.all-hands.dev/swebenchmultimodal/model/run/results.tar.gz",
    )

    by_id = {record["instance_id"]: record for record in records}
    assert by_id["foo"]["status"] == "resolved"
    assert by_id["bar"]["status"] == "unresolved"
    assert by_id["baz"]["status"] == "empty_patch"


def test_extract_records_from_output_errors_jsonl(tmp_path):
    archive = tmp_path / "results.tar.gz"
    _write_tar(
        archive,
        {
            "run/output.report.json": {
                "resolved_ids": [],
                "unresolved_ids": [],
            },
            "run/output_errors.jsonl": (
                b'{"instance_id":"cachetools","test_result":{},"error":"failed"}\n'
            ),
        },
    )

    records = extract_records(
        archive,
        benchmark="commit0",
        source_archive="https://results.eval.all-hands.dev/commit0/model/run/results.tar.gz",
    )

    assert records == [{
        "benchmark": "commit0",
        "instance_id": "cachetools",
        "status": "error",
        "resolved": False,
        "cost": None,
        "source_archive": "https://results.eval.all-hands.dev/commit0/model/run/results.tar.gz",
        "source_path": "run/output_errors.jsonl",
    }]


def test_extract_records_from_output_jsonl_unknown_status(tmp_path):
    archive = tmp_path / "results.tar.gz"
    _write_tar(
        archive,
        {
            "run/output.jsonl": (
                b'{"instance_id":"sympy__sympy-1","test_result":{"git_patch":"diff"}}\n'
            ),
        },
    )

    records = extract_records(
        archive,
        benchmark="swt-bench",
        source_archive="https://results.eval.all-hands.dev/swtbench/model/run/results.tar.gz",
    )

    assert records[0]["instance_id"] == "sympy__sympy-1"
    assert records[0]["status"] == "unknown"
    assert records[0]["resolved"] is None
    assert records[0]["cost"] is None


def test_extract_records_from_output_jsonl_cost(tmp_path):
    archive = tmp_path / "results.tar.gz"
    _write_tar(
        archive,
        {
            "run/output.jsonl": (
                b'{"instance_id":"sympy__sympy-1","metrics":{"accumulated_cost":0.42},'
                b'"test_result":{"resolved":true}}\n'
            ),
        },
    )

    records = extract_records(
        archive,
        benchmark="swt-bench",
        source_archive="https://results.eval.all-hands.dev/swtbench/model/run/results.tar.gz",
    )

    assert records[0]["instance_id"] == "sympy__sympy-1"
    assert records[0]["status"] == "resolved"
    assert records[0]["resolved"] is True
    assert records[0]["cost"] == 0.42


def test_extract_records_recalculates_zero_output_jsonl_cost_from_usage(tmp_path):
    archive = tmp_path / "results.tar.gz"
    _write_tar(
        archive,
        {
            "run/output.jsonl": (
                b'{"instance_id":"sympy__sympy-1","metrics":{'
                b'"accumulated_cost":0,'
                b'"accumulated_token_usage":{'
                b'"prompt_tokens":1000,"completion_tokens":200,"cache_read_tokens":400'
                b'}},'
                b'"test_result":{"resolved":true}}\n'
            ),
        },
    )

    records = extract_records(
        archive,
        benchmark="swe-bench",
        source_archive="https://results.eval.all-hands.dev/swebench/model/run/results.tar.gz",
        pricing=Pricing(input_cache_miss_cost=1.0, input_cache_hit_cost=0.1, output_cost=2.0),
    )

    assert records[0]["instance_id"] == "sympy__sympy-1"
    assert records[0]["cost"] == 0.00104


def test_extract_records_supplements_missing_jsonl_usage_from_conversation(tmp_path):
    archive = tmp_path / "results.tar.gz"
    _write_tar(
        archive,
        {
            "run/output.jsonl": (
                b'{"instance_id":"sympy__sympy-1","metrics":{},'
                b'"test_result":{"resolved":true}}\n'
            ),
            "run/conversations/sympy__sympy-1.tar.gz": _conversation_archive(
                {
                    "stats": {
                        "usage_to_metrics": {
                            "llm": {
                                "accumulated_token_usage": {
                                    "prompt_tokens": 1000,
                                    "completion_tokens": 200,
                                    "cache_read_tokens": 400,
                                }
                            }
                        }
                    }
                }
            ),
        },
    )

    records = extract_records(
        archive,
        benchmark="swe-bench",
        source_archive="https://results.eval.all-hands.dev/swebench/model/run/results.tar.gz",
        pricing=Pricing(input_cache_miss_cost=1.0, input_cache_hit_cost=0.1, output_cost=2.0),
    )

    assert records[0]["instance_id"] == "sympy__sympy-1"
    assert records[0]["cost"] == 0.00104


def test_normalize_costs_to_published_graded_mean():
    records = [
        {"instance_id": "a", "resolved": True, "cost": 1.0},
        {"instance_id": "b", "resolved": False, "cost": 3.0},
        {"instance_id": "c", "resolved": None, "cost": 100.0},
        {"instance_id": "d", "resolved": True, "cost": None},
    ]

    _normalize_costs_to_target_mean(records, target_mean=10.0)

    assert records[0]["cost"] == 5.0
    assert records[1]["cost"] == 15.0
    assert records[2]["cost"] == 500.0
    assert records[3]["cost"] is None


def test_normalize_assigns_published_cost_when_distribution_missing():
    records = [
        {"instance_id": "a", "resolved": True, "cost": None},
        {"instance_id": "b", "resolved": False, "cost": None},
        {"instance_id": "c", "resolved": None, "cost": None},
    ]

    _normalize_costs_to_target_mean(records, target_mean=0.01)

    assert records[0]["cost"] == 0.01
    assert records[1]["cost"] == 0.01
    assert records[2]["cost"] is None
