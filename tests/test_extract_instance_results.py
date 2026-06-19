"""Tests for extracting per-instance benchmark outcomes."""

import io
import json
import tarfile

from extract_instance_results import extract_records


def _write_tar(path, members):
    with tarfile.open(path, "w:gz") as tf:
        for name, payload in members.items():
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))


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
