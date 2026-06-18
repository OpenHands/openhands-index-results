"""Tests for the compare_costs script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import compare_costs


def _report(est, proxy, zero_proxy=0):
    return {
        "summary": {"total_cost": est},
        "proxy_cost_summary": {
            "total_proxy_cost": proxy,
            "zero_proxy_cost_instances": zero_proxy,
        },
    }


# Real Gemini-3.1-Pro swe-bench totals: estimate $2076, proxy $702 (2.96x).
GEMINI_EST, GEMINI_PROXY = 2076.08, 701.56


class TestCostSource:
    def test_proxy_sourced(self):
        assert compare_costs.cost_source("swe-bench", GEMINI_EST, GEMINI_PROXY, 1.40) == "proxy"

    def test_estimate_sourced(self):
        assert compare_costs.cost_source("swe-bench", GEMINI_EST, GEMINI_PROXY, 4.15) == "estimate"

    def test_unknown_benchmark(self):
        assert compare_costs.cost_source("mystery-bench", 100.0, 50.0, 1.0) == "unknown"

    def test_multimodal_full_split(self):
        # swe-bench-multimodal has shipped against the 102-instance split.
        assert compare_costs.cost_source("swe-bench-multimodal", 761.96, 246.47, 2.42) == "proxy"


class TestFetchReport:
    def test_legacy_url_has_no_sibling(self):
        legacy = "eval-123-model_litellm_proxy-prov_25-01-01-00-00.tar.gz"
        assert compare_costs.fetch_report(legacy) is None


class TestReviewEntry:
    ARCHIVE = "https://x/swebench/litellm_proxy-m/1/results.tar.gz"

    def _entry(self, cpi, benchmark="swe-bench"):
        return {"full_archive": self.ARCHIVE, "benchmark": benchmark, "cost_per_instance": cpi}

    def test_warns_on_estimate_sourced_high_divergence(self, monkeypatch):
        monkeypatch.setattr(compare_costs, "fetch_report", lambda _: _report(GEMINI_EST, GEMINI_PROXY))
        status, message = compare_costs.review_entry("L", self._entry(4.15))
        assert status == "warn"
        assert "use-proxy-costs" in message

    def test_ok_when_proxy_sourced_despite_high_divergence(self, monkeypatch):
        monkeypatch.setattr(compare_costs, "fetch_report", lambda _: _report(GEMINI_EST, GEMINI_PROXY))
        status, _ = compare_costs.review_entry("L", self._entry(1.40))
        assert status == "ok"

    def test_ok_when_divergence_low(self, monkeypatch):
        # OH-native: estimate ~= proxy, so estimate-sourced is harmless.
        monkeypatch.setattr(compare_costs, "fetch_report", lambda _: _report(936.21, 973.27))
        status, _ = compare_costs.review_entry("L", self._entry(1.87))
        assert status == "ok"

    def test_skip_when_report_unavailable(self, monkeypatch):
        monkeypatch.setattr(compare_costs, "fetch_report", lambda _: None)
        assert compare_costs.review_entry("L", self._entry(1.40))[0] == "skip"

    def test_skip_when_proxy_incomplete(self, monkeypatch):
        monkeypatch.setattr(compare_costs, "fetch_report", lambda _: _report(GEMINI_EST, GEMINI_PROXY, zero_proxy=5))
        assert compare_costs.review_entry("L", self._entry(1.40))[0] == "skip"

    def test_skip_when_proxy_missing(self, monkeypatch):
        monkeypatch.setattr(compare_costs, "fetch_report", lambda _: _report(GEMINI_EST, 0.0))
        assert compare_costs.review_entry("L", self._entry(1.40))[0] == "skip"

    def test_skip_when_cost_per_instance_missing(self):
        entry = {"full_archive": self.ARCHIVE, "benchmark": "swe-bench"}
        assert compare_costs.review_entry("L", entry)[0] == "skip"
