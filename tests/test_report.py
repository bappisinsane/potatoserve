"""Unit tests for median+spread reporting and the no-noise rule (potato/report.py).

No GPU required. These tests pin down two behaviours that matter for honesty:

  1. the report must show the REAL spread across repeats. A regression here
     (reading ``spread["decode_tps"]`` when llama-bench stores the per-repeat
     samples under ``spread["decode_samples_ts"]``) would silently report
     n=1/stdev=0 and hide the very spread the spec demands we publish;
  2. a difference inside the noise band must never be declared a winner
     (docs/rules.md), while a genuinely disjoint band must be.

Fixture numbers are illustrative test data, not measured results.
"""

import unittest

from potato.report import (
    compare,
    summarize_samples,
    _metric_summary_from_record,
    render_compare_all,
    render_record_report,
)

# A record shaped like what `potato bench` actually writes: per-repeat samples
# under the *_samples_ts keys, plus a single VRAM scalar (n=1) and unmeasured
# metrics left null.
BASE_RECORD = {
    "schema_version": 1,
    "run_id": "base-001",
    "date": "2026-10-11T00:00:00Z",
    "llama_cpp_commit": "d812350",
    "build_flags": {},
    "driver": "616.56",
    "os": "Windows",
    "gpu": "NVIDIA GeForce GTX 1660 Ti",
    "model_id": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "model_sha256": None,
    "config": {},
    "metrics": {
        "prefill_tps": 4238.19,
        "decode_tps": 147.6,
        "vram_peak_mb": 6000,
        "kl_mean": None,
        "top1_agree": None,
        "ppl": None,
        "agent_success": None,
    },
    "repeats": 3,
    "spread": {
        "decode_stddev_ts": 14.543340522574432,
        "prefill_stddev_ts": 73.52,
        "decode_samples_ts": [147.6, 172.454, 137.925],
        "prefill_samples_ts": [4131.95, 4311.0, 4238.19],
    },
}


class SummarizeSamplesTest(unittest.TestCase):
    def test_median_and_population_stdev(self) -> None:
        s = summarize_samples([147.6, 172.454, 137.925])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["median"], 147.6)
        self.assertAlmostEqual(s["min"], 137.925)
        self.assertAlmostEqual(s["max"], 172.454)
        # Population stdev of those three values.
        self.assertAlmostEqual(s["stdev"], 14.543340522574432)

    def test_single_sample_has_zero_stdev(self) -> None:
        s = summarize_samples([42.0])
        self.assertEqual((s["n"], s["stdev"], s["median"]), (1, 0.0, 42.0))

    def test_empty_is_all_none_not_zero(self) -> None:
        s = summarize_samples([])
        self.assertEqual(s, {"median": None, "stdev": None, "min": None,
                             "max": None, "n": 0})

    def test_ignores_bools_and_non_numbers(self) -> None:
        s = summarize_samples([1.0, True, "x", None, 3.0])
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["median"], 2.0)


class MetricSummaryTest(unittest.TestCase):
    def test_reads_real_per_repeat_samples(self) -> None:
        # The key regression: samples live in *_samples_ts, so we must get n=3
        # and the real stdev - not a synthesized n=1/stdev=0.
        s = _metric_summary_from_record(BASE_RECORD, "decode_tps")
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["median"], 147.6)
        self.assertAlmostEqual(s["stdev"], 14.543340522574432)

    def test_falls_back_to_single_scalar(self) -> None:
        # vram_peak_mb is a plain scalar with no sample list -> n=1, stdev 0.
        s = _metric_summary_from_record(BASE_RECORD, "vram_peak_mb")
        self.assertEqual(s["n"], 1)
        self.assertEqual((s["median"], s["stdev"]), (6000.0, 0.0))

    def test_precomputed_summary_wins_when_present(self) -> None:
        rec = {
            "metrics": {"decode_tps": 100.0},
            "spread": {
                "decode_tps": {"median": 100.0, "stdev": 2.0,
                               "min": 98.0, "max": 102.0, "n": 5},
                "decode_samples_ts": [1.0, 2.0],  # must be ignored in favour of above
            },
        }
        s = _metric_summary_from_record(rec, "decode_tps")
        self.assertEqual((s["n"], s["stdev"]), (5, 2.0))

    def test_unmeasured_metric_is_n_zero(self) -> None:
        s = _metric_summary_from_record(BASE_RECORD, "ppl")
        self.assertEqual((s["n"], s["median"]), (0, None))


class CompareTest(unittest.TestCase):
    def test_overlapping_bands_make_no_claim(self) -> None:
        # Two records whose decode median+/-stdev bands overlap.
        a = {"metrics": {}, "spread": {"decode_samples_ts": [100.0, 104.0, 108.0]}}
        b = {"metrics": {}, "spread": {"decode_samples_ts": [105.0, 109.0, 113.0]}}
        result = compare(a, b, "decode_tps")
        self.assertEqual(result["verdict"], "within noise (no claim)")

    def test_disjoint_bands_declare_the_better_side(self) -> None:
        # Higher decode_tps is better and the candidate is clearly above base.
        base = {"metrics": {}, "spread": {"decode_samples_ts": [10.0, 10.2, 10.4]}}
        cand = {"metrics": {}, "spread": {"decode_samples_ts": [50.0, 50.2, 50.4]}}
        self.assertIn("candidate is better", compare(base, cand, "decode_tps")["verdict"])

    def test_lower_is_better_metric_direction(self) -> None:
        # ppl: smaller is better, and candidate is clearly lower.
        base = {"metrics": {"ppl": 10.0}, "spread": {}}
        cand = {"metrics": {"ppl": 5.0}, "spread": {}}
        self.assertIn("candidate is better", compare(base, cand, "ppl")["verdict"])

    def test_missing_measurement_is_incomparable_not_equal(self) -> None:
        base = {"metrics": {"decode_tps": 50.0}, "spread": {}}
        cand = {"metrics": {"decode_tps": None}, "spread": {}}
        self.assertEqual(
            compare(base, cand, "decode_tps")["verdict"],
            "incomparable (missing measurement)",
        )


class RenderTest(unittest.TestCase):
    def test_record_report_shows_spread_not_just_median(self) -> None:
        text = render_record_report(BASE_RECORD)
        self.assertIn("stdev", text)
        self.assertIn("147.60", text)        # decode median
        self.assertIn("not measured", text)  # unmeasured metrics, honestly

    def test_record_report_warns_when_below_three_repeats(self) -> None:
        rec = {"run_id": "thin-001", "metrics": {"decode_tps": 100.0}, "spread": {}}
        text = render_record_report(rec)
        self.assertIn("only 1 sample(s)", text)

    def test_compare_all_renders_every_metric(self) -> None:
        text = render_compare_all(BASE_RECORD, BASE_RECORD)
        self.assertIn("within noise", text)
        self.assertIn("Decode throughput", text)
        self.assertIn("incomparable", text)  # the null metrics


if __name__ == "__main__":
    unittest.main()
