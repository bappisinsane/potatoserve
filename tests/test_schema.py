"""Unit tests for the section-8 result-record schema (potato/schema.py).

Standard library only. Verifies that every §8 top-level and metric key is
always present (self-describing), and that validation catches structural gaps.
"""

import unittest

from potato import schema


class ResultRecordTest(unittest.TestCase):
    def test_to_dict_contains_all_section8_keys(self) -> None:
        rec = schema.new_record("run-001")
        data = rec.to_dict()
        for key in schema.TOP_LEVEL_FIELDS:
            self.assertIn(key, data)

    def test_metrics_all_present_and_null_by_default(self) -> None:
        rec = schema.new_record("run-001")
        for key in schema.METRIC_FIELDS:
            self.assertIn(key, rec.metrics)
            self.assertIsNone(rec.metrics[key])

    def test_date_defaults_to_iso_utc(self) -> None:
        rec = schema.new_record("run-001")
        self.assertIsNotNone(rec.date)
        # Basic ISO-8601 shape check: date then 'T' then time with offset/Z.
        self.assertRegex(rec.date or "", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_round_trip_from_dict_preserves_metrics(self) -> None:
        rec = schema.new_record(
            "run-002",
            llama_cpp_commit="abc123",
            metrics={"decode_tps": 12.3, "prefill_tps": 100.0},
        )
        rebuilt = schema.ResultRecord.from_dict(rec.to_dict())
        self.assertEqual(rebuilt.llama_cpp_commit, "abc123")
        self.assertEqual(rebuilt.metrics["decode_tps"], 12.3)
        # Unmeasured metrics remain present as null.
        self.assertIsNone(rebuilt.metrics["kl_mean"])

    def test_validate_flags_missing_keys(self) -> None:
        problems = schema.validate({"run_id": ""})
        self.assertTrue(any("run_id" in p for p in problems))
        self.assertTrue(any("metrics" in p for p in problems))

    def test_validate_accepts_minimal_record(self) -> None:
        self.assertEqual(schema.validate(schema.new_record("run-003")), [])


if __name__ == "__main__":
    unittest.main()
