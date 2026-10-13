"""Unit tests for the append-only results store (potato/store.py).

No GPU required. Uses a per-test temporary directory. The critical property
under test is ADR-5: a previously written run is NEVER silently overwritten.
"""

import json
import os
import tempfile
import unittest

from potato import store


def _rec(run_id: str, **overrides):
    rec = {"schema_version": 1, "run_id": run_id, "metrics": {"decode_tps": 1.0}}
    rec.update(overrides)
    return rec


class SaveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_creates_named_file(self) -> None:
        path = store.save_record(_rec("run-1"), results_dir=self.dir)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.basename(path), "run-1.json")
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["run_id"], "run-1")

    def test_save_refuses_to_overwrite_by_default(self) -> None:
        store.save_record(_rec("run-1", note="first"), results_dir=self.dir)
        with self.assertRaises(FileExistsError):
            store.save_record(_rec("run-1", note="second"), results_dir=self.dir)
        # The original content must be untouched.
        loaded = store.load_record("run-1", self.dir)
        self.assertEqual(loaded["note"], "first")

    def test_explicit_overwrite_replaces(self) -> None:
        store.save_record(_rec("run-1", note="first"), results_dir=self.dir)
        store.save_record(_rec("run-1", note="second"), results_dir=self.dir,
                          overwrite=True)
        self.assertEqual(store.load_record("run-1", self.dir)["note"], "second")

    def test_missing_run_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            store.save_record({"metrics": {}}, results_dir=self.dir)


class LoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_missing_returns_none(self) -> None:
        self.assertIsNone(store.load_record("nope", self.dir))

    def test_list_and_load_all_are_sorted(self) -> None:
        for rid in ("c-run", "a-run", "b-run"):
            store.save_record(_rec(rid), results_dir=self.dir)
        self.assertEqual(store.list_run_ids(self.dir),
                         ["a-run", "b-run", "c-run"])
        self.assertEqual([r["run_id"] for r in store.load_all(self.dir)],
                         ["a-run", "b-run", "c-run"])

    def test_list_ignores_non_json(self) -> None:
        store.save_record(_rec("real"), results_dir=self.dir)
        with open(os.path.join(self.dir, "README.md"), "w", encoding="utf-8") as fh:
            fh.write("notes")
        self.assertEqual(store.list_run_ids(self.dir), ["real"])

    def test_list_missing_dir_is_empty(self) -> None:
        self.assertEqual(store.list_run_ids(os.path.join(self.dir, "nope")), [])


if __name__ == "__main__":
    unittest.main()
