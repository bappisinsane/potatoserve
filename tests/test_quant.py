"""Unit tests for the Phase 2 quant pipeline (potato/quant.py).

No GPU and no llama.cpp binaries required: command builders are checked
against the pinned v0.6.0 usage strings, manifest I/O against temp dirs, and
``run_quant(dry_run=True)`` must never execute or write anything.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from potato import quant


class BuildQuantizeCommandTest(unittest.TestCase):
    """llama-quantize usage (pinned tools/quantize/quantize.cpp):
    [options] model-f32.gguf [model-quant.gguf] type [nthreads]"""

    def test_basic_invocation(self) -> None:
        cmd = quant.build_quantize_cmd("llama-quantize", "src.gguf", "dst.gguf", "Q4_K_M")
        self.assertEqual(
            cmd, ["llama-quantize", "src.gguf", "dst.gguf", "Q4_K_M"]
        )

    def test_imatrix_option_precedes_positionals(self) -> None:
        cmd = quant.build_quantize_cmd(
            "llama-quantize", "src.gguf", "dst.gguf", "Q5_K_M", imatrix="i.gguf"
        )
        self.assertEqual(cmd[1:3], ["--imatrix", "i.gguf"])
        self.assertEqual(cmd[3:], ["src.gguf", "dst.gguf", "Q5_K_M"])

    def test_threads_appended_last(self) -> None:
        cmd = quant.build_quantize_cmd(
            "llama-quantize", "s.gguf", "d.gguf", "Q6_K", threads=6
        )
        self.assertEqual(cmd[-2:], ["Q6_K", "6"])


class BuildImatrixCommandTest(unittest.TestCase):
    """llama-imatrix usage (pinned tools/imatrix/README.md):
    -m model.gguf -f text.txt [-o out] [--chunks N] [-ngl N]"""

    def test_basic_invocation(self) -> None:
        cmd = quant.build_imatrix_cmd(
            "llama-imatrix", "src.gguf", "wiki.train.raw", "out.gguf"
        )
        self.assertEqual(
            cmd,
            ["llama-imatrix", "-m", "src.gguf", "-f", "wiki.train.raw",
             "-o", "out.gguf"],
        )

    def test_optional_flags(self) -> None:
        cmd = quant.build_imatrix_cmd(
            "llama-imatrix", "m.gguf", "c.txt", "o.gguf", ngl=99, chunks=64
        )
        self.assertEqual(cmd[cmd.index("--chunks") + 1], "64")
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "99")


class VariantPathTest(unittest.TestCase):
    def test_names_are_deterministic(self) -> None:
        p = quant.variant_output_path("models/qwen-f16.gguf", "Q4_K_M", "models")
        self.assertEqual(Path(p).name, "qwen-f16-Q4_K_M.gguf")
        self.assertEqual(Path(p).parent, Path("models"))

    def test_defaults_to_source_directory(self) -> None:
        p = quant.variant_output_path("models/src.gguf", "Q3_K_M")
        self.assertEqual(Path(p).parent, Path("models"))


class ManifestTest(unittest.TestCase):
    def _write_source(self, directory: Path) -> Path:
        src = directory / "src-f16.gguf"
        src.write_bytes(b"GGUF" + b"\x00" * 64)
        return src

    def test_new_manifest_records_measured_source_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write_source(Path(tmp))
            m = quant.new_manifest(str(src), source_sha256="abc123")
            self.assertEqual(m["schema_version"], quant.MANIFEST_SCHEMA_VERSION)
            self.assertEqual(m["source"]["size_bytes"], 68)
            self.assertEqual(m["source"]["sha256"], "abc123")
            self.assertEqual(m["variants"], [])

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._write_source(Path(tmp))
            path = str(Path(tmp) / "quant_manifest.json")
            m = quant.new_manifest(str(src))
            m["variants"] = [
                quant.variant_entry(
                    "Q4_K_M", Path(tmp) / "no-file.gguf", None, 1.5, 0
                )
            ]
            quant.save_manifest(path, m)
            loaded = quant.load_manifest(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded["variants"]), 1)
            self.assertEqual(loaded["variants"][0]["type"], "Q4_K_M")
            # File is valid JSON on disk (readable by other tools).
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertIn("variants", raw)

    def test_load_missing_returns_none(self) -> None:
        self.assertIsNone(quant.load_manifest("does/not/exist.json"))

    def test_merge_supersedes_by_path_keeps_others(self) -> None:
        old = [
            {"type": "Q3_K_M", "path": "a-Q3_K_M.gguf", "size_bytes": 1},
            {"type": "Q4_K_M", "path": "a-Q4_K_M.gguf", "size_bytes": 2},
        ]
        new = [
            {"type": "Q4_K_M", "path": "a-Q4_K_M.gguf", "size_bytes": 99},
            {"type": "Q6_K", "path": "a-Q6_K.gguf", "size_bytes": 3},
        ]
        merged = quant.merge_variants(old, new)
        by_path = {v["path"]: v for v in merged}
        self.assertEqual(len(merged), 3)  # supersede, don't duplicate
        self.assertEqual(by_path["a-Q4_K_M.gguf"]["size_bytes"], 99)  # new wins
        self.assertEqual(by_path["a-Q3_K_M.gguf"]["size_bytes"], 1)  # kept

    def test_variant_entry_measures_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "x-Q4_K_M.gguf"
            payload = b"gguf-bytes"
            dest.write_bytes(payload)
            entry = quant.variant_entry("Q4_K_M", dest, "i.gguf", 2.0, 0)
            self.assertEqual(entry["size_bytes"], len(payload))
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(entry["imatrix"], "i.gguf")
            self.assertEqual(entry["exit_code"], 0)

    def test_sha256_of_missing_file_is_none(self) -> None:
        self.assertIsNone(quant.sha256_file("no/such/file.gguf"))


class RunQuantDryRunTest(unittest.TestCase):
    def test_dry_run_emits_commands_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src-f16.gguf"
            src.write_bytes(b"GGUF" + b"\x00" * 32)
            corpus = Path(tmp) / "wiki.train.raw"
            corpus.write_text("some calibration text\n", encoding="utf-8")
            out_dir = Path(tmp) / "out"
            result = quant.run_quant(
                str(src),
                types=["Q4_K_M", "Q5_K_M"],
                out_dir=str(out_dir),
                imatrix_from=str(corpus),
                imatrix_binary="llama-imatrix",
                quant_binary="llama-quantize",
                dry_run=True,
            )
            cmds = result["commands"]
            self.assertEqual(len(cmds), 3)  # 1 imatrix + 2 quantize
            self.assertEqual(cmds[0][0], "llama-imatrix")
            self.assertIn(str(corpus), cmds[0])
            self.assertEqual(cmds[1][0], "llama-quantize")
            self.assertEqual(cmds[1][1:3], ["--imatrix", str(out_dir / "src-f16.imatrix.gguf")])
            self.assertEqual(cmds[1][-1], "Q4_K_M")
            self.assertEqual(cmds[2][-1], "Q5_K_M")
            # Nothing was created: no out dir, no manifest, no variants.
            self.assertFalse(out_dir.exists())
            self.assertIsNone(result["manifest"])
            self.assertEqual(result["variants"], [])

    def test_missing_source_fails_with_guidance(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            quant.run_quant("missing.gguf", dry_run=True)
        self.assertIn("source GGUF not found", str(ctx.exception))
        self.assertIn("convert_hf_to_gguf", str(ctx.exception))  # dep-policy hint

    def test_empty_types_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "s.gguf"
            src.write_bytes(b"GGUF")
            with self.assertRaises(ValueError):
                quant.run_quant(str(src), types=[], dry_run=True)


if __name__ == "__main__":
    unittest.main()
