"""Unit tests for bench parsing/command-building (potato/bench.py).

No GPU required. We feed a realistic llama-bench ``-o json`` payload (structure
copied from the pinned v0.6.0 tools/llama-bench/README.md) and assert the
section-8 mapping. This validates the parser without inventing numbers: the
fixture values are illustrative test data, not measured results.
"""

import json
import unittest

from potato import bench
from potato.schema import validate

# Two rows mirroring llama-bench v0.6.0 JSON: one prefill (n_gen==0),
# one decode (n_prompt==0). Values are test fixtures, NOT measured data.
FIXTURE = [
    {
        "build_commit": "d8123504",
        "build_number": 6000,
        "cpu_info": "Intel Core i5-9400F",
        "gpu_info": "NVIDIA GeForce GTX 1660 Ti",
        "backends": "CUDA",
        "model_filename": "models/tiny.gguf",
        "model_type": "llama 7B Q4_K - Small",
        "model_size": 4100000000,
        "model_n_params": 6700000000,
        "n_batch": 512,
        "n_ubatch": 512,
        "n_threads": 6,
        "type_k": "f16",
        "type_v": "f16",
        "n_gpu_layers": 99,
        "flash_attn": 1,
        "n_prompt": 512,
        "n_gen": 0,
        "test_time": "2026-10-11T12:00:00Z",
        "avg_ns": 100000000,
        "stddev_ns": 2000000,
        "avg_ts": 123.4,
        "stddev_ts": 5.6,
        "samples_ns": [99000000, 101000000, 100000000],
        "samples_ts": [120.0, 124.0, 126.0],
    },
    {
        "build_commit": "d8123504",
        "build_number": 6000,
        "cpu_info": "Intel Core i5-9400F",
        "gpu_info": "NVIDIA GeForce GTX 1660 Ti",
        "backends": "CUDA",
        "model_filename": "models/tiny.gguf",
        "model_type": "llama 7B Q4_K - Small",
        "n_batch": 512,
        "n_ubatch": 512,
        "n_threads": 6,
        "type_k": "f16",
        "type_v": "f16",
        "n_gpu_layers": 99,
        "flash_attn": 1,
        "n_prompt": 0,
        "n_gen": 128,
        "test_time": "2026-10-11T12:00:05Z",
        "avg_ns": 500000000,
        "stddev_ns": 10000000,
        "avg_ts": 12.3,
        "stddev_ts": 0.4,
        "samples_ns": [510000000, 490000000, 500000000],
        "samples_ts": [12.1, 12.3, 12.5],
    },
]


class CommandBuildTest(unittest.TestCase):
    def test_uses_verified_v060_flags(self) -> None:
        cmd = bench.build_bench_command(
            "llama-bench", "m.gguf", repeats=3, pp=512, tg=128, ngl=99, flash_attn="on",
            cache_type_k="q8_0", cache_type_v="q8_0",
        )
        self.assertIn("-r", cmd)
        self.assertIn("-p", cmd)
        self.assertIn("-n", cmd)
        self.assertIn("-o", cmd)
        self.assertIn("json", cmd)
        self.assertIn("-ngl", cmd)
        self.assertIn("-fa", cmd)
        self.assertIn("-ctk", cmd)
        self.assertIn("-ctv", cmd)
        # repeats must be explicitly 3 (llama-bench default is 5).
        self.assertEqual(cmd[cmd.index("-r") + 1], "3")


class ParseTest(unittest.TestCase):
    def test_classify_prefill_and_decode(self) -> None:
        self.assertEqual(bench._classify({"n_gen": 0}), "prefill")
        self.assertEqual(bench._classify({"n_gen": 128}), "decode")

    def test_parse_maps_avg_stddev_samples(self) -> None:
        parsed = bench.parse_llama_bench_json(json.dumps(FIXTURE))
        self.assertEqual(parsed["avg"]["prefill"], 123.4)
        self.assertEqual(parsed["avg"]["decode"], 12.3)
        self.assertEqual(parsed["stddev"]["decode"], 0.4)
        self.assertEqual(len(parsed["samples"]["decode"]), 3)

    def test_result_record_is_valid_section8(self) -> None:
        parsed = bench.parse_llama_bench_json(json.dumps(FIXTURE))
        record = bench.result_from_bench(parsed, "unit-test-001")
        data = record.to_dict()
        self.assertEqual(validate(record), [])
        self.assertEqual(data["metrics"]["prefill_tps"], 123.4)
        self.assertEqual(data["metrics"]["decode_tps"], 12.3)
        self.assertEqual(data["repeats"], 3)
        self.assertEqual(data["llama_cpp_commit"], "d8123504")
        self.assertEqual(data["gpu"], "NVIDIA GeForce GTX 1660 Ti")
        # Unmeasured section-8 metrics stay null (no invention).
        self.assertIsNone(data["metrics"]["vram_peak_mb"])
        self.assertIsNone(data["metrics"]["kl_mean"])
        self.assertIsNone(data["metrics"]["ppl"])


if __name__ == "__main__":
    unittest.main()
