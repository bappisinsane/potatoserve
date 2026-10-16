"""Unit tests for the Phase 2 quality gate (potato/quality.py).

No GPU and no llama.cpp binaries required: fixtures below were written from
the exact format strings in the pinned v0.6.0
third_party/llama.cpp/tools/perplexity/perplexity.cpp (lines 647 and 1892-1982)
- illustrative test data, NOT measured results.
"""

import unittest

from potato import quality

# Plain mode (no --kl-divergence): only the final estimate is printed.
PLAIN_OUTPUT = """
llama_model_loader: loaded meta data with 23 key-value pairs
load_backend: loading backend from ggml-vulkan.dll
Final estimate: PPL = 12.3456 +/- 0.07890
"""

# KL mode (--kl-divergence-base + --kl-divergence): the summary blocks below
# mirror the pinned printf formats, including the UTF-8 plus-minus and the
# Greek delta of the token-probability section.
KL_OUTPUT = """chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1     6.2100 ±   0.0300      0.00040 ±    0.00009      0.00003 ±     0.00000      0.150 ±   0.010    99.700 ±   0.050 %
   2     6.2400 ±   0.0350      0.00041 ±    0.00008      0.00002 ±     0.00000      0.148 ±   0.011    99.780 ±   0.040 %

====== Perplexity statistics ======
Mean PPL(Q)                   :   6.227711 ±   0.037833
Mean PPL(base)                :   6.225194 ±   0.037771
Cor(ln(PPL(Q)), ln(PPL(base))): 99.99%
Mean ln(PPL(Q)/PPL(base))     :   0.000404 ±   0.000086
Mean PPL(Q)/PPL(base)         :   1.000404 ±   0.000601
Mean PPL(Q)-PPL(base)         :   0.002517 ±   0.000536

====== KL divergence statistics ======
Mean    KLD:   0.000025 ±   0.000000
Maximum KLD:    0.012206
99.9%   KLD:    0.000799
99.0%   KLD:    0.000222
95.0%   KLD:    0.000200
90.0%   KLD:    0.000150
Median  KLD:    0.000013
10.0%   KLD:   -0.000002
 5.0%   KLD:   -0.000008
 1.0%   KLD:   -0.000023
 0.1%   KLD:   -0.000050
Minimum KLD:   -0.000059

====== Token probability statistics ======
Mean    Δp:  -0.000 ±  0.001 %
Median  Δp:   0.000%
RMS Δp    :   0.150 ±  0.001 %
Same top p:  99.739 ±  0.013 %
"""


class ParsePlainOutputTest(unittest.TestCase):
    def test_final_estimate_parses_ppl(self) -> None:
        parsed = quality.parse_perplexity_output(PLAIN_OUTPUT)
        self.assertAlmostEqual(parsed["ppl"], 12.3456)
        self.assertAlmostEqual(parsed["ppl_unc"], 0.07890)

    def test_plain_run_has_no_kl(self) -> None:
        parsed = quality.parse_perplexity_output(PLAIN_OUTPUT)
        self.assertIsNone(parsed["kl_mean"])
        self.assertIsNone(parsed["top1_agree"])


class ParseKlOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parsed = quality.parse_perplexity_output(KL_OUTPUT)

    def test_perplexity_block(self) -> None:
        self.assertAlmostEqual(self.parsed["ppl"], 6.227711)
        self.assertAlmostEqual(self.parsed["ppl_unc"], 0.037833)
        self.assertAlmostEqual(self.parsed["ppl_base"], 6.225194)
        self.assertAlmostEqual(self.parsed["correlation_pct"], 99.99)
        self.assertAlmostEqual(self.parsed["ppl_ratio"], 1.000404)

    def test_kl_statistics_block(self) -> None:
        self.assertAlmostEqual(self.parsed["kl_mean"], 0.000025)
        self.assertAlmostEqual(self.parsed["kl_mean_unc"], 0.000000)
        self.assertAlmostEqual(self.parsed["kl_median"], 0.000013)
        self.assertAlmostEqual(self.parsed["kl_max"], 0.012206)
        self.assertAlmostEqual(self.parsed["kl_min"], -0.000059)
        self.assertAlmostEqual(self.parsed["kl_percentiles"]["99.9"], 0.000799)
        self.assertAlmostEqual(self.parsed["kl_percentiles"]["99.0"], 0.000222)

    def test_top1_agree_is_a_fraction(self) -> None:
        # 99.739% agreement -> 0.99739 (rate metrics are stored 0..1).
        self.assertAlmostEqual(self.parsed["top1_agree"], 0.99739)
        self.assertAlmostEqual(self.parsed["top1_agree_unc"], 0.00013)

    def test_token_probability_block(self) -> None:
        self.assertAlmostEqual(self.parsed["delta_p_rms_pct"], 0.150)
        self.assertAlmostEqual(self.parsed["delta_p_mean_pct"], -0.000)

    def test_chunk_rows_never_match_summary_patterns(self) -> None:
        # The per-chunk table starts with the chunk index; it must not be
        # mistaken for a summary line (e.g. the row's PPL is not final ppl).
        parsed = quality.parse_perplexity_output(
            "   1     6.2100 ±   0.0300      0.00040 ±    0.00009      "
            "0.00003 ±     0.00000      0.150 ±   0.010    99.700 ±   0.050 %"
        )
        self.assertIsNone(parsed["ppl"])
        self.assertIsNone(parsed["kl_mean"])
        self.assertIsNone(parsed["top1_agree"])


class ParserRobustnessTest(unittest.TestCase):
    def test_ansi_colors_are_stripped(self) -> None:
        colored = "\x1b[32mMean    KLD:   0.000042 ±   0.000001\x1b[0m\n"
        parsed = quality.parse_perplexity_output(colored)
        self.assertAlmostEqual(parsed["kl_mean"], 0.000042)

    def test_mojibake_plus_minus_from_cp1252_decode(self) -> None:
        # UTF-8 bytes for '±' (C2 B1) decoded as latin-1/cp1252 -> 'Â±'.
        parsed = quality.parse_perplexity_output(
            "Mean    KLD:   0.000017 Â±   0.000000\n"
        )
        self.assertAlmostEqual(parsed["kl_mean"], 0.000017)

    def test_plus_minus_ascii_spelling(self) -> None:
        parsed = quality.parse_perplexity_output(
            "Same top p:  98.500 +/-  0.020 %\n"
        )
        self.assertAlmostEqual(parsed["top1_agree"], 0.985)

    def test_empty_output_yields_all_none(self) -> None:
        parsed = quality.parse_perplexity_output("")
        self.assertIsNone(parsed["ppl"])
        self.assertIsNone(parsed["kl_mean"])
        self.assertFalse(quality.has_measurement(parsed))


class BuildCommandTest(unittest.TestCase):
    """Flags must match the pinned v0.6.0 (verified in common/arg.cpp)."""

    def test_plain_mode_has_no_kl_flags(self) -> None:
        cmd = quality.build_perplexity_command(
            "llama-perplexity", "m.gguf", "wiki.test.raw"
        )
        self.assertEqual(cmd[0], "llama-perplexity")
        self.assertEqual(cmd[1:5], ["-m", "m.gguf", "-f", "wiki.test.raw"])
        self.assertIn("-c", cmd)
        self.assertNotIn("--kl-divergence-base", cmd)
        self.assertNotIn("--kl-divergence", cmd)

    def test_reference_mode_writes_base_without_computing(self) -> None:
        cmd = quality.build_perplexity_command(
            "llama-perplexity", "ref.gguf", "wiki.test.raw", kl_base="ref.kld"
        )
        i = cmd.index("--kl-divergence-base")
        self.assertEqual(cmd[i + 1], "ref.kld")
        self.assertNotIn("--kl-divergence", cmd)

    def test_kl_mode_adds_divergence_flag(self) -> None:
        cmd = quality.build_perplexity_command(
            "llama-perplexity", "q4.gguf", "wiki.test.raw",
            kl_base="ref.kld", kl_divergence=True,
        )
        i = cmd.index("--kl-divergence-base")
        self.assertEqual(cmd[i + 1], "ref.kld")
        self.assertIn("--kl-divergence", cmd)

    def test_optional_flags(self) -> None:
        cmd = quality.build_perplexity_command(
            "lp", "m.gguf", "c.txt", chunks=16, ngl=99, ctx=1024, stride=512,
        )
        self.assertEqual(cmd[cmd.index("--chunks") + 1], "16")
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "99")
        self.assertEqual(cmd[cmd.index("-c") + 1], "1024")
        self.assertEqual(cmd[cmd.index("--ppl-stride") + 1], "512")


class AggregateQualityTest(unittest.TestCase):
    def test_median_and_samples(self) -> None:
        runs = [
            {"kl_mean": 0.000030, "top1_agree": 0.9970, "ppl": 6.20},
            {"kl_mean": 0.000025, "top1_agree": 0.9974, "ppl": 6.23},
            {"kl_mean": 0.000020, "top1_agree": 0.9978, "ppl": 6.26},
        ]
        agg = quality.aggregate_quality(runs)
        self.assertAlmostEqual(agg["metrics"]["kl_mean"], 0.000025)  # median
        self.assertAlmostEqual(agg["metrics"]["ppl"], 6.23)
        self.assertEqual(agg["spread"]["kl_samples"], [0.000030, 0.000025, 0.000020])
        self.assertEqual(len(agg["spread"]["ppl_samples"]), 3)
        self.assertEqual(agg["spread"]["quality_detail"], runs[0])

    def test_missing_values_are_skipped_not_zeroed(self) -> None:
        runs = [
            {"kl_mean": None, "top1_agree": None, "ppl": 6.20},
            {"kl_mean": 0.000025, "top1_agree": 0.997, "ppl": None},
        ]
        agg = quality.aggregate_quality(runs)
        self.assertAlmostEqual(agg["metrics"]["kl_mean"], 0.000025)
        self.assertAlmostEqual(agg["metrics"]["ppl"], 6.20)
        self.assertEqual(agg["spread"]["kl_samples"], [0.000025])

    def test_empty_runs_yield_null_metrics(self) -> None:
        agg = quality.aggregate_quality([])
        self.assertIsNone(agg["metrics"]["kl_mean"])
        self.assertIsNone(agg["metrics"]["ppl"])


class RunQualityValidationTest(unittest.TestCase):
    """Pure validation - no subprocess is ever launched in these cases."""

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            quality.run_quality("m.gguf", "c.txt", mode="bogus")

    def test_kl_and_reference_modes_require_base(self) -> None:
        for mode in ("kl", "reference"):
            with self.assertRaises(ValueError):
                quality.run_quality("m.gguf", "c.txt", mode=mode, kl_base=None)

    def test_missing_reference_file_fails_loudly(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            quality.run_quality(
                "m.gguf", "c.txt", mode="kl", kl_base="definitely-missing.kld"
            )
        self.assertIn("KL base file not found", str(ctx.exception))
