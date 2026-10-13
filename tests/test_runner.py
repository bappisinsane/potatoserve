"""Unit tests for the benchmark runner / VRAM sampler (potato/runner.py).

No GPU required. nvidia-smi may or may not be present on the CI machine, so
tests never assert a specific VRAM number. What they DO pin down:

  * when nvidia-smi is unavailable the peaks stay None (honest "unknown"),
    never a fabricated 0 or a guess;
  * ``run_benchmark`` captures stdout/exit code and never raises on a normal
    (non-zero) exit, so callers get partial output;
  * a timeout is reported as timed_out=True rather than raising, and the
    process is not allowed to run forever.
"""

import sys
import unittest

from potato import runner


class VramSamplerTest(unittest.TestCase):
    def test_missing_nvidia_smi_yields_none_peaks(self) -> None:
        # On a machine without nvidia-smi (or with it broken), peaks must be
        # None - an honest unknown, never a fabricated number.
        sampler = runner.VramSampler()
        sampler.start()
        sampler.stop()
        if runner.shutil.which("nvidia-smi") is None:
            self.assertIsNone(sampler.peak_mb)
            self.assertIsNone(sampler.baseline_mb)
            self.assertIsNone(sampler.delta_mb)
            self.assertEqual(sampler.samples, [])

    def test_sample_once_without_smi_is_none_and_no_samples(self) -> None:
        sampler = runner.VramSampler()
        if runner.shutil.which("nvidia-smi") is None:
            self.assertIsNone(sampler.sample_once())
            self.assertEqual(sampler.samples, [])

    def test_delta_none_when_unmeasured(self) -> None:
        sampler = runner.VramSampler()
        # Neither baseline nor peak set -> delta must be None (not 0).
        self.assertIsNone(sampler.delta_mb)


class RunBenchmarkTest(unittest.TestCase):
    def test_captures_stdout_and_exit_code(self) -> None:
        # A harmless command that prints a line; no GPU involved.
        code = "import sys; print('hello-bench'); sys.exit(0)"
        run = runner.run_benchmark([sys.executable, "-c", code])
        self.assertTrue(run.ok())
        self.assertEqual(run.exit_code, 0)
        self.assertIn("hello-bench", run.stdout)
        self.assertFalse(run.timed_out)

    def test_nonzero_exit_does_not_raise(self) -> None:
        code = "import sys; sys.stderr.write('boom'); sys.exit(3)"
        run = runner.run_benchmark([sys.executable, "-c", code])
        self.assertFalse(run.ok())
        self.assertEqual(run.exit_code, 3)
        self.assertIn("boom", run.stderr)

    def test_timeout_is_reported_not_raised(self) -> None:
        # Sleep longer than the timeout; the runner must return timed_out=True
        # quickly instead of hanging or raising.
        code = "import time; time.sleep(5)"
        run = runner.run_benchmark([sys.executable, "-c", code], timeout_s=0.5)
        self.assertTrue(run.timed_out)
        self.assertFalse(run.ok())
        self.assertLess(run.duration_s, 5.0)

    def test_missing_executable_raises_filenotfound(self) -> None:
        with self.assertRaises(FileNotFoundError):
            runner.run_benchmark(["definitely-not-a-real-binary-xyz"])


if __name__ == "__main__":
    unittest.main()
