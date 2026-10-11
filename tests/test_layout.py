"""Layout smoke test: asserts the repository matches docs/project-spec.md section 9.

Uses only the standard library so it runs in CI before any dependency is approved.
"""

import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "potato",
    "third_party",
    "lab",
    "evals",
    "evals/agent_tasks",
    "runs",
    "results",
    "docs",
    "tests",
    "scripts",
]

EXPECTED_FILES = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "docs/memory.md",
    "docs/rules.md",
    "docs/prd.md",
    "docs/project-spec.md",
    "docs/architecture.md",
    "docs/phases.md",
]


class LayoutTest(unittest.TestCase):
    def test_expected_directories_exist(self) -> None:
        for rel in EXPECTED_DIRS:
            with self.subTest(dir=rel):
                self.assertTrue((REPO / rel).is_dir(), f"missing directory: {rel}")

    def test_expected_files_exist(self) -> None:
        for rel in EXPECTED_FILES:
            with self.subTest(file=rel):
                self.assertTrue((REPO / rel).is_file(), f"missing file: {rel}")

    def test_package_imports(self) -> None:
        import potato  # noqa: F401

        self.assertTrue(hasattr(potato, "__version__"))


if __name__ == "__main__":
    unittest.main()
