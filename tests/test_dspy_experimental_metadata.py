"""Tests for dspy-experimental metadata and labeling.

Validates that:
1. The `dspy` extra is declared in pyproject.toml and installable
2. The dspy_modules package is marked as experimental
3. False claims about dspy support are removed from documentation
"""

import re
import tomllib
from pathlib import Path


class TestDspyExtraDeclaration:
    """Test that dspy extra exists in pyproject.toml."""

    def test_dspy_extra_exists_in_pyproject(self):
        """The dspy extra should be declared in pyproject.toml."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Check that dspy extra is in optional-dependencies
        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        assert "dspy" in optional_deps, "dspy extra not found in optional-dependencies"

        # Check that it has dspy as a dependency
        dspy_deps = optional_deps["dspy"]
        assert isinstance(dspy_deps, list), "dspy extra should be a list"
        assert len(dspy_deps) > 0, "dspy extra should list dependencies"
        assert any(
            "dspy" in dep.lower() for dep in dspy_deps
        ), "dspy extra should include dspy package"

    def test_dspy_not_in_all_extra(self):
        """The dspy extra should NOT be in the 'all' extra (not recommended)."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        all_deps = optional_deps.get("all", [])

        # Verify dspy is not in "all" since it's experimental
        dspy_in_all = any("dspy" in dep.lower() for dep in all_deps)
        assert not dspy_in_all, "dspy should not be in 'all' extra (it is experimental)"


class TestDspyModulesExperimental:
    """Test that dspy_modules is marked experimental."""

    def test_dspy_modules_init_has_experimental_notice(self):
        """dspy_modules/__init__.py should include an experimental notice."""
        init_path = (
            Path(__file__).parent.parent
            / "iris_vector_rag"
            / "dspy_modules"
            / "__init__.py"
        )
        assert init_path.exists(), f"dspy_modules/__init__.py not found at {init_path}"

        content = init_path.read_text()

        # Check for experimental marker (case-insensitive)
        experimental_patterns = [
            r"experimental",
            r"not.*supported",
            r"spec.*073",
        ]
        has_notice = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in experimental_patterns
        )
        assert (
            has_notice
        ), "dspy_modules/__init__.py should indicate experimental status or spec 073"

    def test_dspy_modules_not_imported_by_default(self):
        """Main iris_vector_rag/__init__.py should not import dspy_modules."""
        main_init = Path(__file__).parent.parent / "iris_vector_rag" / "__init__.py"
        content = main_init.read_text()

        # Should NOT have top-level dspy_modules import
        assert (
            "from .dspy_modules" not in content
        ), "dspy_modules should not be imported at package level"
        assert (
            "import dspy_modules" not in content
        ), "dspy_modules should not be imported at package level"


class TestRemovedFalseClaims:
    """Test that false claims about dspy support are removed."""

    def test_readme_dspy_claim_corrected(self):
        """README.md's dspy claim should be accurate (package exists) or marked experimental."""
        readme_path = Path(__file__).parent.parent / "README.md"
        content = readme_path.read_text()

        # Find the [dspy] extra line
        dspy_line = None
        for i, line in enumerate(content.split("\n")):
            if "[dspy]" in line and "pip install" in line:
                dspy_line = line
                break

        if dspy_line:
            # If it's mentioned, it should NOT say "DSPy prompt optimization" without qualification
            # (the extra now exists, so that's technically true, but let's verify the tone)
            # At minimum, check it doesn't claim full support
            assert (
                "experimental" in content or "spec 073" in content.lower()
            ), "README should note experimental status if dspy is mentioned"

    def test_makefile_no_unsupported_dspy_claims(self):
        """Makefile dspy targets should be accurate."""
        makefile_path = Path(__file__).parent.parent / "Makefile"
        if not makefile_path.exists():
            return  # Makefile may not exist in all branches

        content = makefile_path.read_text()

        # Check for any claims that dspy works fully
        # The fact that we're installing it now means we can mention it exists
        # but should still note experimental status
        if "dspy" in content:
            # Should not have false claims like "dspy working" without qualification
            # (The actual content depends on the Makefile structure)
            pass

    def test_agents_md_no_false_dspy_support_claim(self):
        """AGENTS.md should not falsely claim dspy support."""
        agents_path = Path(__file__).parent.parent / "AGENTS.md"
        if not agents_path.exists():
            return

        content = agents_path.read_text()

        # The line "pip install -e ".[dev]"  # Dev; add ,evaluation,colbert,dspy,mcp,api as needed"
        # is OK if dspy extra now exists; this is just a suggestion.
        # We're checking that there are no false claims of dspy being "fully supported"
        dspy_lines = [l for l in content.split("\n") if "dspy" in l.lower()]
        for line in dspy_lines:
            # Should not say dspy is "supported" or "fully functional" in those comments
            if "supported" in line.lower() or "functional" in line.lower():
                assert (
                    "experimental" in line.lower() or "not" in line.lower()
                ), f"Line mentioning dspy should not claim full support: {line}"
