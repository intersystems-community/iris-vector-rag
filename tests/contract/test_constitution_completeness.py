"""
Contract tests for .specify/memory/constitution.md completeness.

These tests verify that the constitution contains all required IRIS-specific
testing principles as defined in Feature 047 (US3: Document IRIS Testing Principles).

Reference: specs/047-create-a-unified/tasks.md (T040-T048)
"""

import pytest
from pathlib import Path
import re


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def constitution_path():
    """Path to constitution file."""
    return Path(".specify/memory/constitution.md")


@pytest.fixture
def constitution_content(constitution_path):
    """Load constitution file content."""
    if not constitution_path.exists():
        pytest.fail(f"Constitution file not found: {constitution_path}")

    with open(constitution_path, "r", encoding="utf-8") as f:
        return f.read()


# ==============================================================================
# CONSTITUTION STRUCTURE TESTS
# ==============================================================================


@pytest.mark.contract
class TestConstitutionStructure:
    """Contract tests for constitution file structure."""

    def test_constitution_file_exists(self, constitution_path):
        """Constitution file exists at .specify/memory/constitution.md."""
        assert constitution_path.exists(), \
            f"Constitution file not found at {constitution_path}"

    def test_constitution_not_template(self, constitution_content):
        """Constitution is not the default template."""
        # Template contains placeholder text like [PROJECT_NAME]
        assert "[PROJECT_NAME]" not in constitution_content, \
            "Constitution still contains template placeholders"
        assert "[PRINCIPLE_1_NAME]" not in constitution_content, \
            "Constitution still contains principle placeholders"

    def test_has_project_name(self, constitution_content):
        """Constitution has project name in header."""
        # Should have "# IRIS RAG Templates Constitution" or similar
        assert re.search(r"#\s+\w+.*Constitution", constitution_content), \
            "Constitution missing project name header"


# ==============================================================================
# IRIS TESTING PRINCIPLES TESTS
# ==============================================================================


@pytest.mark.contract
class TestIRISTestingPrinciples:
    """Contract tests for IRIS-specific testing principles."""

    def test_has_vector_varchar_principle(self, constitution_content):
        """Constitution documents VECTOR/VARCHAR client limitation."""
        # Should mention:
        # - VECTOR columns cannot be inserted directly via SQL
        # - Must use TO_VECTOR() function
        # - Client-side limitation, not server-side

        content_lower = constitution_content.lower()

        assert "vector" in content_lower, \
            "Constitution missing VECTOR type discussion"
        assert "to_vector" in content_lower or "to_vector()" in content_lower, \
            "Constitution missing TO_VECTOR() function requirement"

        # Should explain the limitation
        assert any(phrase in content_lower for phrase in [
            "client limitation",
            "cannot insert",
            "cannot be inserted",
            "must use to_vector"
        ]), "Constitution missing VECTOR client limitation explanation"

    def test_has_dat_fixture_first_principle(self, constitution_content):
        """Constitution documents .DAT fixture-first approach."""
        # Should mention:
        # - .DAT fixtures are 100-200x faster than JSON
        # - Decision tree for when to use .DAT vs JSON vs programmatic
        # - iris-devtools integration

        content_lower = constitution_content.lower()

        assert ".dat" in content_lower or "dat fixture" in content_lower, \
            "Constitution missing .DAT fixture discussion"
        assert any(phrase in content_lower for phrase in [
            "iris-devtools",
            "100x faster",
            "200x faster",
            "faster than json"
        ]), "Constitution missing .DAT performance benefits"

    def test_has_test_isolation_principle(self, constitution_content):
        """Constitution documents test isolation by database state."""
        # Should mention:
        # - Fixtures provide isolated database states
        # - Checksum validation ensures reproducibility
        # - Version compatibility checking

        content_lower = constitution_content.lower()

        assert any(phrase in content_lower for phrase in [
            "test isolation",
            "isolated",
            "isolation"
        ]), "Constitution missing test isolation discussion"

        assert any(phrase in content_lower for phrase in [
            "checksum",
            "reproducible",
            "reproducibility"
        ]), "Constitution missing checksum/reproducibility discussion"

    def test_has_embedding_standards_principle(self, constitution_content):
        """Constitution documents embedding generation standards."""
        # Should mention:
        # - Default 384 dimensions (all-MiniLM-L6-v2)
        # - Sentence-transformers integration
        # - NULL handling (zero vectors)

        content_lower = constitution_content.lower()

        assert "embedding" in content_lower, \
            "Constitution missing embedding discussion"
        assert "384" in constitution_content or "dimension" in content_lower, \
            "Constitution missing embedding dimension standard"

    def test_has_backend_mode_awareness_principle(self, constitution_content):
        """Constitution documents backend mode awareness."""
        # Should mention:
        # - Community Edition vs Enterprise Edition
        # - Connection pooling differences
        # - Reference to Feature 035

        content_lower = constitution_content.lower()

        assert any(phrase in content_lower for phrase in [
            "backend mode",
            "community edition",
            "enterprise edition",
            "connection pool"
        ]), "Constitution missing backend mode discussion"


# ==============================================================================
# PRINCIPLE COMPLETENESS TESTS
# ==============================================================================


@pytest.mark.contract
class TestPrincipleCompleteness:
    """Tests verifying all 5 principles are documented."""

    def test_all_five_principles_present(self, constitution_content):
        """All 5 IRIS testing principles are documented."""
        content_lower = constitution_content.lower()

        principles = {
            "VECTOR/VARCHAR": any(phrase in content_lower for phrase in [
                "to_vector", "vector column", "client limitation"
            ]),
            ".DAT Fixture-First": any(phrase in content_lower for phrase in [
                ".dat", "dat fixture", "iris-devtools"
            ]),
            "Test Isolation": any(phrase in content_lower for phrase in [
                "isolation", "isolated", "checksum"
            ]),
            "Embedding Standards": any(phrase in content_lower for phrase in [
                "embedding", "384", "dimension"
            ]),
            "Backend Mode": any(phrase in content_lower for phrase in [
                "backend mode", "community", "enterprise", "connection pool"
            ]),
        }

        missing = [name for name, present in principles.items() if not present]

        assert not missing, \
            f"Constitution missing principles: {', '.join(missing)}"

    def test_has_examples_or_code_blocks(self, constitution_content):
        """Constitution includes examples or code blocks."""
        # Principles should have practical examples

        # Check for code blocks (markdown fenced code)
        has_code_blocks = "```" in constitution_content

        # Check for examples section
        has_examples = any(word in constitution_content.lower() for word in [
            "example", "examples", "usage", "how to"
        ])

        assert has_code_blocks or has_examples, \
            "Constitution missing practical examples or code blocks"

    def test_has_decision_trees_or_guidelines(self, constitution_content):
        """Constitution includes decision trees or when-to-use guidelines."""
        content_lower = constitution_content.lower()

        # Check for decision-making guidance
        has_guidance = any(phrase in content_lower for phrase in [
            "when to",
            "use when",
            "decision",
            "choose",
            "prefer",
            "should use"
        ])

        assert has_guidance, \
            "Constitution missing decision trees or when-to-use guidelines"


# ==============================================================================
# CROSS-REFERENCE TESTS
# ==============================================================================


@pytest.mark.contract
class TestConstitutionReferences:
    """Tests for cross-references to other documentation."""

    def test_references_feature_035(self, constitution_content):
        """Constitution references Feature 035 (backend mode)."""
        # Should reference Feature 035 or backend configuration
        assert any(ref in constitution_content for ref in [
            "Feature 035",
            "035",
            "backend_configuration",
            "backend mode"
        ]), "Constitution missing reference to Feature 035"

    def test_references_fixture_readme(self, constitution_content):
        """Constitution references tests/fixtures/README.md or fixture documentation."""
        content_lower = constitution_content.lower()

        # Should reference fixture documentation
        assert any(ref in content_lower for ref in [
            "tests/fixtures",
            "fixture readme",
            "fixture documentation",
            "fixturemanager"
        ]), "Constitution missing reference to fixture documentation"


# ==============================================================================
# FORMATTING TESTS
# ==============================================================================


@pytest.mark.contract
class TestConstitutionFormatting:
    """Tests for proper markdown formatting."""

    def test_uses_markdown_headers(self, constitution_content):
        """Constitution uses proper markdown headers."""
        # Should have multiple levels of headers
        assert re.search(r"^#+\s+.+", constitution_content, re.MULTILINE), \
            "Constitution missing markdown headers"

    def test_has_numbered_or_named_principles(self, constitution_content):
        """Principles are numbered or named (I, II, III or Principle 1, 2, 3)."""
        # Check for numbered principles
        has_numbering = any(pattern in constitution_content for pattern in [
            "## I.", "## II.", "## III.", "## IV.", "## V.",
            "### I.", "### II.", "### III.", "### IV.", "### V.",
            "Principle 1", "Principle 2", "Principle 3", "Principle 4", "Principle 5",
        ])

        assert has_numbering, \
            "Constitution principles not numbered or clearly named"

    def test_has_version_and_dates(self, constitution_content):
        """Constitution has version and ratification date."""
        content_lower = constitution_content.lower()

        assert "version" in content_lower, \
            "Constitution missing version information"
        assert any(word in content_lower for word in ["ratified", "date", "amended"]), \
            "Constitution missing date information"
