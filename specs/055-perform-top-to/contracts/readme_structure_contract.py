"""
Contract Test: README must be concise and professional

This contract validates FR-007 through FR-013 from the feature specification:
- FR-007: README must communicate primary value proposition in first paragraph
- FR-008: README must include concise feature overview with links to detailed guides
- FR-009: README quick start must be testable and complete
- FR-010: README must use professional, clear language
- FR-011: README must be structured for rapid scanning
- FR-012: README must link to comprehensive documentation
- FR-013: README must not exceed 400 lines

These tests will FAIL initially if README is too long or poorly structured.
After implementation, all tests must PASS.
"""

import re
from pathlib import Path
import pytest


def count_lines(file_path: Path) -> int:
    """Count lines in a file"""
    if not file_path.exists():
        return 0
    return len(file_path.read_text().split('\n'))


def extract_first_paragraph(md_file: Path) -> str:
    """Extract first paragraph from markdown file (before first blank line)"""
    if not md_file.exists():
        return ""

    lines = md_file.read_text().split('\n')
    paragraph = []

    # Skip title (first line starting with #)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('# '):
            start_idx = i + 1
            break

    # Collect lines until first blank line
    for line in lines[start_idx:]:
        if line.strip() == '':
            break
        paragraph.append(line)

    return ' '.join(paragraph)


def extract_headings(md_file: Path):
    """Extract all headings from markdown file with their levels"""
    if not md_file.exists():
        return []

    content = md_file.read_text()
    headings = []

    for line_num, line in enumerate(content.split('\n'), start=1):
        if line.startswith('#'):
            # Count number of # to determine level
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            headings.append((line_num, level, title))

    return headings


def extract_links_to_docs(md_file: Path):
    """Extract links that point to documentation files"""
    if not md_file.exists():
        return []

    content = md_file.read_text()
    doc_links = []

    # Pattern for markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    for match in re.finditer(link_pattern, content):
        url = match.group(2)
        # Check if link points to documentation
        if url.startswith('docs/') or url.endswith('.md'):
            doc_links.append(url)

    return doc_links


@pytest.mark.contract
def test_readme_line_count_under_400():
    """README.md must not exceed 400 lines (FR-013)"""
    readme = Path('README.md')
    line_count = count_lines(readme)

    assert line_count <= 400, \
        f"README.md has {line_count} lines (exceeds 400-line limit by {line_count - 400} lines)"


@pytest.mark.contract
def test_readme_has_value_proposition_in_first_paragraph():
    """README first paragraph must communicate value proposition (FR-007)"""
    readme = Path('README.md')
    first_para = extract_first_paragraph(readme)

    # Check for key value proposition terms
    value_terms = [
        'RAG',  # Should mention RAG
        'IRIS',  # Should mention IRIS
    ]

    missing_terms = [term for term in value_terms if term.lower() not in first_para.lower()]

    assert len(first_para) >= 50, \
        f"First paragraph too short ({len(first_para)} chars) - should communicate value proposition"

    assert len(missing_terms) == 0, \
        f"First paragraph missing key terms: {missing_terms} - may not clearly communicate value"


@pytest.mark.contract
def test_readme_has_clear_heading_structure():
    """README must be structured with clear headings for scanning (FR-011)"""
    readme = Path('README.md')
    headings = extract_headings(readme)

    # Check we have reasonable number of top-level sections
    top_level_headings = [h for h in headings if h[1] == 2]  # Level 2 (##)

    assert len(top_level_headings) >= 5, \
        f"README has only {len(top_level_headings)} top-level sections - should have clear structure"

    # Check for essential sections
    heading_titles = [h[2] for h in headings]
    essential_sections = [
        'Quick Start',  # Or similar
        'Features',  # Or "Available", "Pipelines", etc.
        'Documentation',  # Must link to detailed docs
    ]

    # Check if any essential section exists (fuzzy match)
    for section in essential_sections:
        found = any(section.lower() in title.lower() for title in heading_titles)
        assert found, f"README missing essential section related to '{section}'"


@pytest.mark.contract
def test_readme_links_to_detailed_documentation():
    """README must link to comprehensive documentation (FR-008, FR-012)"""
    readme = Path('README.md')
    doc_links = extract_links_to_docs(readme)

    # Should link to at least some core documentation
    expected_docs = [
        'USER_GUIDE.md',
        'API_REFERENCE.md',
        'CONTRIBUTING.md',
    ]

    missing_docs = []
    for doc in expected_docs:
        if not any(doc in link for link in doc_links):
            missing_docs.append(doc)

    assert len(missing_docs) == 0, \
        f"README should link to {missing_docs} but doesn't - violates FR-012"


@pytest.mark.contract
def test_readme_quick_start_is_complete():
    """README Quick Start section must be complete and testable (FR-009)"""
    readme = Path('README.md')
    content = readme.read_text()

    # Check for Quick Start section
    assert '# Quick Start' in content or '## Quick Start' in content, \
        "README missing 'Quick Start' section"

    # Check Quick Start has essential elements
    quickstart_section = content.split('Quick Start')[1].split('#')[0] if 'Quick Start' in content else ""

    essential_elements = [
        'install',  # Installation instructions
        'python',  # Python code example
        'create_pipeline',  # Core API usage
    ]

    missing_elements = [elem for elem in essential_elements
                        if elem.lower() not in quickstart_section.lower()]

    assert len(missing_elements) == 0, \
        f"Quick Start section incomplete - missing: {missing_elements}"


@pytest.mark.contract
def test_readme_uses_professional_language():
    """README must use professional language (FR-010)"""
    readme = Path('README.md')
    content = readme.read_text().lower()

    # Check for informal language that should be avoided
    informal_terms = [
        'awesome',
        'super cool',
        'blazing fast',  # Acceptable if backed by benchmarks, but check
        'insane',
        'sick',
        'dope',
    ]

    # Allow 'blazing fast' if there's a performance section
    has_performance_section = 'performance' in content or 'benchmark' in content

    violations = []
    for term in informal_terms:
        if term == 'blazing fast' and has_performance_section:
            continue  # Acceptable with context
        if term in content:
            violations.append(term)

    # Note: This is a soft check - some informal terms OK with justification
    # Just flag if excessive
    assert len(violations) <= 1, \
        f"README uses informal language: {violations} - consider more professional tone"


@pytest.mark.contract
def test_readme_has_documentation_section():
    """README must have a Documentation section linking to all guides (FR-012)"""
    readme = Path('README.md')
    content = readme.read_text()

    # Check for Documentation section
    has_docs_section = any(term in content for term in ['## Documentation', '# Documentation'])

    assert has_docs_section, \
        "README must have a 'Documentation' section with links to detailed guides (FR-012)"


if __name__ == '__main__':
    # Allow running this contract directly
    pytest.main([__file__, '-v'])
