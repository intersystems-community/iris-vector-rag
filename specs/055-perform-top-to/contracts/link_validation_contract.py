"""
Contract Test: All documentation links must resolve correctly

This contract validates FR-005 and FR-006 from the feature specification:
- FR-005: All links to external resources must resolve correctly (no 404s)
- FR-006: All links to internal documentation must point to existing files

These tests will FAIL initially (no validation implementation exists yet).
After implementation, all tests must PASS.
"""

import re
from pathlib import Path
import requests
import pytest


def extract_links_from_markdown(md_file: Path):
    """
    Extract all markdown links from a file.

    Returns list of tuples: (line_number, link_text, link_url, link_type)
    """
    if not md_file.exists():
        return []

    links = []
    content = md_file.read_text()
    lines = content.split('\n')

    # Pattern for markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    for line_num, line in enumerate(lines, start=1):
        for match in re.finditer(link_pattern, line):
            text = match.group(1)
            url = match.group(2)

            # Determine link type
            if url.startswith('http://') or url.startswith('https://'):
                link_type = 'EXTERNAL'
            elif url.startswith('#'):
                link_type = 'ANCHOR'
            else:
                link_type = 'INTERNAL'

            links.append((line_num, text, url, link_type))

    return links


@pytest.mark.contract
def test_readme_external_links_resolve():
    """All external links in README.md must return HTTP 200"""
    readme = Path('README.md')
    links = extract_links_from_markdown(readme)

    external_links = [(line, text, url) for line, text, url, link_type in links
                      if link_type == 'EXTERNAL']

    broken_links = []
    for line_num, text, url in external_links:
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code >= 400:
                broken_links.append((line_num, url, response.status_code))
        except Exception as e:
            broken_links.append((line_num, url, str(e)))

    assert len(broken_links) == 0, \
        f"Found {len(broken_links)} broken external links in README.md:\n" + \
        "\n".join([f"  Line {line}: {url} ({status})" for line, url, status in broken_links])


@pytest.mark.contract
def test_readme_internal_links_exist():
    """All internal links in README.md must point to existing files"""
    readme = Path('README.md')
    links = extract_links_from_markdown(readme)

    internal_links = [(line, text, url) for line, text, url, link_type in links
                      if link_type == 'INTERNAL']

    broken_links = []
    for line_num, text, url in internal_links:
        # Remove anchor part (#section) if present
        file_path = url.split('#')[0]

        # Resolve relative to README location
        target = (readme.parent / file_path).resolve()

        if not target.exists():
            broken_links.append((line_num, url, target))

    assert len(broken_links) == 0, \
        f"Found {len(broken_links)} broken internal links in README.md:\n" + \
        "\n".join([f"  Line {line}: {url} → {target}" for line, url, target in broken_links])


@pytest.mark.contract
def test_all_docs_external_links_resolve():
    """All external links in all documentation files must resolve"""
    docs_dir = Path('docs')
    if not docs_dir.exists():
        pytest.skip("docs/ directory not found")

    all_broken_links = []

    # Check all markdown files in docs/
    for md_file in docs_dir.rglob('*.md'):
        links = extract_links_from_markdown(md_file)
        external_links = [(line, text, url) for line, text, url, link_type in links
                          if link_type == 'EXTERNAL']

        for line_num, text, url in external_links:
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                if response.status_code >= 400:
                    all_broken_links.append((md_file.name, line_num, url, response.status_code))
            except Exception as e:
                all_broken_links.append((md_file.name, line_num, url, str(e)))

    assert len(all_broken_links) == 0, \
        f"Found {len(all_broken_links)} broken external links across all docs:\n" + \
        "\n".join([f"  {file}:{line}: {url} ({status})"
                   for file, line, url, status in all_broken_links[:10]])  # Show first 10


@pytest.mark.contract
def test_all_docs_internal_links_exist():
    """All internal links in all documentation files must point to existing files"""
    docs_dir = Path('docs')
    if not docs_dir.exists():
        pytest.skip("docs/ directory not found")

    all_broken_links = []

    # Check all markdown files
    for md_file in docs_dir.rglob('*.md'):
        links = extract_links_from_markdown(md_file)
        internal_links = [(line, text, url) for line, text, url, link_type in links
                          if link_type == 'INTERNAL']

        for line_num, text, url in internal_links:
            # Remove anchor part if present
            file_path = url.split('#')[0]

            # Resolve relative to the markdown file's location
            target = (md_file.parent / file_path).resolve()

            if not target.exists():
                all_broken_links.append((md_file.name, line_num, url, target))

    assert len(all_broken_links) == 0, \
        f"Found {len(all_broken_links)} broken internal links across all docs:\n" + \
        "\n".join([f"  {file}:{line}: {url} → {target}"
                   for file, line, url, target in all_broken_links[:10]])  # Show first 10


if __name__ == '__main__':
    # Allow running this contract directly
    pytest.main([__file__, '-v'])
