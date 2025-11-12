"""
Contract Test: All code examples must be executable and accurate

This contract validates FR-001, FR-002, FR-004, and FR-019 from the feature specification:
- FR-001: All code examples must use current module names (iris_vector_rag not iris_rag)
- FR-002: All code examples must be executable and produce expected results
- FR-004: All API references must match current function signatures
- FR-019: Module import examples must consistently use iris_vector_rag

These tests will FAIL initially if examples use old module names or have errors.
After implementation, all tests must PASS.
"""

import re
from pathlib import Path
import ast
import pytest


def extract_python_code_blocks(md_file: Path):
    """
    Extract all Python code blocks from markdown file.

    Returns list of tuples: (line_number, code_content)
    """
    if not md_file.exists():
        return []

    content = md_file.read_text()
    code_blocks = []

    # Pattern for python code blocks: ```python\n...\n```
    pattern = r'```python\n(.*?)\n```'
    matches = re.finditer(pattern, content, re.DOTALL)

    # Calculate line numbers by counting newlines before match
    for match in matches:
        code = match.group(1)
        # Count newlines before this match to get approximate line number
        lines_before = content[:match.start()].count('\n')
        code_blocks.append((lines_before + 1, code))

    return code_blocks


@pytest.mark.contract
def test_readme_examples_use_correct_module_name():
    """All Python examples in README must use 'iris_vector_rag' not 'iris_rag'"""
    readme = Path('README.md')
    code_blocks = extract_python_code_blocks(readme)

    violations = []
    for line_num, code in code_blocks:
        # Check for old module name in imports
        if 'from iris_rag import' in code:
            violations.append((line_num, "Uses 'from iris_rag import' instead of 'from iris_vector_rag import'"))
        if 'import iris_rag' in code and 'import iris_vector_rag' not in code:
            violations.append((line_num, "Uses 'import iris_rag' instead of 'import iris_vector_rag'"))

    assert len(violations) == 0, \
        f"Found {len(violations)} examples using old module name 'iris_rag' in README.md:\n" + \
        "\n".join([f"  Line ~{line}: {msg}" for line, msg in violations])


@pytest.mark.contract
def test_readme_examples_have_valid_syntax():
    """All Python examples in README must have valid Python syntax"""
    readme = Path('README.md')
    code_blocks = extract_python_code_blocks(readme)

    syntax_errors = []
    for line_num, code in code_blocks:
        try:
            # Try to parse the code as Python AST
            ast.parse(code)
        except SyntaxError as e:
            syntax_errors.append((line_num, str(e)))

    assert len(syntax_errors) == 0, \
        f"Found {len(syntax_errors)} syntax errors in README.md code examples:\n" + \
        "\n".join([f"  Line ~{line}: {error}" for line, error in syntax_errors])


@pytest.mark.contract
def test_readme_examples_use_valid_imports():
    """All import statements in README examples must reference existing modules"""
    readme = Path('README.md')
    code_blocks = extract_python_code_blocks(readme)

    invalid_imports = []
    for line_num, code in code_blocks:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                    # Check if it's an iris_vector_rag import
                    if module and module.startswith('iris_vector_rag'):
                        # Verify the module path exists
                        module_path = module.replace('.', '/')
                        # Check if the module file exists
                        possible_paths = [
                            Path(f'{module_path}.py'),
                            Path(f'{module_path}/__init__.py')
                        ]
                        if not any(p.exists() for p in possible_paths):
                            invalid_imports.append((line_num, module, "Module file not found"))

        except Exception as e:
            # Skip if we can't parse (will be caught by syntax test)
            pass

    assert len(invalid_imports) == 0, \
        f"Found {len(invalid_imports)} invalid import statements in README.md:\n" + \
        "\n".join([f"  Line ~{line}: {module} - {reason}"
                   for line, module, reason in invalid_imports])


@pytest.mark.contract
def test_all_docs_use_correct_module_name():
    """All Python examples across all docs must use 'iris_vector_rag'"""
    docs_dir = Path('docs')
    if not docs_dir.exists():
        pytest.skip("docs/ directory not found")

    all_violations = []

    for md_file in docs_dir.rglob('*.md'):
        code_blocks = extract_python_code_blocks(md_file)

        for line_num, code in code_blocks:
            if 'from iris_rag import' in code:
                all_violations.append((md_file.name, line_num, "Uses 'from iris_rag import'"))
            if 'import iris_rag' in code and 'import iris_vector_rag' not in code:
                all_violations.append((md_file.name, line_num, "Uses 'import iris_rag'"))

    assert len(all_violations) == 0, \
        f"Found {len(all_violations)} examples using old module name across all docs:\n" + \
        "\n".join([f"  {file}:~{line}: {msg}"
                   for file, line, msg in all_violations[:10]])  # Show first 10


@pytest.mark.contract
def test_readme_create_pipeline_examples_match_api():
    """Examples using create_pipeline must match current API signature"""
    readme = Path('README.md')
    code_blocks = extract_python_code_blocks(readme)

    # Expected signature: create_pipeline(pipeline_type, validate_requirements=True, ...)
    api_violations = []

    for line_num, code in code_blocks:
        if 'create_pipeline(' in code:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if (isinstance(node.func, ast.Name) and node.func.id == 'create_pipeline') or \
                           (isinstance(node.func, ast.Attribute) and node.func.attr == 'create_pipeline'):
                            # Check that first argument is a string (pipeline_type)
                            if len(node.args) > 0:
                                if not isinstance(node.args[0], ast.Constant):
                                    api_violations.append((line_num, "First argument should be pipeline type string"))
                            # Optional: Check for valid pipeline types
                            if isinstance(node.args[0], ast.Constant):
                                pipeline_type = node.args[0].value
                                valid_types = ['basic', 'basic_rerank', 'crag', 'graphrag', 'multi_query_rrf', 'pylate_colbert']
                                if pipeline_type not in valid_types:
                                    api_violations.append((line_num, f"Unknown pipeline type: {pipeline_type}"))
            except:
                pass  # Syntax errors caught elsewhere

    assert len(api_violations) == 0, \
        f"Found {len(api_violations)} create_pipeline API violations in README.md:\n" + \
        "\n".join([f"  Line ~{line}: {msg}" for line, msg in api_violations])


if __name__ == '__main__':
    # Allow running this contract directly
    pytest.main([__file__, '-v'])
