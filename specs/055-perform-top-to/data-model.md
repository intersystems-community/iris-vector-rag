# Phase 1: Data Model - Documentation Structure

**Feature**: Documentation Review and README Optimization
**Date**: 2025-01-09
**Prerequisites**: research.md complete

## Entity Definitions

### 1. DocumentationFile

Represents a markdown documentation file in the repository.

**Fields**:
- `path`: string - Relative path from repository root (e.g., "README.md", "docs/USER_GUIDE.md")
- `type`: enum - Category of documentation
  - `README` - Main repository README
  - `GUIDE` - User or developer guide
  - `REFERENCE` - API or technical reference
  - `ARCHITECTURE` - System design documentation
  - `DEVELOPMENT` - Development process documentation
  - `ARCHIVED` - Obsolete or historical documentation
- `status`: enum - Current state
  - `CURRENT` - Up-to-date and accurate
  - `NEEDS_UPDATE` - Requires updates for accuracy
  - `OBSOLETE` - No longer relevant, should be archived
- `line_count`: integer - Number of lines in file
- `last_modified`: datetime - Last modification timestamp
- `links_to`: List[string] - Outbound links to other files or URLs
- `linked_from`: List[string] - Inbound links from other documentation
- `code_examples`: List[CodeExample] - Python code blocks in the file

**Validation Rules** (from spec FR-013, FR-014):
- README.md must have `line_count <= 400`
- All files must have clear, descriptive titles in first heading
- ARCHIVED files must not be linked from CURRENT files

**State Transitions**:
```
CURRENT → NEEDS_UPDATE (when code changes break examples or links)
NEEDS_UPDATE → CURRENT (after updates applied and validated)
CURRENT → OBSOLETE (when content no longer relevant)
OBSOLETE → ARCHIVED (when moved to docs/archived/)
```

### 2. CodeExample

Represents a Python code block extracted from documentation.

**Fields**:
- `file_path`: string - Path to documentation file containing this example
- `line_number`: integer - Starting line number in file
- `language`: string - Programming language (always "python" for this project)
- `code`: string - Raw code content
- `module_imports`: List[string] - Import statements extracted from code
- `executable`: boolean - Whether code can be executed in isolation
- `validation_result`: enum - Result of validation
  - `PASS` - Code executes successfully
  - `FAIL` - Syntax error or execution failure
  - `SKIP` - Code requires user input or external resources

**Validation Rules** (from spec FR-001, FR-002, FR-019):
- All `module_imports` must use `iris_vector_rag`, never `iris_rag`
- Executable examples must compile without `SyntaxError`
- All imports must reference existing modules in codebase

**Example**:
```python
CodeExample(
    file_path="README.md",
    line_number=84,
    language="python",
    code="from iris_vector_rag import create_pipeline\npipeline = create_pipeline('basic')",
    module_imports=["iris_vector_rag"],
    executable=True,
    validation_result="PASS"
)
```

### 3. Link

Represents a hyperlink in documentation (markdown or anchor).

**Fields**:
- `source_file`: string - Documentation file containing the link
- `line_number`: integer - Line number where link appears
- `target`: string - Link destination (URL or file path)
- `type`: enum - Link category
  - `INTERNAL` - Link to another file in repository
  - `EXTERNAL` - HTTP/HTTPS link to external resource
  - `ANCHOR` - Link to heading within same file (#section)
- `valid`: boolean - Whether link resolves successfully
- `http_status`: integer - HTTP status code for external links (200=OK, 404=not found)
- `error_message`: string - Description of validation failure (if any)

**Validation Rules** (from spec FR-005, FR-006):
- All EXTERNAL links must return `http_status == 200`
- All INTERNAL links must point to existing files
- All ANCHOR links must reference existing headings

**Example**:
```python
Link(
    source_file="README.md",
    line_number=451,
    target="docs/USER_GUIDE.md",
    type="INTERNAL",
    valid=True,
    http_status=None,
    error_message=None
)
```

### 4. ValidationResult

Represents the outcome of a documentation validation check.

**Fields**:
- `file_path`: string - Documentation file that was validated
- `check_type`: enum - Type of validation performed
  - `LINKS` - Link resolution checking
  - `EXAMPLES` - Code example execution
  - `MODULE_NAMES` - Module import verification
  - `STRUCTURE` - README length and structure
  - `PERFORMANCE_CLAIMS` - Benchmark verification
- `status`: enum - Overall result
  - `PASS` - All checks passed
  - `FAIL` - One or more checks failed
  - `WARNING` - Non-critical issues detected
- `errors`: List[ValidationError] - Specific failures detected
- `warnings`: List[string] - Non-critical issues
- `timestamp`: datetime - When validation was performed

**Validation Rules**:
- FAIL status requires at least one entry in `errors`
- All ValidationResults must be timestamped

**Example**:
```python
ValidationResult(
    file_path="README.md",
    check_type="MODULE_NAMES",
    status="FAIL",
    errors=[
        ValidationError(
            line=84,
            message="Uses 'iris_rag' instead of 'iris_vector_rag'",
            severity="ERROR"
        ),
        ValidationError(
            line=119,
            message="Uses 'iris_rag' instead of 'iris_vector_rag'",
            severity="ERROR"
        )
    ],
    warnings=[],
    timestamp="2025-01-09T10:30:00Z"
)
```

### 5. ValidationError

Represents a specific documentation error detected during validation.

**Fields**:
- `line`: integer - Line number where error occurs (or None if file-level)
- `message`: string - Human-readable error description
- `severity`: enum - Error importance
  - `ERROR` - Must be fixed (blocks validation)
  - `WARNING` - Should be fixed (does not block)
  - `INFO` - Informational only
- `fix_suggestion`: string - Suggested fix (if applicable)

**Example**:
```python
ValidationError(
    line=84,
    message="Uses 'iris_rag' instead of 'iris_vector_rag'",
    severity="ERROR",
    fix_suggestion="Replace 'from iris_rag import' with 'from iris_vector_rag import'"
)
```

## Relationships

```
DocumentationFile (1) ---has-many---> (N) CodeExample
DocumentationFile (1) ---has-many---> (N) Link
DocumentationFile (1) ---has-many---> (N) ValidationResult
ValidationResult (1) ---has-many---> (N) ValidationError
```

## Data Flow

1. **Discovery Phase**:
   ```
   Scan repository → Identify DocumentationFile entities → Extract CodeExample and Link entities
   ```

2. **Validation Phase**:
   ```
   For each DocumentationFile:
       Check links → Create Link entities with validation status
       Check code examples → Create CodeExample entities with execution results
       Check module names → Create ValidationResult for MODULE_NAMES check
       Check structure → Create ValidationResult for STRUCTURE check
   ```

3. **Reporting Phase**:
   ```
   Aggregate ValidationResults → Generate summary report → Identify files with FAIL status
   ```

4. **Update Phase**:
   ```
   For each file with status=NEEDS_UPDATE:
       Apply fixes → Re-run validation → Update status to CURRENT
   ```

## Implementation Notes

**Storage**: Validation results stored as JSON files in `.specify/validation/` directory
- `documentation_scan.json` - Complete scan of all documentation files
- `validation_results.json` - All validation results with timestamps
- `broken_links.json` - All links with valid=False
- `failed_examples.json` - All code examples with validation_result=FAIL

**Persistence**: Results persisted between runs to track fixes over time

**Idempotency**: Re-running validation always produces same results for unchanged files

## Validation Contracts

Based on this data model, the following contracts will be generated in Phase 1:

1. **link_validation_contract.py**: Validates all Link entities have valid=True
2. **code_example_contract.py**: Validates all CodeExample entities have validation_result=PASS
3. **module_name_contract.py**: Validates no CodeExample uses 'iris_rag' imports
4. **readme_structure_contract.py**: Validates README.md has line_count <= 400
5. **performance_claims_contract.py**: Validates benchmark claims are documented or cited
