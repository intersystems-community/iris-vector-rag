# Contract: Requirement-to-Task Mapping Validation

**Contract ID**: MAP-001
**Version**: 1.0.0
**Component**: scripts/validate_task_mapping.py

## Purpose

Define the behavior of the requirement-to-task mapping validator that ensures all functional requirements and edge cases have corresponding implementation tasks.

## Requirements

This contract implements:
- **FR-014**: All functional requirements must have tasks
- **FR-015**: Validate requirement-to-task mapping
- **FR-016**: Edge cases must have test tasks

## Interface Definition

### Command Line Interface

```bash
# Validate current feature
python scripts/validate_task_mapping.py

# Validate specific feature
python scripts/validate_task_mapping.py --feature 025-fixes-for-testing

# Generate gap report
python scripts/validate_task_mapping.py --report gaps.md

# Strict mode (fails on gaps)
python scripts/validate_task_mapping.py --strict
```

### Python API

```python
from scripts.validate_task_mapping import TaskMapper

mapper = TaskMapper(feature_dir='specs/025-fixes-for-testing')
mappings = mapper.analyze_mappings()
gaps = mapper.find_gaps(mappings)
report = mapper.generate_report(gaps)
```

### Configuration

```yaml
# .task-mapping.yml
patterns:
  requirement_id: '\*\*FR-(\d{3})\*\*'
  task_reference: 'FR-(\d{3})'
  edge_case: '^- (What happens when .+\?)$'
  task_id: '^### T(\d{3})'

files:
  spec: 'spec.md'
  tasks: 'tasks.md'

validation:
  require_explicit_references: true
  allow_implicit_mapping: false
  edge_case_prefix: 'Edge:'
```

## Behavior Specifications

### REQ-1: Requirement Extraction

**Given** a spec.md file
**When** extracting requirements
**Then** find all FR-XXX patterns and edge cases

**Extraction Logic**:
```python
def extract_requirements(spec_content):
    requirements = []

    # Extract functional requirements
    fr_pattern = re.compile(r'\*\*FR-(\d{3})\*\*:?\s*(.+)')
    for match in fr_pattern.finditer(spec_content):
        requirements.append({
            'id': f'FR-{match.group(1)}',
            'text': match.group(2).strip(),
            'type': 'functional'
        })

    # Extract edge cases
    edge_pattern = re.compile(r'^- (What happens when .+\?)$', re.MULTILINE)
    for match in edge_pattern.finditer(spec_content):
        requirements.append({
            'id': f'Edge:{hash(match.group(1))[:6]}',
            'text': match.group(1),
            'type': 'edge_case'
        })

    return requirements
```

### REQ-2: Task Extraction

**Given** a tasks.md file
**When** extracting tasks
**Then** find all task IDs and descriptions

**Task Structure**:
```python
def extract_tasks(tasks_content):
    tasks = []
    task_pattern = re.compile(
        r'^### T(\d{3})(?:\s*\[P\])?: (.+)$',
        re.MULTILINE
    )

    for match in task_pattern.finditer(tasks_content):
        task_id = f'T{match.group(1)}'
        description = match.group(2)

        # Extract referenced requirements
        refs = extract_requirement_refs(description)

        tasks.append({
            'id': task_id,
            'description': description,
            'references': refs
        })

    return tasks
```

### REQ-3: Mapping Analysis

**Given** requirements and tasks
**When** analyzing mappings
**Then** identify which requirements have tasks

**Mapping Logic**:
```python
def analyze_mappings(requirements, tasks):
    mappings = []

    for req in requirements:
        req_id = req['id']
        # Find tasks that reference this requirement
        covering_tasks = [
            task for task in tasks
            if req_id in task['references'] or
               req_id in task['description']
        ]

        mapping = RequirementTaskMapping(
            requirement_id=req_id,
            requirement_text=req['text'],
            task_ids=[t['id'] for t in covering_tasks],
            coverage_status='COVERED' if covering_tasks else 'MISSING'
        )
        mappings.append(mapping)

    return mappings
```

### REQ-4: Gap Detection

**Given** requirement mappings
**When** checking coverage
**Then** identify requirements without tasks

**Gap Categories**:
1. **Missing Coverage**: Requirement with no tasks
2. **Partial Coverage**: Complex requirement with insufficient tasks
3. **Edge Case Gaps**: Edge cases without test tasks

```python
def find_gaps(mappings):
    gaps = {
        'missing': [],
        'partial': [],
        'edge_cases': []
    }

    for mapping in mappings:
        if mapping.coverage_status == 'MISSING':
            if mapping.requirement_id.startswith('Edge:'):
                gaps['edge_cases'].append(mapping)
            else:
                gaps['missing'].append(mapping)
        elif is_partial_coverage(mapping):
            gaps['partial'].append(mapping)

    return gaps
```

### REQ-5: Report Generation

**Given** coverage gaps
**When** generating report
**Then** create actionable markdown report

**Report Format**:
```markdown
# Requirement-to-Task Mapping Report

**Feature**: 025-fixes-for-testing
**Date**: 2024-10-04
**Coverage**: 17/20 (85%)

## Summary
- ✅ Covered Requirements: 17
- ❌ Missing Requirements: 3
- ⚠️  Partial Coverage: 0
- 🔸 Edge Cases Missing Tests: 2

## Missing Coverage

### Functional Requirements
1. **FR-007**: System MUST provide coverage threshold warnings
   - Status: NO TASKS FOUND
   - Suggested Task: Implement coverage warning hooks

2. **FR-013**: Error messages MUST be actionable
   - Status: NO TASKS FOUND
   - Suggested Task: Add error message validation

### Edge Cases Without Tests
1. **Edge:a3f8c2**: What happens when coverage calculation fails?
   - Status: NO TEST TASK
   - Suggested Task: Add test for coverage failure handling

## Task Suggestions

Based on gaps found, consider adding these tasks:

```yaml
T086: Implement pytest hook for coverage threshold warnings (FR-007)
T087: Add error message validation plugin (FR-013)
T088: Test edge case - coverage calculation failure
```

## Validation Details

### Requirement Extraction
- Source: specs/025-fixes-for-testing/spec.md
- Requirements Found: 20 (20 functional, 4 edge cases)

### Task Analysis
- Source: specs/025-fixes-for-testing/tasks.md
- Tasks Found: 85
- Tasks with References: 65
```

## Error Handling

### ERR-1: Missing Files

**Given** spec.md or tasks.md missing
**When** running validation
**Then** provide helpful error message

```
Error: Cannot find spec.md in specs/025-fixes-for-testing/
Run this command from the repository root or specify --feature
```

### ERR-2: Malformed Requirements

**Given** requirement without proper ID format
**When** extracting
**Then** warn but continue with partial ID

```
Warning: Requirement missing ID format: "System must do something"
Assigned temporary ID: TEMP-001
```

### ERR-3: Circular References

**Given** tasks referencing each other
**When** analyzing mappings
**Then** detect and report cycles

## Integration Points

### With /analyze Command

- Uses same requirement extraction logic
- Reports complement analyze findings
- Can be run as part of analyze workflow

### With CI/CD

```yaml
# .github/workflows/task-validation.yml
- name: Validate Task Coverage
  run: |
    python scripts/validate_task_mapping.py --strict
  if: contains(github.event.pull_request.title, 'tasks.md')
```

### With Development Workflow

```bash
# Pre-task generation check
alias check-reqs='python scripts/validate_task_mapping.py'

# After tasks.md updates
alias validate-tasks='python scripts/validate_task_mapping.py --report'
```

## Contract Tests

```python
# tests/contract/test_task_mapping_contract.py

def test_MAP001_requirement_extraction():
    """Verify all requirement patterns are extracted."""
    spec_content = """
    **FR-001**: System MUST do something
    **FR-002**: System MUST do another thing
    - What happens when system fails?
    """
    requirements = extract_requirements(spec_content)
    assert len(requirements) == 3
    assert requirements[2]['type'] == 'edge_case'

def test_MAP001_task_reference_detection():
    """Verify task references are found."""
    task_desc = "Implement validation (FR-001, FR-002)"
    refs = extract_requirement_refs(task_desc)
    assert 'FR-001' in refs
    assert 'FR-002' in refs

def test_MAP001_gap_detection():
    """Verify gaps are correctly identified."""
    # Create mappings with gaps
    # Run gap detection
    # Assert correct gaps found

def test_MAP001_report_generation():
    """Verify report contains all sections."""
    # Generate report
    # Assert markdown structure
    # Assert actionable suggestions

def test_MAP001_edge_case_handling():
    """Verify edge cases are tracked."""
    # Extract edge cases
    # Check for test tasks
    # Assert gaps reported
```

## Performance Requirements

- Requirement extraction < 100ms for 1000-line spec
- Task analysis < 200ms for 500 tasks
- Full validation < 1 second for large features

## Security Considerations

- Read-only file operations
- No code execution
- Path traversal protection
- Sanitized report output

## Configuration Examples

### Strict Validation
```yaml
validation:
  require_explicit_references: true
  min_tasks_per_requirement: 1
  fail_on_missing: true
  fail_on_edge_gaps: true
```

### Relaxed Validation
```yaml
validation:
  allow_implicit_mapping: true
  fuzzy_matching: true
  warn_only: true
```

### Custom Patterns
```yaml
patterns:
  requirement_id: 'REQ\[(\d+)\]'  # Custom format
  task_reference: '#(\d+)'         # Issue-style refs
```

## Future Extensions

1. **Smart Mapping**: Use NLP to suggest task-requirement links
2. **Complexity Analysis**: Detect requirements needing multiple tasks
3. **Visual Reports**: Generate dependency graphs
4. **IDE Integration**: Real-time validation while editing