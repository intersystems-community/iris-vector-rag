# Quick Start: Documentation Review and README Optimization

**Feature**: 055-perform-top-to
**Purpose**: Validate and update documentation for accuracy, completeness, and professional quality

## Prerequisites

- Python 3.11+ installed
- Repository cloned locally
- No additional dependencies (uses standard library + pytest + requests)

## Quick Validation (30 seconds)

Run automated validation checks to identify issues:

```bash
# 1. Check README line count (must be ≤400)
wc -l README.md

# 2. Run all contract tests
pytest specs/055-perform-top-to/contracts/ -v

# 3. Quick link check (README only)
python -c "
import re
from pathlib import Path

readme = Path('README.md')
content = readme.read_text()
links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
broken = []
for text, url in links:
    if not url.startswith('http') and not url.startswith('#'):
        target = Path(url.split('#')[0])
        if not target.exists():
            broken.append(url)
if broken:
    print(f'❌ Found {len(broken)} broken internal links:')
    for link in broken:
        print(f'  - {link}')
else:
    print('✅ All internal links OK')
"

# 4. Check for old module names
grep -n "from iris_rag import\|import iris_rag" README.md || echo "✅ No old module names found"
```

**Expected Output** (after implementation):
```
README.md: ≤400 lines ✅
Contract tests: 4/4 passing ✅
Internal links: 0 broken ✅
Module names: All use iris_vector_rag ✅
```

## Full Validation (5 minutes)

Run comprehensive validation across all documentation:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install validation dependencies (if not already installed)
pip install requests pytest

# 3. Run full contract test suite
pytest specs/055-perform-top-to/contracts/ -v --tb=short

# 4. Check all documentation files for old module names
find docs/ -name "*.md" -exec grep -Hn "from iris_rag import\|import iris_rag" {} \;

# 5. Validate documentation index exists
test -f docs/README.md && echo "✅ Documentation index exists" || echo "❌ Missing docs/README.md"

# 6. Check for obsolete documents
ls docs/archived/ 2>/dev/null && echo "✅ Archived docs exist" || echo "⚠️  No archived docs (may be OK)"

# 7. Verify new detailed guides created
test -f docs/IRIS_EMBEDDING_GUIDE.md && echo "✅ IRIS EMBEDDING guide created" || echo "❌ Missing IRIS_EMBEDDING_GUIDE.md"
test -f docs/PIPELINE_GUIDE.md && echo "✅ Pipeline guide created" || echo "❌ Missing PIPELINE_GUIDE.md"
```

**Expected Results** (after implementation):
- All contract tests PASS
- Zero occurrences of old module name across all docs
- Documentation index (docs/README.md) exists
- New detailed guides created
- Obsolete docs archived (if any)

## Manual Verification (10 minutes)

Some aspects require human review:

### 1. README Readability
```bash
# Open README in browser or editor
open README.md  # macOS
xdg-open README.md  # Linux
```

**Check**:
- [ ] First paragraph clearly explains what iris-vector-rag does
- [ ] Can understand available pipelines from table
- [ ] Quick start example is copy-pasteable
- [ ] Professional tone throughout
- [ ] Detailed sections link to separate guides (not inline)

### 2. Documentation Navigation
```bash
# Open documentation index
open docs/README.md
```

**Check**:
- [ ] Clear categories (Getting Started, Advanced, Contributing, Architecture)
- [ ] Each document has brief description
- [ ] Progression from beginner to advanced is logical
- [ ] No duplicate information across docs

### 3. Code Example Execution
```bash
# Test that quick start example actually works
python -c "
from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document

# Create pipeline
pipeline = create_pipeline('basic', validate_requirements=False)

# Load sample documents
docs = [
    Document(
        page_content='RAG combines retrieval with generation.',
        metadata={'source': 'test.pdf', 'page': 1}
    )
]

pipeline.load_documents(documents=docs)
print('✅ Quick start example executes successfully')
"
```

**Expected**: No errors, confirmation message printed

### 4. Performance Claims Validation
```bash
# Check if performance benchmarks are documented
grep -n "346x\|50-100x\|speedup" README.md
```

**Check**:
- [ ] Each performance claim cites source or test conditions
- [ ] No unsubstantiated claims
- [ ] Benchmarks include test methodology if present

## Success Criteria

After running all validations, the following must be true:

**Automated Checks** (pytest):
- ✅ All links resolve (HTTP 200 for external, files exist for internal)
- ✅ All code examples have valid syntax
- ✅ All code examples use `iris_vector_rag` module name
- ✅ README line count ≤ 400
- ✅ README has clear structure with essential sections

**Manual Checks**:
- ✅ README is professional and scannable
- ✅ Quick start example executes successfully
- ✅ Documentation index provides clear navigation
- ✅ No duplicate content across documentation files
- ✅ Obsolete documents archived

## Troubleshooting

### Contract tests fail with import errors
```bash
# Ensure you're in the repository root
cd /path/to/iris-vector-rag-private

# Ensure pytest is installed
pip install pytest requests
```

### README still exceeds 400 lines
```bash
# Check current line count
wc -l README.md

# Identify sections that could be moved to guides
grep "^## " README.md
```

**Common culprits**:
- IRIS EMBEDDING section (should be in IRIS_EMBEDDING_GUIDE.md)
- MCP section details (should be in MCP_INTEGRATION.md)
- Architecture details (should link to COMPREHENSIVE_ARCHITECTURE_OVERVIEW.md)

### Broken links detected
```bash
# List all broken links
pytest specs/055-perform-top-to/contracts/link_validation_contract.py -v

# Fix by either:
# 1. Creating missing target file
# 2. Updating link to correct target
# 3. Removing link if target no longer relevant
```

### Old module names still present
```bash
# Find all occurrences
grep -rn "from iris_rag import\|import iris_rag" . --include="*.md"

# Replace with correct module name
sed -i '' 's/from iris_rag import/from iris_vector_rag import/g' README.md
sed -i '' 's/import iris_rag/import iris_vector_rag/g' README.md
```

## Next Steps

After all validations pass:

1. **Commit changes**:
   ```bash
   git add README.md docs/
   git commit -m "docs: optimize README and documentation structure

   - Reduced README from 518 to <400 lines
   - Fixed all old module name references (iris_rag → iris_vector_rag)
   - Created documentation index (docs/README.md)
   - Moved detailed content to separate guides
   - Validated all links and code examples

   All contract tests passing."
   ```

2. **Create pull request**:
   ```bash
   git push origin 055-perform-top-to
   # Then create PR on GitHub
   ```

3. **Update constitution** (if needed):
   - If new documentation practices identified
   - If validation approach should be standardized for future docs

## Validation Scripts

For reference, the contract tests are located at:
- `specs/055-perform-top-to/contracts/link_validation_contract.py`
- `specs/055-perform-top-to/contracts/code_example_contract.py`
- `specs/055-perform-top-to/contracts/readme_structure_contract.py`

Run them individually:
```bash
pytest specs/055-perform-top-to/contracts/link_validation_contract.py -v
pytest specs/055-perform-top-to/contracts/code_example_contract.py -v
pytest specs/055-perform-top-to/contracts/readme_structure_contract.py -v
```

Or all together:
```bash
pytest specs/055-perform-top-to/contracts/ -v
```
