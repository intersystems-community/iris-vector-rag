# Research: Python Project Organization Best Practices

**Feature**: Root Directory Cleanup and Reorganization
**Research Phase**: Phase 0
**Date**: 2025-11-24

## Research Questions

### 1. What is the standard Python project structure per PEP guidelines?

**Decision**: Follow **PEP 518** (pyproject.toml) and **Python Packaging Authority (PyPA)** recommendations for modern Python project layout.

**Rationale**:
- PEP 518 standardizes `pyproject.toml` as the canonical project metadata file
- PyPA recommends a flat root directory with clear separation of concerns
- Modern tooling (uv, pip, build) expects this structure

**Standard Python Project Root Structure**:
```
project-root/
├── src/ or package_name/    # Source code (single top-level package)
├── tests/                   # All test files
├── docs/                    # Documentation
├── pyproject.toml           # Project metadata (PEP 518)
├── README.md                # Project overview
├── LICENSE                  # License file
├── CHANGELOG.md             # Version history
├── .gitignore               # VCS ignore patterns
├── Makefile (optional)      # Build automation
└── [<10 additional config files]
```

**Key Principles**:
1. **Flat is better than nested** - Avoid deep directory hierarchies at root
2. **One primary package** - Single source of truth for importable code
3. **Clear boundaries** - Tests separate from source, docs separate from both
4. **Minimal root clutter** - Aim for <20 root items

**References**:
- PEP 518: https://peps.python.org/pep-0518/
- PyPA Packaging Guide: https://packaging.python.org/en/latest/
- Python Application Layouts (Kenneth Reitz): https://docs.python-guide.org/writing/structure/

---

### 2. What items typically remain at project root vs. moved to subdirectories?

**Decision**: Keep only **essential project metadata and top-level configuration** at root. Move everything else to subdirectories.

**Items That MUST Stay at Root**:
```
✅ README.md              # First thing users see
✅ LICENSE                # Legal requirement
✅ CHANGELOG.md           # Version history (standard location)
✅ pyproject.toml         # PEP 518 standard
✅ setup.py (legacy)      # Only if supporting Python <3.11
✅ .gitignore             # VCS configuration
✅ .git/                  # VCS metadata (hidden)
✅ Makefile               # Build automation entry point
✅ docker-compose.yml     # Docker orchestration entry point
✅ Dockerfile             # Container build instructions
```

**Items That MUST Move to Subdirectories**:
```
❌ test_*.py              → tests/
❌ *.log                  → Remove or docs/logs/historical/
❌ *.md (except above)    → docs/
❌ scratch/               → Remove or docs/archive/
❌ examples/              → Keep at root OR move to docs/examples/
❌ scripts/               → Keep at root OR move to tools/
❌ outputs/               → Add to .gitignore (regenerate on demand)
❌ reports/               → Add to .gitignore (regenerate on demand)
❌ build/, dist/          → Add to .gitignore (build artifacts)
❌ *.egg-info/            → Add to .gitignore (build artifacts)
❌ .env (secrets)         → Add to .gitignore, keep .env.example
❌ *.key (secrets)        → Move to .gitignored config/ directory
```

**Rationale**:
- Root directory is the "entry point" for developers and tools
- Too many items cause "decision fatigue" when navigating
- Clear organization signals project maturity and maintainability

**Benchmark**: Well-maintained Python projects (requests, flask, fastapi) have 15-25 root items.

---

### 3. What are best practices for `.gitignore` organization and commenting?

**Decision**: Use **hierarchical sections with comments** explaining each category. Follow GitHub's Python .gitignore template as baseline.

**Recommended Structure**:
```gitignore
# ==============================================================================
# Python
# ==============================================================================
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/
.eggs/

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage_reports/

# ==============================================================================
# Environment & Configuration
# ==============================================================================
# Virtual environments
.venv/
venv/
ENV/

# Environment variables (keep .env.example)
.env
.env.local
.env.*.local

# Secrets (keep examples)
*.key
!*.key.example
config/*.key

# ==============================================================================
# Project-Specific
# ==============================================================================
# Output directories (regenerate on demand)
outputs/
reports/
validation_results/
test_results/

# Logs (keep historical in docs/logs/)
*.log

# ==============================================================================
# IDE / Editor
# ==============================================================================
.vscode/
.idea/
*.swp
*.swo
*~

# ==============================================================================
# OS
# ==============================================================================
.DS_Store
Thumbs.db
```

**Key Principles**:
1. **Comments as section headers** - Group related patterns
2. **Explain the "why"** - E.g., "(keep .env.example)" clarifies intent
3. **Negative patterns for exceptions** - `!*.key.example` keeps examples
4. **Order matters** - More specific patterns override general ones
5. **Trailing slashes for directories** - `outputs/` ignores directory, not file

**Rationale**:
- Future developers understand WHY files are ignored
- Easier to maintain as project evolves
- Self-documenting - no separate "gitignore guide" needed

**Reference**: GitHub's Python gitignore template: https://github.com/github/gitignore/blob/main/Python.gitignore

---

### 4. How to safely verify import statements before removing legacy package directories?

**Decision**: Use **multi-stage verification** with grep + AST parsing + test execution.

**Safe Verification Strategy**:

**Stage 1: Text Search (Fast)**
```bash
# Search for all import statements referencing legacy packages
grep -r "from iris_rag" --include="*.py" . > imports_iris_rag.txt
grep -r "from rag_templates" --include="*.py" . > imports_rag_templates.txt
grep -r "from common" --include="*.py" . > imports_common.txt
grep -r "import iris_rag" --include="*.py" . >> imports_iris_rag.txt
grep -r "import rag_templates" --include="*.py" . >> imports_rag_templates.txt
grep -r "import common" --include="*.py" . >> imports_common.txt
```

**Stage 2: AST Parsing (Accurate)**
```python
# check_imports.py
import ast
import pathlib

def find_imports(file_path, target_packages):
    """Parse Python file and find imports of target packages."""
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(pkg) for pkg in target_packages):
                        imports.append((file_path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(pkg) for pkg in target_packages):
                    imports.append((file_path, node.lineno, node.module))

        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

# Search all Python files
target_packages = ["iris_rag", "rag_templates", "common"]
all_imports = []

for py_file in pathlib.Path(".").rglob("*.py"):
    if ".venv" not in str(py_file) and "build" not in str(py_file):
        all_imports.extend(find_imports(py_file, target_packages))

if all_imports:
    print(f"⚠️  Found {len(all_imports)} imports of legacy packages:")
    for file, line, module in all_imports:
        print(f"  {file}:{line} → {module}")
    print("\n❌ CANNOT remove legacy packages - imports still exist")
else:
    print("✅ No imports found - safe to remove legacy packages")
```

**Stage 3: Test Execution (Final Validation)**
```bash
# Run full test suite to catch dynamic imports
pytest tests/ -v

# If tests pass, verify imports are truly unused
# (tests might not cover all code paths)
```

**Decision Matrix**:
| grep output | AST finds imports | Tests pass | Action |
|-------------|-------------------|------------|--------|
| Empty | None | ✅ Pass | ✅ Safe to delete |
| Non-empty | Found imports | ✅ Pass | ⚠️ Keep with deprecation warning (per FR-010) |
| Non-empty | Found imports | ❌ Fail | ❌ DO NOT delete - fix imports first |
| Empty | Found imports | ❌ Fail | ❌ DO NOT delete - AST is authoritative |

**Rationale**:
- **grep** catches 95% of cases quickly
- **AST parsing** is 100% accurate (handles comments, strings, etc.)
- **Test execution** catches runtime import errors
- **Fail-safe approach**: When in doubt, keep with deprecation warning

**Reference**: Python AST documentation: https://docs.python.org/3/library/ast.html

---

### 5. What is the recommended approach for consolidating multiple `.env` files?

**Decision**: Use **single `.env` + `.env.example` pattern** with clear documentation of variable purposes.

**Consolidation Strategy**:

**Step 1: Audit existing `.env` files**
```bash
# Find all .env files
find . -name ".env*" -type f | grep -v ".env.example"
# Typical findings: .env, .env.local, .env.development, .env.test
```

**Step 2: Merge with precedence rules**
```bash
# Merge strategy (highest to lowest precedence):
# 1. .env.local (developer-specific overrides)
# 2. .env.development (development defaults)
# 3. .env (base configuration)
# 4. .env.example (template only)

# Keep ONLY:
# - .env (gitignored, merged content)
# - .env.example (tracked, template with no real values)
```

**Step 3: Create consolidated `.env`**
```bash
# .env (gitignored - contains real values)
# ==============================================================================
# IRIS Database Configuration
# ==============================================================================
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USER=_SYSTEM
IRIS_PASSWORD=SYS

# ==============================================================================
# API Keys (DO NOT COMMIT)
# ==============================================================================
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# ==============================================================================
# Feature Flags
# ==============================================================================
ENABLE_MONITORING=false
ENABLE_GRAPHRAG=true

# ==============================================================================
# Performance Tuning
# ==============================================================================
MAX_CONNECTIONS=10
QUERY_TIMEOUT_SECONDS=30
```

**Step 4: Create template `.env.example`**
```bash
# .env.example (tracked in git - no real secrets)
# ==============================================================================
# IRIS Database Configuration
# ==============================================================================
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USER=_SYSTEM
IRIS_PASSWORD=SYS

# ==============================================================================
# API Keys (obtain from respective providers)
# ==============================================================================
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# ==============================================================================
# Feature Flags
# ==============================================================================
ENABLE_MONITORING=false
ENABLE_GRAPHRAG=true

# ==============================================================================
# Performance Tuning
# ==============================================================================
MAX_CONNECTIONS=10
QUERY_TIMEOUT_SECONDS=30
```

**Step 5: Update `.gitignore`**
```gitignore
# Environment variables (keep .env.example)
.env
.env.local
.env.*.local
*.env
!.env.example
```

**Step 6: Update README setup instructions**
```markdown
## Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   - `OPENAI_API_KEY`: Get from https://platform.openai.com/api-keys
   - `ANTHROPIC_API_KEY`: Get from https://console.anthropic.com/

3. Configure IRIS connection parameters if using non-default values
```

**Rationale**:
- **Single source of truth** - One `.env` file eliminates confusion
- **No secrets in git** - `.env` gitignored, `.env.example` is template
- **Self-documenting** - Comments explain purpose of each variable
- **Easy onboarding** - `cp .env.example .env` is industry standard

**Alternative Rejected**: Environment-specific files (`.env.development`, `.env.production`) create complexity. Use feature flags or deployment config instead.

**Reference**: The Twelve-Factor App - Config: https://12factor.net/config

---

## Summary of Decisions

| Research Question | Decision | Implementation Impact |
|-------------------|----------|----------------------|
| **Python project structure** | Follow PEP 518 + PyPA guidelines | Use `pyproject.toml`, maintain <30 root items |
| **Root vs. subdirectories** | Keep only essential metadata at root | Move tests, docs, logs to subdirectories |
| **`.gitignore` organization** | Hierarchical sections with comments | Update `.gitignore` with clear structure |
| **Import verification** | Multi-stage (grep → AST → tests) | Create `check_imports.py` script |
| **`.env` consolidation** | Single `.env` + `.env.example` | Merge existing .env files, update gitignore |

## Implementation Checklist

- [ ] Create `check_imports.py` script for Stage 2 verification
- [ ] Audit existing `.env` files and create consolidation plan
- [ ] Draft new `.gitignore` with hierarchical structure
- [ ] Define root directory "keep list" (15-25 items)
- [ ] Document move operations in migration script

**Next Phase**: Phase 1 - No design artifacts needed (organizational feature). Proceed directly to `/speckit.tasks` for task breakdown.
