# Quickstart: Repository Cleanup Execution

**Feature**: Repository Cleanup and Organization
**Date**: 2025-10-08
**Phase**: Phase 1 - Design

## Overview

This quickstart guide provides step-by-step instructions for executing the repository cleanup following the planned approach. Each step includes validation to ensure no functionality is broken.

## Prerequisites

- Git repository with clean working directory
- All tests passing (136/136)
- Python 3.11+ environment activated
- pytest installed

## Execution Steps

### Step 1: Create Backup Branch

Create safety backup before making changes.

```bash
# Ensure working directory is clean
git status

# Create backup branch
git checkout -b 037-clean-up-this-backup
git checkout 037-clean-up-this

# Verify on correct branch
git branch --show-current
# Expected output: 037-clean-up-this
```

**Validation**: `git log` shows same commit as main branch.

---

### Step 2: Establish Test Baseline

Run full test suite and record baseline.

```bash
# Run all tests
pytest --tb=short -v > baseline_tests.txt 2>&1

# Extract test count
grep "passed" baseline_tests.txt | tail -1

# Expected output: 136 passed
```

**Validation**: All 136 tests pass, no failures.

**Save baseline**:
```bash
baseline_count=136
echo "Baseline: $baseline_count tests passed" > cleanup_log.txt
```

---

### Step 3: Scan and Classify Files

Identify all files and classify them.

```bash
# List all top-level files
ls -la | grep "^-" > top_level_files.txt

# Count files
wc -l top_level_files.txt
```

**Manual Classification** (or use automated script):

**Essential** (keep at root):
- README.md
- USER_GUIDE.md
- CLAUDE.md
- DOCUMENTATION_INDEX.md
- DOCUMENTATION_AUDIT_REPORT.md
- docker-compose*.yml
- Makefile
- pyproject.toml
- pytest.ini
- .gitignore
- LICENSE
- .env.example

**Relocatable** (move to docs/):
- docs/docs/STATUS.md
- docs/docs/PROGRESS.md
- docs/docs/TODO.md
- docs/docs/docs/CHANGELOG.md

**Removable**:
- MORNING_BRIEFING.md (historical)
- outputs/pipeline_verification_*.json (old outputs)
- outputs/reports/ragas_evaluations/*.html (old reports)
- outputs/reports/ragas_evaluations/*.json (old reports)
- Any *_COMPLETION_SUMMARY.md
- Any *_SESSION_NOTES.md

**Validation**: All top-level files categorized.

---

### Step 4: BATCH 1 - Remove Temporary/Cache Files

Remove Python cache and temp files.

```bash
# Find and remove cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache/
rm -f .coverage

# Stage changes
git add -A

# Validate tests
pytest --tb=short > current_tests.txt 2>&1
current_count=$(grep "passed" current_tests.txt | tail -1 | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    echo "ERROR: Test count mismatch. Rolling back."
    git restore --staged .
    git restore .
    exit 1
fi

echo "BATCH 1: PASS - Cache files removed" >> cleanup_log.txt
git commit -m "chore: remove Python cache and temporary files"
```

**Validation**: 136 tests still pass.

---

### Step 5: BATCH 2 - Remove Old Evaluation Reports

Remove old RAGAS reports and pipeline verification outputs.

```bash
# Remove old evaluation reports
rm -f outputs/pipeline_verification_*.json
rm -f outputs/reports/ragas_evaluations/*.html
rm -f outputs/reports/ragas_evaluations/*.json

# Stage changes
git add -A

# Validate tests
pytest --tb=short > current_tests.txt 2>&1
current_count=$(grep "passed" current_tests.txt | tail -1 | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    echo "ERROR: Test count mismatch. Rolling back."
    git restore --staged .
    git restore .
    exit 1
fi

echo "BATCH 2: PASS - Old reports removed" >> cleanup_log.txt
git commit -m "chore: remove old evaluation reports and outputs

Removed regeneratable files:
- pipeline_verification outputs
- RAGAS evaluation reports (HTML/JSON)

All test validation passed."
```

**Validation**: 136 tests still pass.

---

### Step 6: BATCH 3 - Remove Historical Tracking Files

Remove old session notes and completion summaries.

```bash
# Remove historical tracking (if any exist)
find . -name "*_COMPLETION_SUMMARY.md" -type f -delete
find . -name "*_SESSION_NOTES.md" -type f -delete
rm -f MORNING_BRIEFING.md

# Stage changes
git add -A

# Validate tests
pytest --tb=short > current_tests.txt 2>&1
current_count=$(grep "passed" current_tests.txt | tail -1 | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    echo "ERROR: Test count mismatch. Rolling back."
    git restore --staged .
    git restore .
    exit 1
fi

echo "BATCH 3: PASS - Historical files removed" >> cleanup_log.txt
git commit -m "chore: remove historical tracking files

Removed outdated status tracking:
- MORNING_BRIEFING.md
- Other historical session notes and completion summaries

All test validation passed."
```

**Validation**: 136 tests still pass.

---

### Step 7: BATCH 4 - Consolidate Duplicate Documentation

Find and remove duplicate documentation files (keeping newest).

```bash
# Manually identify duplicates
# Example: If multiple SUMMARY.md files exist

# For each duplicate group:
# ls -lt SUMMARY*.md  # List by modification time
# Keep the first (newest), remove others

# Example (if duplicates found):
# rm SUMMARY_old.md
# git add SUMMARY_old.md

# Validate tests
pytest --tb=short > current_tests.txt 2>&1
current_count=$(grep "passed" current_tests.txt | tail -1 | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    echo "ERROR: Test count mismatch. Rolling back."
    git restore --staged .
    git restore .
    exit 1
fi

echo "BATCH 4: PASS - Duplicates consolidated" >> cleanup_log.txt
# git commit -m "chore: consolidate duplicate documentation" (if changes made)
```

**Validation**: 136 tests still pass, newest versions kept.

---

### Step 8: BATCH 5 - Move Status Files to docs/

Move current status tracking files to docs directory.

```bash
# Create docs directory if doesn't exist
mkdir -p docs

# Move status files
git mv docs/docs/STATUS.md docs/docs/docs/STATUS.md
git mv docs/docs/PROGRESS.md docs/docs/docs/PROGRESS.md
git mv docs/docs/TODO.md docs/docs/docs/TODO.md
git mv docs/docs/docs/CHANGELOG.md docs/docs/docs/docs/CHANGELOG.md

# Validate tests
pytest --tb=short > current_tests.txt 2>&1
current_count=$(grep "passed" current_tests.txt | tail -1 | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    echo "ERROR: Test count mismatch. Rolling back."
    git restore --staged .
    git restore .
    exit 1
fi

echo "BATCH 5: PASS - Status files moved to docs/" >> cleanup_log.txt
git commit -m "chore: move status tracking files to docs directory

Moved to docs/:
- docs/docs/STATUS.md
- docs/docs/PROGRESS.md
- docs/docs/TODO.md
- docs/docs/docs/CHANGELOG.md

All test validation passed."
```

**Validation**: 136 tests still pass, files exist in docs/.

---

### Step 9: Update Documentation Links

Update any documentation that references moved files.

```bash
# Check for references to moved files
grep -r "docs/docs/STATUS.md" --include="*.md" . | grep -v "docs/docs/docs/STATUS.md"
grep -r "docs/docs/PROGRESS.md" --include="*.md" . | grep -v "docs/docs/docs/PROGRESS.md"
grep -r "docs/docs/TODO.md" --include="*.md" . | grep -v "docs/docs/docs/TODO.md"
grep -r "docs/docs/docs/CHANGELOG.md" --include="*.md" . | grep -v "docs/docs/docs/docs/CHANGELOG.md"

# Manually update any found references
# For example, update README.md links if needed

# Validate tests
pytest --tb=short > current_tests.txt 2>&1
current_count=$(grep "passed" current_tests.txt | tail -1 | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    echo "ERROR: Test count mismatch. Rolling back."
    git restore --staged .
    git restore .
    exit 1
fi

echo "BATCH 6: PASS - Documentation links updated" >> cleanup_log.txt
# git commit -m "docs: update links to moved status files" (if changes made)
```

**Validation**: No broken links, 136 tests still pass.

---

### Step 10: Update DOCUMENTATION_INDEX.md

Update documentation index with new file locations.

```bash
# Edit DOCUMENTATION_INDEX.md
# Update links for moved files:
# - [docs/docs/STATUS.md](docs/docs/STATUS.md) → [docs/docs/STATUS.md](docs/docs/docs/STATUS.md)
# - [docs/docs/PROGRESS.md](docs/docs/PROGRESS.md) → [docs/docs/PROGRESS.md](docs/docs/docs/PROGRESS.md)
# - [docs/docs/TODO.md](docs/docs/TODO.md) → [docs/docs/docs/TODO.md](docs/docs/docs/TODO.md)
# - [docs/docs/docs/CHANGELOG.md](docs/docs/docs/CHANGELOG.md) → [docs/docs/docs/CHANGELOG.md](docs/docs/docs/docs/CHANGELOG.md)

# Stage changes
git add DOCUMENTATION_INDEX.md

# Validate tests
pytest --tb=short > current_tests.txt 2>&1
current_count=$(grep "passed" current_tests.txt | tail -1 | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    echo "ERROR: Test count mismatch. Rolling back."
    git restore --staged .
    git restore .
    exit 1
fi

echo "BATCH 7: PASS - DOCUMENTATION_INDEX.md updated" >> cleanup_log.txt
git commit -m "docs: update DOCUMENTATION_INDEX with new file locations

Updated links to reflect moved status files in docs/ directory.

All test validation passed."
```

**Validation**: All links in index are valid, 136 tests pass.

---

### Step 11: Final Validation

Run comprehensive final validation.

```bash
# Run full test suite one more time
pytest --tb=short -v

# Expected: 136 passed

# Check git status
git status

# Expected: clean working directory

# Review cleanup log
cat cleanup_log.txt

# Verify file count reduction
echo "Top-level files before (reference): ~20-25"
ls -la | grep "^-" | wc -l
echo "Top-level files after (goal): ~15-20"
```

**Success Criteria**:
- ✓ All 136 tests pass
- ✓ Git working directory clean
- ✓ Status files in docs/
- ✓ No old evaluation reports
- ✓ No historical tracking files
- ✓ Documentation links valid
- ✓ Top-level directory cleaner

---

### Step 12: Push Changes

Push cleanup branch to remote.

```bash
# Review all commits
git log --oneline origin/037-clean-up-this..HEAD

# Push to remote
git push origin 037-clean-up-this

# Create pull request (if needed)
# gh pr create --title "Repository cleanup and organization" --body "..."
```

---

## Rollback Procedure

If any batch fails validation:

```bash
# Rollback staged and working directory changes
git restore --staged .
git restore .

# Verify rollback
git status  # Should show clean working directory
pytest      # Should show 136 passed

# Investigate failure
cat current_tests.txt  # Review test output
```

---

## Expected Results

**Before Cleanup**:
- Top-level files: ~20-25 files
- outputs/ directory: Contains old RAGAS reports
- Status files at root: docs/docs/STATUS.md, docs/docs/PROGRESS.md, docs/docs/TODO.md, docs/docs/docs/CHANGELOG.md
- Historical files: MORNING_BRIEFING.md

**After Cleanup**:
- Top-level files: ~15-20 files (only essential)
- outputs/ directory: Empty or only current outputs
- Status files: Moved to docs/
- Historical files: Removed
- All 136 tests: PASSING ✓

**Git History**:
- 6-8 commits documenting each cleanup batch
- Reversible via git revert if needed

---

## Troubleshooting

**Test failures after cleanup**:
1. Check which tests failed: `pytest --tb=short -v | grep FAILED`
2. Identify removed/moved file causing failure
3. Rollback that specific batch
4. Reclassify file as ESSENTIAL
5. Re-run cleanup

**Broken documentation links**:
1. Run: `grep -r "](.*\.md)" --include="*.md" .`
2. Manually verify each link resolves
3. Update links to new locations

**Missing files after cleanup**:
1. Check git log for deletion: `git log --all --full-history -- path/to/file`
2. Restore if needed: `git checkout <commit> -- path/to/file`
3. Reclassify as ESSENTIAL or RELOCATABLE

---

**Quickstart Complete**: 2025-10-08
**Ready for Implementation**: Phase 2 (/tasks command)
