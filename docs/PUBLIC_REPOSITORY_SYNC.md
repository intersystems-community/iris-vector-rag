# Public Repository Sync & Redaction Guide

**Last Updated**: 2025-10-14

This guide describes the process for syncing the internal repository to the public-facing GitHub repository with automated redaction of internal references.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Redaction Rules](#redaction-rules)
- [Scripts](#scripts)
- [Workflow](#workflow)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Overview

The RAG-Templates project is developed internally at InterSystems with references to internal infrastructure (GitLab, Docker registry, etc.). Before publishing to the public GitHub repository, all internal references must be redacted and replaced with public equivalents.

### What Gets Redacted

- **Internal GitLab URLs** → Public GitHub URLs
- **Internal Docker Registry** → Public Docker Hub
- **Internal Email Addresses** → Generic maintainer emails
- **Internal Merge Request References** → Pull Request references
- **Internal User Paths** → Community organization paths

## Quick Start

### Prerequisites

1. **Configure Public Remote** (first time only):
```bash
git remote add public git@github.com:intersystems-community/iris-rag-templates.git

# Verify configuration
git remote -v
```

2. **Install Python Dependencies**:
```bash
# Already included in requirements.txt
python -m pip install --upgrade pip
```

### Option 1: Automated Sync (Recommended)

```bash
# 1. Preview changes (dry run)
./scripts/sync_to_public.sh --dry-run

# 2. Apply redaction and push to public repository
./scripts/sync_to_public.sh --branch main
```

### Option 2: Manual Redaction + Push

```bash
# 1. Preview redaction changes
python scripts/redact_for_public.py --dry-run --verbose

# 2. Apply redaction
python scripts/redact_for_public.py --backup

# 3. Review changes
git diff

# 4. Commit redacted changes
git add -A
git commit -m "chore: redact internal references for public release"

# 5. Push to public repository
git push public main
```

## Redaction Rules

The following redaction rules are automatically applied:

| Internal Reference | Public Replacement | Count |
|-------------------|-------------------|-------|
| `gitlab.iscinternal.com` | `github.com/intersystems-community` | ~30 |
| `https://gitlab.iscinternal.com/tdyar/rag-templates` | `https://github.com/intersystems-community/iris-rag-templates` | ~15 |
| `git@gitlab.iscinternal.com:tdyar/rag-templates.git` | `git@github.com:intersystems-community/iris-rag-templates.git` | ~5 |
| `docker.iscinternal.com/intersystems/iris` | `intersystemsdc/iris-community` | ~10 |
| `merge request` / `Merge Request` | `pull request` / `Pull Request` | ~20 |
| `MR #` | `PR #` | ~5 |
| `tdyar@intersystems.com` | `maintainer@example.com` | ~3 |
| `/tdyar/` | `/intersystems-community/` | ~500 |

**Total**: ~614 replacements across ~92 files

## Scripts

### 1. `sync_to_public.sh`

Automated sync script that handles redaction and pushing to public repository.

**Features**:
- Dry-run mode for preview
- Automatic remote validation
- Comprehensive redaction
- Git integration

**Usage**:
```bash
# Show help
./scripts/sync_to_public.sh --help

# Preview changes
./scripts/sync_to_public.sh --dry-run

# Sync specific branch
./scripts/sync_to_public.sh --branch feature/new-feature

# Full sync to main
./scripts/sync_to_public.sh --branch main
```

### 2. `redact_for_public.py`

Python script for detailed redaction with logging.

**Features**:
- Pattern-based redaction
- Detailed change logging (JSON)
- Backup creation
- Verbose output

**Usage**:
```bash
# Show help
python scripts/redact_for_public.py --help

# Dry run with verbose output
python scripts/redact_for_public.py --dry-run --verbose

# Apply redaction with backup
python scripts/redact_for_public.py --backup

# Custom backup location
python scripts/redact_for_public.py --backup --backup-dir /tmp/my-backup

# Save detailed log
python scripts/redact_for_public.py --log-file redaction-$(date +%Y%m%d).json
```

## Workflow

### Standard Release Workflow

1. **Develop Feature on Internal Repository**:
   ```bash
   git checkout -b feature/my-feature
   # ... development work ...
   git commit -m "feat: implement my feature"
   git push origin feature/my-feature
   ```

2. **Merge to Main (Internal)**:
   ```bash
   # Create merge request on GitLab
   # Review and merge
   ```

3. **Sync to Public Repository**:
   ```bash
   # Checkout main branch
   git checkout main
   git pull origin main

   # Preview redaction
   python scripts/redact_for_public.py --dry-run --verbose

   # Apply redaction and sync
   ./scripts/sync_to_public.sh --branch main
   ```

4. **Verify Public Repository**:
   ```bash
   # Clone public repo to verify
   git clone git@github.com:intersystems-community/iris-rag-templates.git /tmp/public-verify
   cd /tmp/public-verify

   # Check for any remaining internal references
   grep -r "iscinternal" . --exclude-dir=.git || echo "✅ No internal references found"
   grep -r "tdyar@" . --exclude-dir=.git || echo "✅ No internal emails found"
   ```

### Feature Branch Workflow

For feature branches that need to be public immediately:

```bash
# 1. Develop feature
git checkout -b feature/public-feature
# ... work ...

# 2. Sync feature branch to public
./scripts/sync_to_public.sh --branch feature/public-feature

# 3. Create pull request on GitHub
# Navigate to: https://github.com/intersystems-community/iris-rag-templates/pulls
```

## Verification

### Pre-Sync Verification

Before syncing, verify your changes don't contain sensitive information:

```bash
# Check for API keys or secrets
git diff main | grep -i "api_key\|secret\|password\|token" || echo "✅ No secrets found"

# Check for internal references in changed files
git diff main --name-only | xargs grep "iscinternal" || echo "✅ No internal refs in changes"
```

### Post-Sync Verification

After syncing to public repository:

```bash
# 1. Clone public repo
git clone git@github.com:intersystems-community/iris-rag-templates.git /tmp/public-check
cd /tmp/public-check

# 2. Run comprehensive checks
# Check for internal URLs
grep -r "iscinternal" . --exclude-dir=.git || echo "✅ No internal URLs"

# Check for internal Docker registry
grep -r "docker.iscinternal" . --exclude-dir=.git || echo "✅ No internal registry"

# Check for internal emails
grep -r "@intersystems.com" . --exclude-dir=.git || echo "✅ No internal emails"

# Check for internal paths
grep -r "/tdyar/" . --exclude-dir=.git || echo "✅ No internal paths"

# 3. Test build and basic functionality
make install
make test
```

### Automated Verification

Use the verification script:

```bash
# Create verification script
cat > scripts/verify_public_repo.sh << 'EOF'
#!/bin/bash
set -e

REPO_PATH="${1:-.}"

echo "Verifying public repository at: $REPO_PATH"

ISSUES=0

# Check for internal references
if grep -r "iscinternal" "$REPO_PATH" --exclude-dir=.git; then
    echo "❌ Found internal URLs"
    ((ISSUES++))
fi

if grep -r "tdyar@intersystems.com" "$REPO_PATH" --exclude-dir=.git; then
    echo "❌ Found internal emails"
    ((ISSUES++))
fi

if grep -r "docker.iscinternal" "$REPO_PATH" --exclude-dir=.git; then
    echo "❌ Found internal Docker registry"
    ((ISSUES++))
fi

if [[ $ISSUES -eq 0 ]]; then
    echo "✅ All verification checks passed"
    exit 0
else
    echo "❌ Found $ISSUES issues"
    exit 1
fi
EOF

chmod +x scripts/verify_public_repo.sh

# Run verification
./scripts/verify_public_repo.sh
```

## Troubleshooting

### Problem 1: Public Remote Not Configured

**Error**:
```
❌ Error: Public remote 'public' not configured
```

**Solution**:
```bash
# Add public remote
git remote add public git@github.com:intersystems-community/iris-rag-templates.git

# Verify
git remote -v
```

### Problem 2: Permission Denied (Public Repository)

**Error**:
```
Permission denied (publickey)
```

**Solution**:
```bash
# 1. Check SSH key is added to GitHub
ssh -T git@github.com

# 2. Add SSH key if needed
# Generate new key:
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub:
cat ~/.ssh/id_ed25519.pub
# Copy output and add to GitHub: Settings → SSH Keys
```

### Problem 3: Redaction Script Finds New Patterns

**Symptom**: Script reports unexpected internal references

**Solution**:
```bash
# 1. Review detailed log
python scripts/redact_for_public.py --dry-run --verbose

# 2. Update redaction rules in redact_for_public.py
# Edit the `redaction_rules` dictionary:
self.redaction_rules = {
    # ... existing rules ...
    'new-internal-pattern': 'public-replacement',
}

# 3. Re-run redaction
python scripts/redact_for_public.py --dry-run
```

### Problem 4: Merge Conflicts on Public Repository

**Error**:
```
! [rejected] main -> main (non-fast-forward)
```

**Solution**:
```bash
# 1. Fetch latest from public
git fetch public main

# 2. Rebase on public main
git rebase public/main

# 3. Resolve conflicts if any
# ... conflict resolution ...

# 4. Force push (with caution)
git push public main --force-with-lease
```

### Problem 5: Incomplete Redaction

**Symptom**: Internal references still visible after sync

**Solution**:
```bash
# 1. Run verification script
./scripts/verify_public_repo.sh

# 2. If issues found, update redaction rules
# Edit scripts/redact_for_public.py

# 3. Re-run redaction
python scripts/redact_for_public.py

# 4. Force update public repository
git push public main --force-with-lease
```

## Best Practices

### 1. Always Use Dry-Run First

```bash
# Preview changes before applying
python scripts/redact_for_public.py --dry-run --verbose
./scripts/sync_to_public.sh --dry-run
```

### 2. Create Backups

```bash
# Always create backup before redaction
python scripts/redact_for_public.py --backup --backup-dir /tmp/backup-$(date +%Y%m%d)
```

### 3. Review Detailed Logs

```bash
# Save and review redaction log
python scripts/redact_for_public.py --log-file redaction-$(date +%Y%m%d).json

# Review the log
cat redaction-*.json | python -m json.tool | less
```

### 4. Verify Public Repository

```bash
# Always verify after sync
./scripts/verify_public_repo.sh /tmp/public-clone
```

### 5. Keep Redaction Rules Updated

Regularly review and update redaction rules:
- New internal references
- Changed URLs or paths
- New team members' emails
- Internal service changes

## Security Considerations

### Secrets and API Keys

**IMPORTANT**: The redaction scripts only handle URL and reference redaction. They do NOT scan for:
- API keys
- Passwords
- Tokens
- Credentials

**Before syncing**:
```bash
# Manual secret check
git diff main | grep -iE "(api_key|password|secret|token|credential)" && echo "⚠️ Check for secrets!"

# Use git-secrets for automated scanning
git secrets --scan
```

### .env Files

Ensure `.env` files are in `.gitignore`:
```bash
# Verify .env is ignored
cat .gitignore | grep "\.env"

# Check for committed .env files
git ls-files | grep "\.env$" && echo "⚠️ .env file committed!"
```

## Maintenance

### Update Redaction Rules

When internal infrastructure changes:

1. **Edit `scripts/redact_for_public.py`**:
   ```python
   self.redaction_rules = {
       # Add new rule
       'new-internal-url.com': 'new-public-url.com',
       # ... existing rules ...
   }
   ```

2. **Test changes**:
   ```bash
   python scripts/redact_for_public.py --dry-run --verbose
   ```

3. **Commit updated script**:
   ```bash
   git add scripts/redact_for_public.py
   git commit -m "chore: update redaction rules for new infrastructure"
   git push origin main
   ```

### Regular Verification

Schedule regular verification of public repository:

```bash
# Weekly verification cron job
0 9 * * 1 /path/to/verify_public_repo.sh /tmp/public-clone && echo "✅ Weekly verification passed"
```

## Support

For issues with the redaction process:

1. Check this guide first
2. Review the troubleshooting section
3. Check the detailed redaction log (`redaction_changes.json`)
4. Verify redaction rules in `scripts/redact_for_public.py`

## See Also

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [CLAUDE.md](../CLAUDE.md) - Development guidance
- [README.md](../README.md) - Main project documentation
