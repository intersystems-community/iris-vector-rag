# Public Repository Sync - Quick Reference

**Last Updated**: 2025-10-14

This guide provides quick-reference commands for syncing the internal `rag-templates` repository to the public-facing `rag-templates-sanitized` working directory and GitHub.

## Quick Commands

### Preview Changes (Recommended First Step)

```bash
# See what would be redacted without making changes
./scripts/sync_to_sanitized.sh --dry-run
```

### Standard Sync (Copy + Redact)

```bash
# Copy current repo to ../rag-templates-sanitized and apply redaction
./scripts/sync_to_sanitized.sh
```

### Full Sync (Copy + Redact + Push to GitHub)

```bash
# Copy, redact, and automatically push to GitHub
./scripts/sync_to_sanitized.sh --push
```

### Custom Sanitized Directory

```bash
# Use a different target directory
./scripts/sync_to_sanitized.sh --sanitized-dir /path/to/public-repo
```

## Workflow

### Daily Development Workflow

```bash
# 1. Work on internal repository as usual
git checkout -b feature/my-feature
# ... make changes ...
git commit -m "feat: implement my feature"
git push origin feature/my-feature

# 2. When ready to sync to public repository
./scripts/sync_to_sanitized.sh --dry-run  # Preview changes
./scripts/sync_to_sanitized.sh            # Apply sync

# 3. Review changes in sanitized repo
cd ../rag-templates-sanitized
git status
git diff

# 4. Commit and push to GitHub
git add -A
git commit -m "feat: implement my feature"
git push origin feature/my-feature
```

### One-Command Sync

```bash
# For trusted changes that don't need manual review
./scripts/sync_to_sanitized.sh --push
```

## What Gets Redacted

| Internal Reference | Public Replacement |
|-------------------|--------------------|
| `github.com/intersystems-community` | `github.com/intersystems-community` |
| `intersystemsdc/iris-community` | `intersystemsdc/iris-community` |
| `maintainer@example.com` | `maintainer@example.com` |
| `pull request` / `MR` | `pull request` / `PR` |
| `/intersystems-community/` | `/intersystems-community/` |

## Files Excluded from Sync

The following are automatically excluded:
- `.git/` - Git repository metadata
- `.venv/` - Virtual environment
- `__pycache__/`, `*.pyc` - Python cache files
- `.pytest_cache/` - Pytest cache
- `node_modules/` - Node.js dependencies
- `.coverage`, `htmlcov/` - Coverage reports
- `dist/`, `build/`, `*.egg-info/` - Build artifacts
- `.DS_Store` - macOS metadata
- `redaction_changes.json` - Redaction logs

## Troubleshooting

### Sanitized directory doesn't exist

```bash
# Create and initialize sanitized directory
mkdir -p ../rag-templates-sanitized
cd ../rag-templates-sanitized
git init
git remote add origin git@github.com:intersystems-community/iris-rag-templates.git
cd ../rag-templates
```

### Permission denied pushing to GitHub

```bash
# Verify SSH key is configured
ssh -T git@github.com

# If needed, add your SSH key to GitHub:
# https://github.com/settings/keys
```

### Need to see detailed redaction log

```bash
# After running sync, check the log file
cat ../rag-templates-sanitized/redaction_changes.json | python -m json.tool | less
```

### Want to verify no internal references remain

```bash
# Search for internal references in sanitized repo
cd ../rag-templates-sanitized
grep -r "iscinternal" . --exclude-dir=.git || echo "✅ No internal URLs"
grep -r "@intersystems.com" . --exclude-dir=.git || echo "✅ No internal emails"
```

## Scripts Reference

### sync_to_sanitized.sh

Main sync script that handles copy, redaction, and optional push.

**Options**:
- `--dry-run` - Preview changes without applying
- `--push` - Automatically push to GitHub after sync
- `--sanitized-dir DIR` - Use custom sanitized directory
- `--help` - Show help message

**Examples**:
```bash
./scripts/sync_to_sanitized.sh --dry-run
./scripts/sync_to_sanitized.sh --push
./scripts/sync_to_sanitized.sh --sanitized-dir /tmp/public-repo
```

### redact_for_public.py

Python script for detailed redaction with logging.

**Options**:
- `--repo-root DIR` - Repository to redact (default: current directory)
- `--dry-run` - Preview changes
- `--backup` - Create backup before redaction
- `--backup-dir DIR` - Custom backup directory
- `--log-file FILE` - Output log file path
- `--verbose` - Enable verbose output

**Examples**:
```bash
python scripts/redact_for_public.py --dry-run --verbose
python scripts/redact_for_public.py --backup --backup-dir /tmp/backup
python scripts/redact_for_public.py --log-file redaction-$(date +%Y%m%d).json
```

## Advanced Usage

### Sync Specific Branch

```bash
# Checkout the branch you want to sync
git checkout feature/my-feature

# Sync to sanitized repo (preserves branch name)
./scripts/sync_to_sanitized.sh --push
```

### Multiple Sanitized Repos

```bash
# Sync to different public repos
./scripts/sync_to_sanitized.sh --sanitized-dir ../rag-templates-github
./scripts/sync_to_sanitized.sh --sanitized-dir ../rag-templates-gitlab-public
```

### Redact Only (No Copy)

```bash
# Apply redaction to existing sanitized repo without copying
cd ../rag-templates-sanitized
python ../rag-templates/scripts/redact_for_public.py
```

### Create Redaction Backup

```bash
# Backup before redaction (safety measure)
python scripts/redact_for_public.py \
    --repo-root ../rag-templates-sanitized \
    --backup \
    --backup-dir /tmp/sanitized-backup-$(date +%Y%m%d)
```

## Verification Checklist

After syncing to public repository, verify:

- [ ] No internal GitLab URLs (`grep -r "iscinternal"`)
- [ ] No internal Docker registry references
- [ ] No internal email addresses
- [ ] All tests pass (`make test`)
- [ ] Documentation is complete
- [ ] README links work
- [ ] No sensitive data or API keys

## See Also

- [PUBLIC_REPOSITORY_SYNC.md](../docs/PUBLIC_REPOSITORY_SYNC.md) - Detailed documentation
- [CLAUDE.md](../CLAUDE.md) - Development guidance
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

## Environment Variables

### SANITIZED_DIR

Override the default sanitized directory location:

```bash
# Use custom directory
export SANITIZED_DIR=/path/to/public-repo
./scripts/sync_to_sanitized.sh
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Sync to Public Repository

on:
  push:
    branches: [ main ]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Sync to Public
        run: |
          ./scripts/sync_to_sanitized.sh --push
        env:
          SANITIZED_DIR: ${{ secrets.PUBLIC_REPO_PATH }}
```

### Pre-Push Hook

Install a pre-push hook to remind about syncing:

```bash
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "⚠️  Remember to sync to public repository!"
echo "Run: ./scripts/sync_to_sanitized.sh --push"
EOF

chmod +x .git/hooks/pre-push
```

## Support

For issues with syncing:

1. Check this quick reference
2. Review [PUBLIC_REPOSITORY_SYNC.md](../docs/PUBLIC_REPOSITORY_SYNC.md)
3. Verify sanitized directory exists and is a git repo
4. Check GitHub SSH access (`ssh -T git@github.com`)

---

**Quick Reference Card**

| Command | What it does |
|---------|-------------|
| `./scripts/sync_to_sanitized.sh --dry-run` | Preview changes |
| `./scripts/sync_to_sanitized.sh` | Copy + redact |
| `./scripts/sync_to_sanitized.sh --push` | Copy + redact + push |
| `python scripts/redact_for_public.py --dry-run` | Preview redaction only |

**Remember**: Always run `--dry-run` first to preview changes!
