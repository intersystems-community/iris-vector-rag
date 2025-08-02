#!/bin/bash
# Script to remove internal files from git history in sanitized repository

set -e

SANITIZED_DIR="../rag-templates-sanitized"

echo "🧹 Removing internal files from sanitized repository history..."
echo ""
echo "This script will remove:"
echo "  - CLAUDE.md"
echo "  - .clinerules"
echo "  - docs/CRITICAL_SECURITY_AUDIT_REPORT.md"
echo ""

# Check if we're in the right place
if [ ! -d "$SANITIZED_DIR/.git" ]; then
    echo "❌ Error: $SANITIZED_DIR is not a git repository"
    exit 1
fi

cd "$SANITIZED_DIR"

echo "📍 Current branch: $(git branch --show-current)"
echo ""

# Method 1: Using git filter-repo (recommended if available)
if command -v git-filter-repo &> /dev/null; then
    echo "✅ Using git filter-repo (recommended method)"
    
    # Create a backup tag before we start
    git tag backup-before-scrub-$(date +%Y%m%d-%H%M%S)
    
    # Remove the files from all history
    git filter-repo --path CLAUDE.md --invert-paths --force
    git filter-repo --path .clinerules --invert-paths --force
    git filter-repo --path docs/CRITICAL_SECURITY_AUDIT_REPORT.md --invert-paths --force
    
    echo "✅ Files removed from history using git filter-repo"
    
else
    echo "⚠️  git filter-repo not found. Using git filter-branch (slower method)"
    echo ""
    echo "To install git filter-repo:"
    echo "  brew install git-filter-repo  # on macOS"
    echo "  pip install git-filter-repo   # with Python"
    echo ""
    read -p "Continue with git filter-branch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create a backup tag
        git tag backup-before-scrub-$(date +%Y%m%d-%H%M%S)
        
        # Remove files using filter-branch
        git filter-branch --force --index-filter \
            'git rm --cached --ignore-unmatch CLAUDE.md .clinerules docs/CRITICAL_SECURITY_AUDIT_REPORT.md' \
            --prune-empty --tag-name-filter cat -- --all
        
        # Clean up refs
        git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d
        
        echo "✅ Files removed from history using git filter-branch"
    else
        echo "❌ Aborted"
        exit 1
    fi
fi

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo "🎯 Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. Add new remote if needed: git remote add origin-clean <url>"
echo "3. Force push ALL branches: git push --force --all"
echo "4. Force push tags: git push --force --tags"
echo "5. Delete backup tags when confirmed: git tag -d backup-before-scrub-*"
echo ""
echo "⚠️  WARNING: This rewrites history! All collaborators need to re-clone."