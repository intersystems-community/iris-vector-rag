#!/bin/bash
set -e

# Sync to Public Repository with Redaction Script
# This script redacts internal references and syncs to the public-facing repository
#
# Usage: ./scripts/sync_to_public.sh [--dry-run] [--branch BRANCH]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
DRY_RUN=false
SOURCE_BRANCH=$(git rev-parse --abbrev-ref HEAD)
PUBLIC_REMOTE="public"
PUBLIC_BRANCH="${SOURCE_BRANCH}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --branch)
            PUBLIC_BRANCH="$2"
            shift 2
            ;;
        --help)
            cat << EOF
Sync to Public Repository with Redaction

Usage: $0 [OPTIONS]

Options:
  --dry-run       Show what would be changed without making changes
  --branch NAME   Target branch name for public repo (default: current branch)
  --help          Show this help message

Examples:
  $0 --dry-run                    # Preview changes
  $0 --branch main                # Sync to public main branch
  $0 --branch 047-create-a-unified  # Sync specific branch
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Sync to Public Repository"
echo "=============================================="
echo "Source branch: $SOURCE_BRANCH"
echo "Target branch: $PUBLIC_BRANCH"
echo "Dry run: $DRY_RUN"
echo ""

# Step 1: Check if public remote exists
if ! git remote | grep -q "^${PUBLIC_REMOTE}$"; then
    echo "❌ Error: Public remote '${PUBLIC_REMOTE}' not configured"
    echo ""
    echo "To add the public remote, run:"
    echo "  git remote add public git@github.com:intersystems-community/iris-rag-templates.git"
    echo ""
    echo "Or configure your public repository URL:"
    echo "  git remote add public <YOUR_PUBLIC_REPO_URL>"
    exit 1
fi

# Step 2: Create temporary working directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "📂 Creating temporary working directory: $TEMP_DIR"

# Step 3: Clone current repo to temp location
cd "$TEMP_DIR"
git clone "$REPO_ROOT" redacted-repo
cd redacted-repo

# Step 4: Checkout the source branch
git checkout "$SOURCE_BRANCH"

echo ""
echo "🔍 Scanning for internal references..."

# Redaction patterns
declare -A REDACTIONS=(
    # Internal GitLab URLs
    ["gitlab.iscinternal.com"]="github.com/intersystems-community"
    ["https://gitlab.iscinternal.com/tdyar/rag-templates"]="https://github.com/intersystems-community/iris-rag-templates"
    ["git@gitlab.iscinternal.com:tdyar/rag-templates.git"]="git@github.com:intersystems-community/iris-rag-templates.git"

    # Internal Docker registry
    ["docker.iscinternal.com/intersystems/iris"]="intersystemsdc/iris-community"

    # Internal merge request references
    ["https://gitlab.iscinternal.com/tdyar/rag-templates/-/merge_requests"]="https://github.com/intersystems-community/iris-rag-templates/pulls"

    # Internal email/username references (if any)
    ["tdyar@intersystems.com"]="maintainer@example.com"
)

# Step 5: Apply redactions to all files
echo ""
echo "🔧 Applying redactions..."

TOTAL_FILES=0
TOTAL_CHANGES=0

for pattern in "${!REDACTIONS[@]}"; do
    replacement="${REDACTIONS[$pattern]}"
    echo "  • Replacing: '$pattern' → '$replacement'"

    # Find and replace in all tracked files (excluding binary files)
    while IFS= read -r file; do
        if file "$file" | grep -q text; then
            if grep -q "$pattern" "$file" 2>/dev/null; then
                if [[ "$DRY_RUN" == "false" ]]; then
                    if [[ "$OSTYPE" == "darwin"* ]]; then
                        # macOS sed
                        sed -i '' "s|$pattern|$replacement|g" "$file"
                    else
                        # Linux sed
                        sed -i "s|$pattern|$replacement|g" "$file"
                    fi
                fi
                echo "    ✓ $file"
                ((TOTAL_FILES++))
                TOTAL_CHANGES=$((TOTAL_CHANGES + $(grep -o "$pattern" "$file" | wc -l)))
            fi
        fi
    done < <(git ls-files)
done

echo ""
echo "📊 Redaction Summary:"
echo "  • Files modified: $TOTAL_FILES"
echo "  • Total replacements: $TOTAL_CHANGES"

# Step 6: Show git diff for review
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "=============================================="
    echo "DRY RUN - Preview of Changes"
    echo "=============================================="
    git diff --stat
    echo ""
    echo "Run without --dry-run to apply changes and push to public repository"
    exit 0
fi

# Step 7: Commit redacted changes if any
if ! git diff --quiet; then
    echo ""
    echo "💾 Committing redacted changes..."
    git add -A
    git commit -m "chore: redact internal references for public repository

- Replaced internal GitLab URLs with public GitHub URLs
- Replaced internal Docker registry with public registry
- Redacted internal email/username references

This commit prepares the codebase for public release while preserving
all functionality and documentation."
else
    echo ""
    echo "✅ No changes needed - already redacted or no internal references found"
fi

# Step 8: Push to public remote
echo ""
echo "📤 Pushing to public repository..."
echo "  Remote: $PUBLIC_REMOTE"
echo "  Branch: $PUBLIC_BRANCH"

cd "$REPO_ROOT"
git remote update "$PUBLIC_REMOTE"

# Create a new branch for the redacted version
REDACTED_BRANCH="${PUBLIC_BRANCH}-redacted-$(date +%Y%m%d-%H%M%S)"

# Copy the redacted changes back to main repo
git checkout -b "$REDACTED_BRANCH"
rsync -av --exclude='.git' "$TEMP_DIR/redacted-repo/" "$REPO_ROOT/"

# Commit and push
git add -A
if ! git diff --cached --quiet; then
    git commit -m "chore: sync from internal repository with redaction

Automated sync from internal repository with the following redactions:
- Internal GitLab URLs → Public GitHub URLs
- Internal Docker registry → Public Docker Hub
- Internal references → Public references

Source branch: $SOURCE_BRANCH
Sync date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
"
fi

git push "$PUBLIC_REMOTE" "$REDACTED_BRANCH:$PUBLIC_BRANCH"

echo ""
echo "=============================================="
echo "✅ Sync Complete!"
echo "=============================================="
echo "Public repository: $(git remote get-url $PUBLIC_REMOTE)"
echo "Branch: $PUBLIC_BRANCH"
echo ""
echo "Next steps:"
echo "1. Review the changes on GitHub"
echo "2. Create a pull request if needed"
echo "3. Clean up temporary branch: git branch -D $REDACTED_BRANCH"
echo ""
