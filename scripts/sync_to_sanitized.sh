#!/bin/bash
set -e

# Sync to Sanitized Working Directory and Public Repository
#
# This script handles the complete workflow:
# 1. Copy current repo to ../rag-templates-sanitized
# 2. Apply redaction to remove internal references
# 3. Optionally push to public GitHub repository
#
# Usage: ./scripts/sync_to_sanitized.sh [--push] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SANITIZED_DIR="${SANITIZED_DIR:-$REPO_ROOT/../rag-templates-sanitized}"

# Configuration
DRY_RUN=false
PUSH_TO_GITHUB=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --push)
            PUSH_TO_GITHUB=true
            shift
            ;;
        --sanitized-dir)
            SANITIZED_DIR="$2"
            shift 2
            ;;
        --help)
            cat << EOF
Sync to Sanitized Working Directory and Public Repository

Usage: $0 [OPTIONS]

Options:
  --dry-run           Preview changes without applying them
  --push              Push to public GitHub repository after sync
  --sanitized-dir DIR Custom sanitized directory path (default: ../rag-templates-sanitized)
  --help              Show this help message

Workflow:
  1. Verify sanitized directory exists
  2. Copy current repo to sanitized directory (excluding .git, .venv, etc.)
  3. Apply redaction to remove internal references
  4. Generate redaction report
  5. Optionally push to public GitHub repository

Examples:
  $0                    # Copy and redact only
  $0 --dry-run          # Preview what would be changed
  $0 --push             # Copy, redact, and push to GitHub
  $0 --sanitized-dir /tmp/public-repo  # Use custom directory

Environment Variables:
  SANITIZED_DIR       Override default sanitized directory path
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================="
echo "Sync to Sanitized Repository"
echo -e "==============================================${NC}"
echo "Source: $REPO_ROOT"
echo "Target: $SANITIZED_DIR"
echo "Dry run: $DRY_RUN"
echo "Push to GitHub: $PUSH_TO_GITHUB"
echo ""

# Step 1: Verify sanitized directory exists
if [[ ! -d "$SANITIZED_DIR" ]]; then
    echo -e "${RED}❌ Error: Sanitized directory not found: $SANITIZED_DIR${NC}"
    echo ""
    echo "To create it, run:"
    echo "  mkdir -p $SANITIZED_DIR"
    echo "  cd $SANITIZED_DIR"
    echo "  git init"
    echo "  git remote add origin git@github.com:intersystems-community/iris-rag-templates.git"
    exit 1
fi

if [[ ! -d "$SANITIZED_DIR/.git" ]]; then
    echo -e "${RED}❌ Error: Sanitized directory is not a git repository: $SANITIZED_DIR${NC}"
    echo ""
    echo "To initialize git, run:"
    echo "  cd $SANITIZED_DIR"
    echo "  git init"
    echo "  git remote add origin git@github.com:intersystems-community/iris-rag-templates.git"
    exit 1
fi

# Step 2: Get current branch name
CURRENT_BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
echo -e "${BLUE}📋 Current branch: $CURRENT_BRANCH${NC}"

# Step 3: Copy files to sanitized directory (excluding .git, .venv, etc.)
echo ""
echo -e "${BLUE}📂 Copying files to sanitized directory...${NC}"

if [[ "$DRY_RUN" == "false" ]]; then
    # Use rsync for efficient copying with excludes
    rsync -av --delete \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        --exclude='node_modules' \
        --exclude='.coverage' \
        --exclude='htmlcov' \
        --exclude='dist' \
        --exclude='build' \
        --exclude='*.egg-info' \
        --exclude='.DS_Store' \
        --exclude='redaction_changes.json' \
        "$REPO_ROOT/" "$SANITIZED_DIR/"
    echo -e "${GREEN}✓ Files copied successfully${NC}"
else
    echo -e "${YELLOW}[DRY RUN] Would copy files with rsync${NC}"
fi

# Step 4: Apply redaction
echo ""
echo -e "${BLUE}🔧 Applying redaction to remove internal references...${NC}"

REDACTION_LOG="$SANITIZED_DIR/redaction_changes.json"

if [[ "$DRY_RUN" == "true" ]]; then
    # Dry run - just show what would change
    python "$SCRIPT_DIR/redact_for_public.py" \
        --repo-root "$SANITIZED_DIR" \
        --dry-run \
        --log-file "$REDACTION_LOG"
else
    # Apply redaction
    python "$SCRIPT_DIR/redact_for_public.py" \
        --repo-root "$SANITIZED_DIR" \
        --log-file "$REDACTION_LOG"

    echo -e "${GREEN}✓ Redaction complete${NC}"
    echo -e "${BLUE}  Log file: $REDACTION_LOG${NC}"
fi

# Step 5: Show redaction summary
if [[ -f "$REDACTION_LOG" ]]; then
    echo ""
    echo -e "${BLUE}📊 Redaction Summary:${NC}"

    # Extract key stats from JSON log
    FILES_MODIFIED=$(cat "$REDACTION_LOG" | python -c "import sys, json; data = json.load(sys.stdin); print(len(data))")
    TOTAL_REPLACEMENTS=$(cat "$REDACTION_LOG" | python -c "import sys, json; data = json.load(sys.stdin); print(sum(item['replacements'] for item in data))")

    echo -e "  Files modified: ${GREEN}$FILES_MODIFIED${NC}"
    echo -e "  Total replacements: ${GREEN}$TOTAL_REPLACEMENTS${NC}"

    # Show top 5 most modified files
    echo ""
    echo -e "${BLUE}  Top modified files:${NC}"
    cat "$REDACTION_LOG" | python -c "
import sys, json
data = json.load(sys.stdin)
sorted_data = sorted(data, key=lambda x: x['replacements'], reverse=True)[:5]
for item in sorted_data:
    print(f\"    • {item['file']}: {item['replacements']} replacements\")
"
fi

# Step 6: Git status in sanitized directory
if [[ "$DRY_RUN" == "false" ]]; then
    echo ""
    echo -e "${BLUE}📝 Git status in sanitized repository:${NC}"
    cd "$SANITIZED_DIR"

    # Show what changed
    if git diff --quiet && git diff --cached --quiet; then
        echo -e "${GREEN}  ✓ No changes detected - already synchronized${NC}"
    else
        echo -e "${YELLOW}  Modified files:${NC}"
        git status --short | head -20

        if [[ $(git status --short | wc -l) -gt 20 ]]; then
            echo -e "${YELLOW}  ... and $(( $(git status --short | wc -l) - 20 )) more files${NC}"
        fi
    fi
fi

# Step 7: Optionally push to GitHub
if [[ "$PUSH_TO_GITHUB" == "true" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        echo -e "${YELLOW}[DRY RUN] Would commit and push changes to GitHub${NC}"
    else
        echo ""
        echo -e "${BLUE}📤 Committing and pushing to GitHub...${NC}"

        cd "$SANITIZED_DIR"

        # Stage all changes
        git add -A

        # Check if there are changes to commit
        if git diff --cached --quiet; then
            echo -e "${GREEN}  ✓ No changes to commit${NC}"
        else
            # Create commit
            COMMIT_MSG="chore: sync from internal repository ($(date -u +%Y-%m-%d))

Automated sync from internal repository with redaction applied.

Branch: $CURRENT_BRANCH
Sync date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Files modified: $FILES_MODIFIED
Redactions: $TOTAL_REPLACEMENTS

Changes:
- Redacted internal GitLab URLs → Public GitHub URLs
- Redacted internal Docker registry → Public Docker Hub
- Redacted internal email addresses
- Updated merge request references → pull request references
"

            git commit -m "$COMMIT_MSG"
            echo -e "${GREEN}  ✓ Changes committed${NC}"

            # Push to GitHub
            echo -e "${BLUE}  Pushing to origin/$CURRENT_BRANCH...${NC}"
            git push origin "$CURRENT_BRANCH"
            echo -e "${GREEN}  ✓ Pushed to GitHub${NC}"
        fi
    fi
fi

# Step 8: Final summary
echo ""
echo -e "${BLUE}=============================================="
echo "✅ Sync Complete!"
echo -e "==============================================${NC}"
echo "Sanitized repository: $SANITIZED_DIR"
echo "Branch: $CURRENT_BRANCH"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}This was a DRY RUN - no files were modified${NC}"
    echo "Run without --dry-run to apply changes"
fi

if [[ "$PUSH_TO_GITHUB" == "false" && "$DRY_RUN" == "false" ]]; then
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  cd $SANITIZED_DIR"
    echo "  git status          # Review changes"
    echo "  git add -A          # Stage changes"
    echo "  git commit -m 'your message'"
    echo "  git push origin $CURRENT_BRANCH"
    echo ""
    echo "Or run this script with --push to automatically commit and push"
fi

if [[ "$PUSH_TO_GITHUB" == "true" && "$DRY_RUN" == "false" ]]; then
    echo ""
    GITHUB_URL=$(cd "$SANITIZED_DIR" && git remote get-url origin | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
    echo -e "${GREEN}View on GitHub: $GITHUB_URL${NC}"
fi

echo ""
